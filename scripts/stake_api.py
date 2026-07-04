#!/usr/bin/env python3
"""
Stake.com GraphQL API Client
=============================

A client for Stake.com's GraphQL API that handles:
- Fixture market fetching (FixtureIndex query)
- Outcome ID lookup for any market/selection
- Multibet validation (customBetOdds query)
- Bet placement (BetSlipFooter_CustomSportBet mutation)
- Single bet placement (same mutation with one outcome)

KASADA BOT PROTECTION — READ THIS
---------------------------------
Stake.com uses Kasada for bot protection. Every request requires
4 dynamically-generated headers (x-kpsdk-cd, x-kpsdk-ct, x-kpsdk-h,
x-kpsdk-v) that CANNOT be statically copied. They must be solved
fresh for each request using one of:

  1. Manual session harvest (default, free, requires periodic refresh)
  2. Stealth browser automation via Playwright (autonomous, heavy)
  3. Third-party solver service (autonomous, paid)

This client uses a pluggable KasadaHeaderProvider interface so you
can swap providers without touching the API logic.

USAGE
-----
    from stake_api import StakeAPIClient, ManualSessionProvider

    provider = ManualSessionProvider.from_file("stake_session.json")
    client = StakeAPIClient(provider)

    # 1. Get markets for a fixture
    markets = client.get_fixture_markets("46602331-w73-w75")

    # 2. Find the outcome ID for "Over 2.5" in the "Goals" market
    outcome_id = client.find_outcome_id(markets, "Total Goals", "Over 2.5")

    # 3. Validate a multibet (3 selections)
    result = client.validate_multibet([id1, id2, id3])
    if result["compatible"]:
        print(f"Combined odds: {result['odds']}")

    # 4. Place the bet
    bet = client.place_multibet(amount=1.0, currency="usdt",
                                 outcome_ids=[id1, id2, id3])
    print(f"Bet placed: {bet['id']}")
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# --------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------
STAKE_GRAPHQL_URL = "https://stake.com/_api/graphql"
KASADA_VERSION = "j-1.2.543"   # captured from recent traffic; may change

# Standard market groups to fetch (covers all common bet types)
DEFAULT_MARKET_GROUPS = [
    "main", "goals", "Overtime", "AsianLines",
    "1st2ndhalfmarkets", "goalscorers", "Players",
    "Sure Sub", "CardsCorners", "specials",
]


# --------------------------------------------------------------------------
# Kasada header providers (pluggable)
# --------------------------------------------------------------------------
class KasadaHeaderProvider(ABC):
    """Abstract base for Kasada header providers."""

    @abstractmethod
    def get_headers(self) -> Dict[str, str]:
        """Return the 4 x-kpsdk-* headers + session token."""
        pass

    @abstractmethod
    def is_valid(self) -> bool:
        """Return True if the headers are still valid (not expired)."""
        pass


class ManualSessionProvider(KasadaHeaderProvider):
    """Reads Kasada headers + session token from a JSON file.

    The user manually harvests these from their browser (via devtools or
    a browser extension) and saves them to stake_session.json.

    Headers expire after ~10-30 minutes (Kasada tokens are short-lived),
    so this requires periodic manual refresh. NOT suitable for fully
    autonomous operation.
    """

    def __init__(self, session_token: str, kpsdk_headers: Dict[str, str],
                 cf_clearance: str = "", harvested_at: Optional[float] = None):
        self.session_token = session_token
        self.kpsdk_headers = kpsdk_headers
        self.cf_clearance = cf_clearance
        self.harvested_at = harvested_at or time.time()

    @classmethod
    def from_file(cls, path: str | Path) -> "ManualSessionProvider":
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(
                f"Session file not found: {p}\n"
                f"Create it from stake_session.example.json"
            )
        data = json.loads(p.read_text())
        return cls(
            session_token=data["session_token"],
            kpsdk_headers=data["kpsdk_headers"],
            cf_clearance=data.get("cf_clearance", ""),
            harvested_at=data.get("harvested_at", time.time()),
        )

    def get_headers(self) -> Dict[str, str]:
        headers = {
            "x-access-token": self.session_token,
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                           "AppleWebKit/537.36 (KHTML, like Gecko) "
                           "Chrome/120.0.0.0 Safari/537.36"),
            "Origin": "https://stake.com",
            "Referer": "https://stake.com/sports",
        }
        headers.update(self.kpsdk_headers)
        return headers

    def get_cookies(self) -> str:
        cookies = [f"session={self.session_token}"]
        if self.cf_clearance:
            cookies.append(f"cf_clearance={self.cf_clearance}")
        return "; ".join(cookies)

    def is_valid(self) -> bool:
        # Kasada headers typically expire after ~15-30 min
        age = time.time() - self.harvested_at
        return age < 900   # 15 min conservative TTL


class BrowserSessionProvider(KasadaHeaderProvider):
    """Uses Playwright to load Stake.com in a real browser and harvest
    Kasada headers automatically. Heavier but fully autonomous.

    Requires: pip install playwright && playwright install chromium
    """

    def __init__(self, session_token: str, headless: bool = True):
        self.session_token = session_token
        self.headless = headless
        self._cached_headers: Optional[Dict[str, str]] = None
        self._cached_at: float = 0

    def _harvest_headers(self) -> Dict[str, str]:
        try:
            from playwright.sync_api import sync_playwright
        except ImportError:
            raise RuntimeError(
                "BrowserSessionProvider requires playwright. Install with:\n"
                "  pip install playwright && playwright install chromium"
            )
        headers = {}
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=self.headless)
            context = browser.new_context()
            # Inject session cookie
            context.add_cookies([{
                "name": "session",
                "value": self.session_token,
                "domain": ".stake.com",
                "path": "/",
            }])
            page = context.new_page()
            # Capture requests to extract Kasada headers
            captured = {}

            def on_request(request):
                if "graphql" in request.url and not captured:
                    h = request.headers
                    for key in ["x-kpsdk-cd", "x-kpsdk-ct", "x-kpsdk-h", "x-kpsdk-v"]:
                        if key in h:
                            captured[key] = h[key]

            page.on("request", on_request)
            try:
                page.goto("https://stake.com/sports", wait_until="networkidle", timeout=30000)
                page.wait_for_timeout(3000)   # let Kasada settle
            except Exception as e:
                pass
            browser.close()
            return captured

    def get_headers(self) -> Dict[str, str]:
        if self._cached_headers is None or not self.is_valid():
            self._cached_headers = self._harvest_headers()
            self._cached_at = time.time()
        headers = {
            "x-access-token": self.session_token,
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        headers.update(self._cached_headers)
        return headers

    def get_cookies(self) -> str:
        return f"session={self.session_token}"

    def is_valid(self) -> bool:
        return (time.time() - self._cached_at) < 900   # 15 min TTL


# --------------------------------------------------------------------------
# Stake API client
# --------------------------------------------------------------------------
@dataclass
class Outcome:
    id: str
    name: str
    odds: float
    active: bool
    market_id: str
    market_name: str


@dataclass
class BetResult:
    id: str
    amount: float
    currency: str
    potential_multiplier: float
    custom_prices: List[Dict[str, Any]] = field(default_factory=list)
    raw: Dict[str, Any] = field(default_factory=dict)


class StakeAPIClient:
    """High-level Stake.com GraphQL API client."""

    def __init__(self, kasada_provider: KasadaHeaderProvider,
                 verbose: bool = False):
        self.provider = kasada_provider
        self.verbose = verbose
        self._rate_limit_delay = 1.0   # seconds between requests

    # ----------------------------------------------------------------------
    # Low-level HTTP
    # ----------------------------------------------------------------------
    def _request(self, operation_name: str, operation_type: str,
                 query: str, variables: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a GraphQL request with Kasada headers."""
        if not self.provider.is_valid():
            if self.verbose:
                print(f"  [Stake] Kasada headers expired, refreshing...")
            # Force refresh by clearing cache (BrowserSessionProvider auto-refreshes)
            if hasattr(self.provider, "_cached_headers"):
                self.provider._cached_headers = None

        headers = self.provider.get_headers()
        headers["x-operation-name"] = operation_name
        headers["x-operation-type"] = operation_type
        headers["Cookie"] = self.provider.get_cookies()

        body = json.dumps({
            "query": query,
            "variables": variables,
            "operationName": operation_name,
        }).encode("utf-8")

        req = urllib.request.Request(STAKE_GRAPHQL_URL, data=body, headers=headers, method="POST")

        # Retry with exponential backoff
        last_err = None
        for attempt in range(3):
            try:
                with urllib.request.urlopen(req, timeout=15) as resp:
                    result = json.loads(resp.read().decode("utf-8"))
                if "errors" in result:
                    raise RuntimeError(f"GraphQL errors: {result['errors']}")
                time.sleep(self._rate_limit_delay)
                return result.get("data", {})
            except urllib.error.HTTPError as e:
                last_err = e
                if e.code == 403:
                    raise RuntimeError(
                        f"403 Forbidden — Kasada headers likely invalid or expired. "
                        f"Refresh your session file."
                    )
                if e.code == 429:
                    wait = 2 ** (attempt + 1)
                    if self.verbose:
                        print(f"  [Stake] 429 rate limited, waiting {wait}s ...")
                    time.sleep(wait)
                    continue
                raise
            except (urllib.error.URLError, TimeoutError) as e:
                last_err = e
                time.sleep(2 * (attempt + 1))
        raise RuntimeError(f"Stake API request failed after 3 retries: {last_err}")

    # ----------------------------------------------------------------------
    # Step 1: Fetch fixture markets
    # ----------------------------------------------------------------------
    def get_fixture_markets(self, fixture_slug: str,
                            groups: Optional[List[str]] = None) -> Dict[str, Any]:
        """Fetch all markets for a fixture.

        Args:
            fixture_slug: e.g. "46602331-w73-w75" (from Stake URL)
            groups: market groups to fetch (defaults to all common groups)

        Returns:
            Parsed fixture data with markets and outcomes.
        """
        if groups is None:
            groups = DEFAULT_MARKET_GROUPS

        query = """
        query FixtureIndex($fixture: String!, $groups: [String!]!) {
          slugFixture(fixture: $fixture) {
            id
            status
            group: groups(groups: $groups) {
              name
              translation
              templates(includeEmpty: false) {
                id
                name
                markets {
                  id
                  name
                  status
                  specifiers
                  outcomes {
                    id
                    active
                    odds
                    name
                  }
                }
              }
            }
          }
        }
        """
        data = self._request("FixtureIndex", "query", query, {
            "fixture": fixture_slug,
            "groups": groups,
        })
        return data.get("slugFixture", {})

    # ----------------------------------------------------------------------
    # Helper: find outcome ID by market + selection name
    # ----------------------------------------------------------------------
    @staticmethod
    def find_outcome(fixture_data: Dict[str, Any],
                     market_name: str, selection_name: str,
                     specifier: Optional[str] = None) -> Optional[Outcome]:
        """Find a specific outcome in the fixture data.

        Args:
            fixture_data: result from get_fixture_markets
            market_name: e.g. "Total Goals", "1X2", "Both Teams to Score"
            selection_name: e.g. "Over 2.5", "Home", "Yes"
            specifier: optional market specifier (e.g. "2.5" for O/U 2.5)

        Returns:
            Outcome object or None if not found.
        """
        market_name_lower = market_name.lower()
        selection_name_lower = selection_name.lower()

        for group in fixture_data.get("group", []):
            for template in group.get("templates", []):
                for market in template.get("markets", []):
                    m_name = market.get("name", "").lower()
                    if market_name_lower not in m_name and m_name not in market_name_lower:
                        continue
                    if specifier and specifier not in (market.get("specifiers") or ""):
                        continue
                    for outcome in market.get("outcomes", []):
                        o_name = outcome.get("name", "").lower()
                        if selection_name_lower in o_name or o_name in selection_name_lower:
                            return Outcome(
                                id=outcome["id"],
                                name=outcome["name"],
                                odds=float(outcome.get("odds", 0)),
                                active=outcome.get("active", False),
                                market_id=market["id"],
                                market_name=market["name"],
                            )
        return None

    @staticmethod
    def list_markets(fixture_data: Dict[str, Any]) -> List[Dict[str, str]]:
        """List all available markets in a fixture (for debugging)."""
        markets = []
        for group in fixture_data.get("group", []):
            for template in group.get("templates", []):
                for market in template.get("markets", []):
                    markets.append({
                        "group": group.get("name"),
                        "market_name": market.get("name"),
                        "specifiers": market.get("specifiers"),
                        "outcome_count": len(market.get("outcomes", [])),
                    })
        return markets

    # ----------------------------------------------------------------------
    # Step 2: Validate multibet and get combined odds
    # ----------------------------------------------------------------------
    def validate_multibet(self, outcome_ids: List[str]) -> Dict[str, Any]:
        """Validate a multibet and get combined odds.

        Returns:
            {
                "odds": float,           # combined decimal odds
                "compatible": bool,      # True if no conflicts
                "conflicts": [...],      # list of conflicting selections
            }
        """
        query = """
        query customBetOdds($selections: [String!]!) {
          customBetOdds(selections: $selections) {
            odds
            compatibleSelections {
              conflict
              marketId
              specifiers
              outcomes {
                conflict
                id
              }
            }
          }
        }
        """
        data = self._request("customBetOdds", "query", query, {
            "selections": outcome_ids,
        })
        result = data.get("customBetOdds", {})
        odds = float(result.get("odds", 0))
        compat = result.get("compatibleSelections", [])
        conflicts = [
            {"market_id": c.get("marketId"),
             "conflicting_outcomes": [o["id"] for o in c.get("outcomes", []) if o.get("conflict")]}
            for c in compat if c.get("conflict")
        ]
        return {
            "odds": odds,
            "compatible": len(conflicts) == 0 and odds > 0,
            "conflicts": conflicts,
            "raw": result,
        }

    # ----------------------------------------------------------------------
    # Step 3: Place the bet (single or multibet)
    # ----------------------------------------------------------------------
    def place_multibet(self, amount: float, currency: str,
                       outcome_ids: List[str],
                       odds_change: str = "any") -> BetResult:
        """Place a multibet (or single bet if only one outcome).

        Args:
            amount: stake amount (e.g. 1.0 for 1 USDT)
            currency: "usdt", "btc", "eth", etc.
            outcome_ids: list of outcome IDs (1 = single, 2+ = multibet)
            odds_change: "any" (accept any odds) or "accept" (exact only)

        Returns:
            BetResult with bet ID and confirmation.
        """
        mutation = """
        mutation BetSlipFooter_CustomSportBet(
          $amount: Float!,
          $currency: CurrencyEnum!,
          $outcomeIds: [String!]!,
          $oddsChange: SportOddsChangeEnum!,
          $identifier: String,
          $sessionIds: [CustomSportBetSessionIdsInput!]
        ) {
          customSportBet(
            amount: $amount
            currency: $currency
            outcomeIds: $outcomeIds
            oddsChange: $oddsChange
            identifier: $identifier
            sessionIds: $sessionIds
          ) {
            id
            amount
            currency
            potentialMultiplier
            customPrices {
              customOdds
              type
              promotion {
                id
                name
              }
            }
          }
        }
        """
        data = self._request("BetSlipFooter_CustomSportBet", "mutation", mutation, {
            "amount": amount,
            "currency": currency,
            "outcomeIds": outcome_ids,
            "oddsChange": odds_change,
            "identifier": "",
        })
        bet_data = data.get("customSportBet", {})
        return BetResult(
            id=bet_data.get("id", ""),
            amount=float(bet_data.get("amount", 0)),
            currency=bet_data.get("currency", currency),
            potential_multiplier=float(bet_data.get("potentialMultiplier", 0)),
            custom_prices=bet_data.get("customPrices", []),
            raw=bet_data,
        )

    # Convenience: place a single bet
    def place_single_bet(self, amount: float, currency: str,
                          outcome_id: str,
                          odds_change: str = "any") -> BetResult:
        """Place a single-outcome bet (wrapper around place_multibet)."""
        return self.place_multibet(amount, currency, [outcome_id], odds_change)


# --------------------------------------------------------------------------
# CLI for testing
# --------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Stake.com API client test")
    ap.add_argument("--session-file", default="stake_session.json",
                    help="Path to session JSON file (default: stake_session.json)")
    ap.add_argument("--fixture", help="Fixture slug to fetch markets for")
    ap.add_argument("--list-markets", action="store_true",
                    help="Just list all markets for the fixture")
    ap.add_argument("--place-test-bet", action="store_true",
                    help="Place a 0.01 USDT test bet on the first outcome found")
    args = ap.parse_args()

    try:
        provider = ManualSessionProvider.from_file(args.session_file)
    except FileNotFoundError as e:
        print(e)
        print("\nCreate stake_session.json from stake_session.example.json")
        sys.exit(1)

    client = StakeAPIClient(provider, verbose=True)

    if not args.fixture:
        print("Use --fixture <slug> to fetch markets")
        print("Example slug: 46602331-w73-w75")
        sys.exit(0)

    print(f"\nFetching markets for fixture: {args.fixture}")
    fixture = client.get_fixture_markets(args.fixture)
    print(f"Fixture status: {fixture.get('status')}")

    if args.list_markets:
        markets = client.list_markets(fixture)
        print(f"\n{len(markets)} markets available:")
        for m in markets[:20]:
            print(f"  [{m['group']}] {m['market_name']} (specifiers: {m['specifiers']}) — {m['outcome_count']} outcomes")
        if len(markets) > 20:
            print(f"  ... and {len(markets) - 20} more")
        sys.exit(0)

    # Find a sample outcome
    sample = client.find_outcome(fixture, "Total Goals", "Over 2.5")
    if not sample:
        print("\nCouldn't find Over 2.5 — trying 'Over 1.5'")
        sample = client.find_outcome(fixture, "Total Goals", "Over 1.5")
    if sample:
        print(f"\nFound outcome: {sample.name} @ {sample.odds} (id: {sample.id})")

        if args.place_test_bet:
            confirm = input(f"\nPlace 0.01 USDT test bet on '{sample.name}' @ {sample.odds}? [y/N] ")
            if confirm.lower() == "y":
                bet = client.place_single_bet(0.01, "usdt", sample.id)
                print(f"\n✓ Bet placed!")
                print(f"  Bet ID: {bet.id}")
                print(f"  Amount: {bet.amount} {bet.currency}")
                print(f"  Potential multiplier: {bet.potential_multiplier}")
