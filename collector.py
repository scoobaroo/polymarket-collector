"""
Polymarket Data Collector
=========================
Standalone service that polls Polymarket CLOB, Chainlink oracle, and Binance spot
at regular intervals, logging snapshots to CSV for ML training data.

Collects:
- Polymarket UpDown 5m market orderbook (midpoint, depth)
- Chainlink on-chain oracle price (what actually settles UpDown markets)
- Binance BTC spot price (exchange VWAP proxy)
- Oracle-vs-exchange spread (key signal for UpDown mispricing)

Usage:
    python collector.py
    python collector.py --interval 30 --asset btc
    python collector.py --polygon-rpc https://polygon-mainnet.g.alchemy.com/v2/KEY

Runs indefinitely. Ctrl+C to stop. Data appends to data/market_context_log.csv.
"""
import argparse
import csv
import json
import logging
import os
import signal
import time
from pathlib import Path
from typing import Optional

import requests

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

# --- API endpoints ---
GAMMA_API_URL = "https://gamma-api.polymarket.com"
CLOB_BASE_URL = "https://clob.polymarket.com"
BINANCE_TICKER_URL = "https://api.binance.com/api/v3/ticker/bookTicker"

# --- Output ---
DATA_DIR = Path(__file__).parent / "data"
CSV_PATH = DATA_DIR / "market_context_log.csv"
CSV_FIELDS = [
    "timestamp", "poly_midpoint", "poly_best_bid", "poly_best_ask", "poly_spread",
    "poly_depth_bid", "poly_depth_ask",
    "chainlink_oracle_price", "binance_price", "oracle_vs_binance_spread",
    "oracle_price", "vwap", "cross_exchange_spread", "realtime_volatility",
    "token_id", "market_question", "window_start",
]

# --- Asset config ---
# UpDown slug prefixes must match Polymarket's naming convention
UPDOWN_ASSETS = {
    "btc": ("btc", "BTCUSDT"),
    "eth": ("eth", "ETHUSDT"),
}

# Chainlink price feed contracts on Polygon
# Same addresses as pmbot-rust/src/exchanges/chainlink.rs
CHAINLINK_FEEDS = {
    "btc": "0xc907E116054Ad103354f2D350FD2514433D57F6f",  # BTC/USD
    "eth": "0xF9680D99D6C9589e2a93a78A04A279e509205945",  # ETH/USD
}
CHAINLINK_DECIMALS = 8
LATEST_ROUND_DATA_SELECTOR = "0xfeaf968c"  # keccak256("latestRoundData()")[:4]

_shutdown = False


def signal_handler(sig, frame):
    global _shutdown
    logger.info("Shutdown requested")
    _shutdown = True


# ---------------------------------------------------------------------------
# Polymarket
# ---------------------------------------------------------------------------

def current_5m_window_start() -> int:
    """Return Unix timestamp of the current 5-minute window start."""
    now = int(time.time())
    return now - (now % 300)


def discover_updown_token(session: requests.Session, asset: str) -> Optional[dict]:
    """Find the active UpDown 5m market for the given asset via Gamma API."""
    slug_prefix = UPDOWN_ASSETS.get(asset, (asset, ""))[0]
    window_start = current_5m_window_start()
    slug = f"{slug_prefix}-updown-5m-{window_start}"

    try:
        resp = session.get(
            f"{GAMMA_API_URL}/events",
            params={"slug": slug},
            timeout=10,
        )
        if resp.status_code != 200:
            logger.warning("Gamma API %d for slug=%s", resp.status_code, slug)
            return None

        events = resp.json()
        if not events:
            return None

        for event in events:
            markets = event.get("markets", [])
            if not markets:
                continue
            market = markets[0]
            raw_tokens = market.get("clobTokenIds", [])
            # Gamma API returns clobTokenIds as a JSON-encoded string
            if isinstance(raw_tokens, str):
                raw_tokens = json.loads(raw_tokens)
            if not raw_tokens:
                continue
            return {
                "token_id": raw_tokens[0],  # Yes token for "Up"
                "question": market.get("question", ""),
                "window_start": window_start,
            }
    except Exception as e:
        logger.warning("Gamma API error: %s", e)

    return None


def fetch_orderbook(session: requests.Session, token_id: str) -> Optional[dict]:
    """Fetch CLOB orderbook + midpoint for a token. No auth required.

    Uses /midpoint as the primary implied probability (more reliable than
    sparse book bids on short-lived UpDown markets), plus /book for depth.
    """
    result = {}

    # Midpoint gives the most accurate implied probability
    try:
        resp = session.get(
            f"{CLOB_BASE_URL}/midpoint",
            params={"token_id": token_id},
            timeout=10,
        )
        if resp.status_code == 200:
            mid = float(resp.json().get("mid", 0))
            result["midpoint"] = mid
    except Exception as e:
        logger.debug("CLOB midpoint error: %s", e)

    # Full book for depth and spread
    try:
        resp = session.get(
            f"{CLOB_BASE_URL}/book",
            params={"token_id": token_id},
            timeout=10,
        )
        if resp.status_code != 200:
            logger.warning("CLOB book %d for token=%s", resp.status_code, token_id)
            if "midpoint" not in result:
                return None
            return {
                "best_bid": result["midpoint"],
                "best_ask": result["midpoint"],
                "spread": 0.0,
                "depth_bid": 0.0,
                "depth_ask": 0.0,
                "midpoint": result["midpoint"],
            }

        data = resp.json()
        bids = sorted(data.get("bids", []), key=lambda x: float(x["price"]), reverse=True)
        asks = sorted(data.get("asks", []), key=lambda x: float(x["price"]))

        best_bid = float(bids[0]["price"]) if bids else result.get("midpoint")
        best_ask = float(asks[0]["price"]) if asks else result.get("midpoint")

        # For UpDown markets, book can be very sparse (bid=0.01, ask=0.99).
        # Use midpoint as the implied probability instead.
        if "midpoint" in result:
            best_bid = result["midpoint"]
            best_ask = result["midpoint"]

        spread = (best_ask - best_bid) if (best_bid is not None and best_ask is not None) else None

        # Depth: total size across all levels
        depth_bid = sum(float(b.get("size", 0)) for b in bids)
        depth_ask = sum(float(a.get("size", 0)) for a in asks)

        return {
            "best_bid": best_bid,
            "best_ask": best_ask,
            "spread": spread,
            "depth_bid": depth_bid,
            "depth_ask": depth_ask,
            "midpoint": result.get("midpoint"),
        }
    except Exception as e:
        logger.warning("CLOB book error: %s", e)
        if "midpoint" in result:
            return {
                "best_bid": result["midpoint"],
                "best_ask": result["midpoint"],
                "spread": 0.0,
                "depth_bid": 0.0,
                "depth_ask": 0.0,
                "midpoint": result["midpoint"],
            }
        return None


# ---------------------------------------------------------------------------
# Chainlink Oracle
# ---------------------------------------------------------------------------

def fetch_chainlink_price(
    session: requests.Session, asset: str, polygon_rpc_url: str,
) -> Optional[float]:
    """Fetch latest price from Chainlink oracle on Polygon via JSON-RPC.

    This is the price that actually settles Polymarket UpDown markets.
    Same contract addresses and decoding as pmbot-rust/src/exchanges/chainlink.rs.
    """
    contract = CHAINLINK_FEEDS.get(asset)
    if not contract:
        return None

    try:
        resp = session.post(
            polygon_rpc_url,
            json={
                "jsonrpc": "2.0",
                "id": 1,
                "method": "eth_call",
                "params": [
                    {"to": contract, "data": LATEST_ROUND_DATA_SELECTOR},
                    "latest",
                ],
            },
            timeout=10,
        )
        if resp.status_code != 200:
            logger.warning("Chainlink RPC HTTP %d", resp.status_code)
            return None

        data = resp.json()
        if "error" in data:
            logger.warning("Chainlink RPC error: %s", data["error"])
            return None

        result_hex = data.get("result", "")
        hex_str = result_hex[2:] if result_hex.startswith("0x") else result_hex

        if len(hex_str) < 128:
            logger.warning("Chainlink response too short: %d chars", len(hex_str))
            return None

        # answer is bytes 32..64 -> hex chars 64..128
        answer_hex = hex_str[64:128]
        answer = int(answer_hex, 16)

        if answer <= 0:
            logger.warning("Chainlink invalid price: %d", answer)
            return None

        price = answer / (10 ** CHAINLINK_DECIMALS)
        return price

    except Exception as e:
        logger.warning("Chainlink fetch error: %s", e)
        return None


# ---------------------------------------------------------------------------
# Exchange price (Binance with CoinGecko fallback)
# ---------------------------------------------------------------------------

COINGECKO_IDS = {"btc": "bitcoin", "eth": "ethereum"}

def fetch_exchange_price(session: requests.Session, symbol: str, asset: str) -> Optional[dict]:
    """Fetch exchange price. Tries Binance first, falls back to CoinGecko."""
    result = _fetch_binance_price(session, symbol)
    if result:
        return result

    return _fetch_coingecko_price(session, asset)


def _fetch_binance_price(session: requests.Session, symbol: str) -> Optional[dict]:
    """Fetch Binance best bid/ask for a symbol."""
    try:
        resp = session.get(
            BINANCE_TICKER_URL,
            params={"symbol": symbol},
            timeout=5,
        )
        if resp.status_code != 200:
            return None

        data = resp.json()
        bid = float(data["bidPrice"])
        ask = float(data["askPrice"])
        return {
            "price": (bid + ask) / 2,
            "spread": ask - bid,
            "source": "binance",
        }
    except Exception:
        return None


def _fetch_coingecko_price(session: requests.Session, asset: str) -> Optional[dict]:
    """Fetch price from CoinGecko free API (no key needed, works everywhere)."""
    cg_id = COINGECKO_IDS.get(asset)
    if not cg_id:
        return None

    try:
        resp = session.get(
            "https://api.coingecko.com/api/v3/simple/price",
            params={"ids": cg_id, "vs_currencies": "usd"},
            timeout=10,
        )
        if resp.status_code != 200:
            logger.debug("CoinGecko %d", resp.status_code)
            return None

        data = resp.json()
        price = float(data[cg_id]["usd"])
        return {
            "price": price,
            "spread": 0.0,  # CoinGecko doesn't provide bid/ask
            "source": "coingecko",
        }
    except Exception as e:
        logger.debug("CoinGecko error: %s", e)
        return None


# ---------------------------------------------------------------------------
# CSV output
# ---------------------------------------------------------------------------

def append_csv(row: dict) -> None:
    """Append a single row to the CSV file."""
    write_header = not CSV_PATH.exists()
    CSV_PATH.parent.mkdir(parents=True, exist_ok=True)

    with open(CSV_PATH, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


# ---------------------------------------------------------------------------
# Collection loop
# ---------------------------------------------------------------------------

def collect_once(
    session: requests.Session,
    asset: str,
    binance_symbol: str,
    polygon_rpc_url: str,
    last_token: Optional[dict],
) -> Optional[dict]:
    """Run one collection cycle. Returns current token info for reuse."""
    timestamp = int(time.time())

    # Discover active UpDown market (re-discover every 5m window)
    window_start = current_5m_window_start()
    if last_token is None or last_token.get("window_start") != window_start:
        token_info = discover_updown_token(session, asset)
        if token_info:
            logger.info(
                "Discovered market: %s (token=%s..., window=%d)",
                token_info["question"], token_info["token_id"][:20], window_start,
            )
        else:
            logger.debug("No active UpDown market for %s at window %d", asset, window_start)
    else:
        token_info = last_token

    # Fetch Polymarket orderbook
    poly = None
    if token_info:
        poly = fetch_orderbook(session, token_info["token_id"])

    # Fetch Chainlink oracle price (the settlement price for UpDown markets)
    chainlink_price = fetch_chainlink_price(session, asset, polygon_rpc_url)

    # Fetch exchange spot price (Binance with CoinGecko fallback)
    binance = fetch_exchange_price(session, binance_symbol, asset)

    # Compute oracle-vs-exchange spread (key signal: when Chainlink diverges
    # from Binance, UpDown implied prob may be mispriced)
    oracle_vs_binance = None
    if chainlink_price and binance:
        oracle_vs_binance = chainlink_price - binance["price"]

    # Best available price for the oracle_price field (prefer Chainlink)
    best_oracle = chainlink_price or (binance["price"] if binance else None)
    best_vwap = binance["price"] if binance else chainlink_price

    # Build row
    row = {
        "timestamp": timestamp,
        "poly_midpoint": poly["midpoint"] if poly else None,
        "poly_best_bid": poly["best_bid"] if poly else None,
        "poly_best_ask": poly["best_ask"] if poly else None,
        "poly_spread": poly["spread"] if poly else None,
        "poly_depth_bid": poly["depth_bid"] if poly else None,
        "poly_depth_ask": poly["depth_ask"] if poly else None,
        "chainlink_oracle_price": chainlink_price,
        "binance_price": binance["price"] if binance else None,
        "oracle_vs_binance_spread": oracle_vs_binance,
        "oracle_price": best_oracle,
        "vwap": best_vwap,
        "cross_exchange_spread": binance["spread"] if binance else None,
        "realtime_volatility": None,  # computed downstream from collected data
        "token_id": token_info["token_id"] if token_info else None,
        "market_question": token_info["question"] if token_info else None,
        "window_start": token_info["window_start"] if token_info else None,
    }

    append_csv(row)

    midpoint = poly["midpoint"] if poly and poly.get("midpoint") else "N/A"
    cl_str = f"${chainlink_price:.2f}" if chainlink_price else "N/A"
    src = binance.get("source", "?") if binance else "N/A"
    bn_str = f"${binance['price']:.2f}({src})" if binance else "N/A"
    spread_str = f"${oracle_vs_binance:.2f}" if oracle_vs_binance is not None else "N/A"
    logger.info(
        "Collected: midpoint=%s, chainlink=%s, exchange=%s, oracle_spread=%s",
        midpoint, cl_str, bn_str, spread_str,
    )

    return token_info


def main():
    parser = argparse.ArgumentParser(description="Polymarket UpDown data collector")
    parser.add_argument("--interval", type=int, default=30,
                        help="Polling interval in seconds (default: 30)")
    parser.add_argument("--asset", type=str, default="btc",
                        choices=list(UPDOWN_ASSETS.keys()),
                        help="Asset to collect (default: btc)")
    parser.add_argument("--polygon-rpc", type=str,
                        default=os.environ.get("POLYGON_RPC_URL", "https://1rpc.io/matic"),
                        help="Polygon RPC URL for Chainlink oracle")
    args = parser.parse_args()

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    asset = args.asset
    binance_symbol = UPDOWN_ASSETS[asset][1]

    logger.info(
        "Starting collector: asset=%s, interval=%ds, chainlink=%s, csv=%s",
        asset, args.interval, "enabled" if CHAINLINK_FEEDS.get(asset) else "disabled", CSV_PATH,
    )

    session = requests.Session()
    session.headers["User-Agent"] = "pmbot-collector/1.0"

    last_token = None
    cycles = 0

    while not _shutdown:
        try:
            last_token = collect_once(
                session, asset, binance_symbol, args.polygon_rpc, last_token,
            )
            cycles += 1
            if cycles % 60 == 0:
                rows = sum(1 for _ in open(CSV_PATH)) - 1 if CSV_PATH.exists() else 0
                logger.info("Stats: %d cycles, %d rows in CSV", cycles, rows)
        except Exception as e:
            logger.error("Collection cycle failed: %s", e)

        # Sleep in small increments to allow clean shutdown
        for _ in range(args.interval):
            if _shutdown:
                break
            time.sleep(1)

    logger.info("Collector stopped after %d cycles", cycles)


if __name__ == "__main__":
    main()
