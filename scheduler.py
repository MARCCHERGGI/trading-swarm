"""
Trading Swarm Scheduler

Scans the watchlist every 30 minutes during market hours (9:30-16:00 ET, weekdays).
- Logs signals to signals_log.csv
- Sends Telegram alerts on |consensus| > 0.6
- Saves daily reports to daily_reports/YYYY-MM-DD.json
- Optionally auto-trades via AlpacaTrader
"""

import csv
import json
import logging
import os
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional
from urllib.request import urlopen, Request
from urllib.error import URLError

import schedule

from trading_swarm import create_swarm, LiveScanner
from alpaca_trader import AlpacaTrader

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).parent
WATCHLIST_PATH = BASE_DIR / "watchlist.json"
SIGNALS_LOG_PATH = BASE_DIR / "signals_log.csv"
DAILY_REPORTS_DIR = BASE_DIR / "daily_reports"

# Eastern Time offset (UTC-5, UTC-4 during DST)
ET_OFFSET = timedelta(hours=-4)  # DST default; adjusted in is_market_hours


def load_watchlist() -> list[str]:
    """Load tickers from watchlist.json."""
    try:
        with open(WATCHLIST_PATH, "r") as f:
            tickers = json.load(f)
        return [t.strip().upper() for t in tickers if isinstance(t, str) and t.strip()]
    except (FileNotFoundError, json.JSONDecodeError):
        logger.warning("Could not load watchlist.json, using defaults")
        return ["AAPL", "TSLA", "NVDA", "MSFT", "SPY"]


def is_market_hours() -> bool:
    """Check if current time is within US market hours (9:30-16:00 ET, weekdays)."""
    now_utc = datetime.now(timezone.utc)

    # Simple DST check: second Sunday in March to first Sunday in November
    year = now_utc.year
    # March: second Sunday
    march_1 = datetime(year, 3, 1, tzinfo=timezone.utc)
    dst_start = march_1 + timedelta(days=(6 - march_1.weekday()) % 7 + 7)
    # November: first Sunday
    nov_1 = datetime(year, 11, 1, tzinfo=timezone.utc)
    dst_end = nov_1 + timedelta(days=(6 - nov_1.weekday()) % 7)

    if dst_start <= now_utc < dst_end:
        et_offset = timedelta(hours=-4)  # EDT
    else:
        et_offset = timedelta(hours=-5)  # EST

    now_et = now_utc + et_offset
    # Weekday check (Monday=0 ... Friday=4)
    if now_et.weekday() > 4:
        return False
    market_open = now_et.replace(hour=9, minute=30, second=0, microsecond=0)
    market_close = now_et.replace(hour=16, minute=0, second=0, microsecond=0)
    return market_open <= now_et <= market_close


def send_telegram_alert(message: str) -> bool:
    """Send a Telegram message via bot API. Returns True on success."""
    bot_token = os.environ.get("TELEGRAM_BOT_TOKEN", "")
    chat_id = os.environ.get("TELEGRAM_CHAT_ID", "")

    if not bot_token or not chat_id:
        return False

    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    payload = json.dumps({"chat_id": chat_id, "text": message, "parse_mode": "HTML"}).encode()
    req = Request(url, data=payload, headers={"Content-Type": "application/json"})

    try:
        with urlopen(req, timeout=10) as resp:
            return resp.status == 200
    except (URLError, OSError) as e:
        logger.warning("Telegram alert failed: %s", e)
        return False


def log_signals_csv(results: list[dict]):
    """Append scan results to signals_log.csv."""
    file_exists = SIGNALS_LOG_PATH.exists()
    with open(SIGNALS_LOG_PATH, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow([
                "timestamp", "ticker", "price", "consensus", "confidence",
                "action", "sma", "rsi", "momentum", "bollinger", "volume", "macd",
            ])
        now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
        for r in results:
            agents = r.get("agents", {})
            writer.writerow([
                now,
                r.get("ticker", ""),
                r.get("price", 0),
                r.get("consensus", 0),
                r.get("confidence", 0),
                r.get("action", "HOLD"),
                agents.get("SMA Crossover", {}).get("signal", 0),
                agents.get("RSI(14)", {}).get("signal", 0),
                agents.get("Momentum(10)", {}).get("signal", 0),
                agents.get("Bollinger Bands", {}).get("signal", 0),
                agents.get("Volume/OBV", {}).get("signal", 0),
                agents.get("MACD", {}).get("signal", 0),
            ])


def save_daily_report(results: list[dict]):
    """Save full scan results as a daily JSON report."""
    DAILY_REPORTS_DIR.mkdir(exist_ok=True)
    now_utc = datetime.now(timezone.utc)
    et_time = now_utc + ET_OFFSET
    date_str = et_time.strftime("%Y-%m-%d")
    report_path = DAILY_REPORTS_DIR / f"{date_str}.json"

    report = {
        "date": date_str,
        "generated_at": now_utc.isoformat(),
        "ticker_count": len(results),
        "results": results,
    }

    # Append to existing report if one exists for today
    if report_path.exists():
        try:
            with open(report_path, "r") as f:
                existing = json.load(f)
            if "scans" not in existing:
                existing = {"date": date_str, "scans": [existing]}
            existing["scans"].append(report)
            report = existing
        except (json.JSONDecodeError, IOError):
            report = {"date": date_str, "scans": [report]}
    else:
        report = {"date": date_str, "scans": [report]}

    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)

    logger.info("Daily report saved: %s", report_path)


def run_scan_cycle(trader: Optional[AlpacaTrader] = None, force: bool = False):
    """Execute one scan cycle: scan watchlist, log, alert, optionally trade."""
    if not force and not is_market_hours():
        logger.debug("Outside market hours, skipping scan")
        return

    tickers = load_watchlist()
    if not tickers:
        logger.warning("Empty watchlist, nothing to scan")
        return

    logger.info("Scanning %d tickers...", len(tickers))
    swarm = create_swarm()
    scanner = LiveScanner(swarm)
    results = scanner.scan(tickers)

    # Log to CSV
    log_signals_csv(results)
    logger.info("Signals logged to %s", SIGNALS_LOG_PATH)

    # Save daily report
    save_daily_report(results)

    # Telegram alerts for strong signals
    alerts = [r for r in results if abs(r.get("consensus", 0)) > 0.6]
    if alerts:
        lines = ["<b>Trading Swarm Alert</b>\n"]
        for r in alerts:
            emoji = "\U0001f7e2" if r["consensus"] > 0 else "\U0001f534"
            lines.append(
                f"{emoji} <b>{r['ticker']}</b> ${r['price']:.2f} "
                f"| {r['action']} | consensus: {r['consensus']:+.3f}"
            )
        msg = "\n".join(lines)
        sent = send_telegram_alert(msg)
        if sent:
            logger.info("Telegram alert sent for %d strong signals", len(alerts))
        else:
            logger.debug("Telegram not configured or alert failed")

    # Auto-trade if trader is provided
    if trader is not None:
        orders = trader.auto_trade(results)
        if orders:
            logger.info("Auto-traded %d orders", len(orders))
            # Alert on trades too
            trade_lines = ["<b>Auto-Trade Executed</b>\n"]
            for o in orders:
                if "error" not in o:
                    trade_lines.append(
                        f"{'BUY' if o.get('side') == 'buy' else 'SELL'} "
                        f"<b>{o['ticker']}</b> x{o.get('qty', '?')} @ ${o.get('price', 0):.2f}"
                    )
            if len(trade_lines) > 1:
                send_telegram_alert("\n".join(trade_lines))

    logger.info("Scan cycle complete. %d tickers scanned.", len(results))
    return results


def start_scheduler(auto_trade: bool = True, force_first_run: bool = True):
    """
    Start the scheduler loop. Scans every 30 minutes during market hours.

    Args:
        auto_trade: If True, auto-trades on strong signals via AlpacaTrader
        force_first_run: If True, run first scan immediately regardless of market hours
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    trader = AlpacaTrader() if auto_trade else None
    if trader:
        account = trader.get_account()
        logger.info(
            "Trader: %s mode | Equity: $%.2f | Cash: $%.2f",
            account["mode"], account["equity"], account["cash"],
        )

    # Run first scan immediately
    if force_first_run:
        logger.info("Running initial scan...")
        run_scan_cycle(trader=trader, force=True)

    # Schedule every 30 minutes
    schedule.every(30).minutes.do(run_scan_cycle, trader=trader)

    logger.info("Scheduler started. Scanning every 30 minutes during market hours.")
    logger.info("Press Ctrl+C to stop.")

    try:
        while True:
            schedule.run_pending()
            time.sleep(10)
    except KeyboardInterrupt:
        logger.info("Scheduler stopped.")


if __name__ == "__main__":
    start_scheduler()

