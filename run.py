#!/usr/bin/env python3
"""
CLI entry point for the Trading Swarm system.

Usage:
    python run.py scan AAPL TSLA NVDA BTC-USD    # Scan tickers, print ranked signals
    python run.py backtest AAPL                    # Backtest single ticker over 1yr
    python run.py serve                            # Start Flask dashboard on port 5000
    python run.py auto                             # Start scheduler (scans every 30min)
    python run.py portfolio                        # Show positions + P&L
    python run.py history                          # Show last 50 signals
"""

import sys
import json
import csv
from pathlib import Path
from trading_swarm import (
    create_swarm, fetch_data, BacktestEngine, LiveScanner
)


def print_scan_results(results: list[dict]):
    """Pretty-print scanner results as a table."""
    header = f"{'Ticker':<10} {'Price':>10} {'Action':<6} {'Consensus':>10} {'Confidence':>11}"
    print("\n" + "=" * 60)
    print("  TRADING SWARM --LIVE SCAN")
    print("=" * 60)
    print(header)
    print("-" * 60)

    for r in results:
        if r.get("error"):
            print(f"{r['ticker']:<10} {'N/A':>10} {'ERROR':<6} {'':>10} {'':>11}  {r['error']}")
            continue

        color_action = r["action"]
        print(f"{r['ticker']:<10} {r['price']:>10.2f} {color_action:<6} {r['consensus']:>+10.3f} {r['confidence']:>10.3f}")

        # Show per-agent breakdown
        for agent_name, detail in r["agents"].items():
            sig_char = "+" if detail["signal"] > 0 else ("-" if detail["signal"] < 0 else "·")
            print(f"  {sig_char} {agent_name:<20} {detail['label']:<5} conf={detail['confidence']:.2f}  {detail['reason']}")
        print()


def print_backtest_results(ticker: str, stats: dict):
    """Pretty-print backtest statistics."""
    print("\n" + "=" * 60)
    print(f"  TRADING SWARM --BACKTEST: {ticker}")
    print("=" * 60)

    if "error" in stats:
        print(f"  Error: {stats['error']}")
        return

    print(f"  Total Trades:    {stats['total_trades']}")
    print(f"  Win Rate:        {stats['win_rate']}%")
    print(f"  Total Return:    {stats['total_return']:+.2f}%")
    print(f"  Sharpe Ratio:    {stats['sharpe_ratio']:.2f}")
    print(f"  Max Drawdown:    {stats['max_drawdown']:.2f}%")
    print()

    # Agent weights after backtest adaptation
    print("  Adapted Agent Weights:")
    for name, weight in stats["agent_weights"].items():
        bar = "#" * int(weight * 20)
        print(f"    {name:<20} {weight:.2f}  {bar}")
    print()

    # Trade log
    if stats["trades"]:
        print(f"  {'Entry Date':<12} {'Exit Date':<12} {'Entry':>8} {'Exit':>8} {'P&L':>8}")
        print("  " + "-" * 52)
        for t in stats["trades"]:
            pnl_str = f"{t['pnl_pct']:+.2f}%"
            print(f"  {t['entry_date']:<12} {t['exit_date']:<12} {t['entry_price']:>8.2f} {t['exit_price']:>8.2f} {pnl_str:>8}")
    print()


def cmd_scan(tickers: list[str]):
    """Run live scan on provided tickers."""
    if not tickers:
        print("Error: provide at least one ticker. Example: python run.py scan AAPL TSLA")
        sys.exit(1)

    swarm = create_swarm()
    scanner = LiveScanner(swarm)
    print(f"Scanning {len(tickers)} ticker(s)...")
    results = scanner.scan(tickers)
    print_scan_results(results)


def cmd_backtest(ticker: str):
    """Run backtest on a single ticker."""
    print(f"Fetching 1 year of data for {ticker}...")
    df = fetch_data(ticker, period="1y")
    if df is None or len(df) < 60:
        print(f"Error: could not fetch sufficient data for {ticker}")
        sys.exit(1)

    print(f"Running walk-forward backtest ({len(df)} bars)...")
    swarm = create_swarm()
    engine = BacktestEngine(swarm)
    stats = engine.run(df)
    print_backtest_results(ticker, stats)


def cmd_serve():
    """Start Flask web server for the dashboard."""
    from flask import Flask, send_file, jsonify, request

    app = Flask(__name__)

    @app.route("/")
    def index():
        return send_file("dashboard.html")

    @app.route("/api/scan", methods=["GET"])
    def api_scan():
        tickers_param = request.args.get("tickers", "AAPL,TSLA,NVDA")
        tickers = [t.strip().upper() for t in tickers_param.split(",") if t.strip()]
        if not tickers:
            return jsonify({"error": "No tickers provided"}), 400

        swarm = create_swarm()
        scanner = LiveScanner(swarm)
        results = scanner.scan(tickers)
        return jsonify(results)

    @app.route("/api/backtest", methods=["GET"])
    def api_backtest():
        ticker = request.args.get("ticker", "AAPL").strip().upper()
        df = fetch_data(ticker, period="1y")
        if df is None or len(df) < 60:
            return jsonify({"error": f"Could not fetch data for {ticker}"}), 400

        swarm = create_swarm()
        engine = BacktestEngine(swarm)
        stats = engine.run(df)
        return jsonify(stats)

    @app.route("/api/portfolio", methods=["GET"])
    def api_portfolio():
        from alpaca_trader import AlpacaTrader
        trader = AlpacaTrader()
        return jsonify(trader.get_pnl())

    @app.route("/api/history", methods=["GET"])
    def api_history():
        limit = request.args.get("limit", "50", type=str)
        try:
            limit = int(limit)
        except ValueError:
            limit = 50
        signals_path = Path(__file__).parent / "signals_log.csv"
        if not signals_path.exists():
            return jsonify([])
        rows = []
        with open(signals_path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows.append(row)
        # Return last N rows (most recent last)
        return jsonify(rows[-limit:])

    @app.route("/api/watchlist", methods=["GET"])
    def api_watchlist_get():
        watchlist_path = Path(__file__).parent / "watchlist.json"
        try:
            with open(watchlist_path, "r") as f:
                return jsonify(json.load(f))
        except (FileNotFoundError, json.JSONDecodeError):
            return jsonify([])

    @app.route("/api/watchlist", methods=["POST"])
    def api_watchlist_set():
        data = request.get_json(silent=True)
        if not data or not isinstance(data, list):
            return jsonify({"error": "Expected JSON array of tickers"}), 400
        tickers = [t.strip().upper() for t in data if isinstance(t, str) and t.strip()]
        watchlist_path = Path(__file__).parent / "watchlist.json"
        with open(watchlist_path, "w") as f:
            json.dump(tickers, f, indent=2)
        return jsonify({"ok": True, "tickers": tickers})

    print("Starting Trading Swarm Dashboard on http://127.0.0.1:5000")
    print("Press Ctrl+C to stop.")
    app.run(host="127.0.0.1", port=5000, debug=False)


def cmd_auto():
    """Start the scheduler for automated scanning and trading."""
    from scheduler import start_scheduler
    start_scheduler(auto_trade=True, force_first_run=True)


def cmd_portfolio():
    """Display current positions and P&L."""
    from alpaca_trader import AlpacaTrader

    trader = AlpacaTrader()
    pnl = trader.get_pnl()

    print("\n" + "=" * 60)
    print("  TRADING SWARM --PORTFOLIO")
    print("=" * 60)
    print(f"  Mode:            {pnl['mode'].upper()}")
    print(f"  Equity:          ${pnl['equity']:,.2f}")
    print(f"  Cash:            ${pnl['cash']:,.2f}")
    pnl_color = "+" if pnl['total_pnl'] >= 0 else ""
    print(f"  Total P&L:       {pnl_color}${pnl['total_pnl']:,.2f} ({pnl_color}{pnl['total_return_pct']:.2f}%)")
    print(f"  Realized P&L:    ${pnl['realized_pnl']:,.2f}")
    print(f"  Unrealized P&L:  ${pnl['unrealized_pnl']:,.2f}")
    print()

    positions = pnl["positions"]
    if positions:
        print(f"  {'Ticker':<10} {'Qty':>8} {'Avg Price':>10} {'Current':>10} {'Value':>12} {'P&L':>10} {'P&L%':>8}")
        print("  " + "-" * 70)
        for p in positions:
            pnl_str = f"{'+'if p['unrealized_pnl']>=0 else ''}{p['unrealized_pnl']:,.2f}"
            pct_str = f"{'+'if p['unrealized_pnl_pct']>=0 else ''}{p['unrealized_pnl_pct']:.2f}%"
            print(f"  {p['ticker']:<10} {p['qty']:>8.2f} {p['avg_price']:>10.2f} {p['current_price']:>10.2f} {p['market_value']:>12,.2f} {pnl_str:>10} {pct_str:>8}")
    else:
        print("  No open positions.")
    print()


def cmd_history(count: int = 50):
    """Display last N signals from signals_log.csv."""
    signals_path = Path(__file__).parent / "signals_log.csv"
    if not signals_path.exists():
        print("No signal history found. Run a scan first or start the scheduler.")
        return

    rows = []
    with open(signals_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    if not rows:
        print("Signal log is empty.")
        return

    recent = rows[-count:]

    print("\n" + "=" * 90)
    print(f"  TRADING SWARM --SIGNAL HISTORY (last {len(recent)} entries)")
    print("=" * 90)
    print(f"  {'Timestamp':<20} {'Ticker':<10} {'Price':>10} {'Consensus':>10} {'Action':<6} {'SMA':>4} {'RSI':>4} {'Mom':>4} {'BB':>4} {'Vol':>4} {'MACD':>5}")
    print("  " + "-" * 85)

    for r in recent:
        print(
            f"  {r.get('timestamp', '?'):<20} "
            f"{r.get('ticker', '?'):<10} "
            f"{float(r.get('price', 0)):>10.2f} "
            f"{float(r.get('consensus', 0)):>+10.3f} "
            f"{r.get('action', '?'):<6} "
            f"{r.get('sma', '0'):>4} "
            f"{r.get('rsi', '0'):>4} "
            f"{r.get('momentum', '0'):>4} "
            f"{r.get('bollinger', '0'):>4} "
            f"{r.get('volume', '0'):>4} "
            f"{r.get('macd', '0'):>5}"
        )
    print()


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    command = sys.argv[1].lower()

    if command == "scan":
        cmd_scan(sys.argv[2:])
    elif command == "backtest":
        if len(sys.argv) < 3:
            print("Error: provide a ticker. Example: python run.py backtest AAPL")
            sys.exit(1)
        cmd_backtest(sys.argv[2].upper())
    elif command == "serve":
        cmd_serve()
    elif command == "auto":
        cmd_auto()
    elif command == "portfolio":
        cmd_portfolio()
    elif command == "history":
        count = 50
        if len(sys.argv) >= 3:
            try:
                count = int(sys.argv[2])
            except ValueError:
                pass
        cmd_history(count)
    else:
        print(f"Unknown command: {command}")
        print("Available commands: scan, backtest, serve, auto, portfolio, history")
        sys.exit(1)


if __name__ == "__main__":
    main()
