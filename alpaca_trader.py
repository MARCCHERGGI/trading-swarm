"""
Alpaca Paper Trading + Simulation Mode

Supports two modes:
  - PAPER: Uses Alpaca paper trading API (requires ALPACA_API_KEY + ALPACA_SECRET_KEY)
  - SIMULATION: No keys needed. Uses real yfinance prices, tracks portfolio in
    simulation_portfolio.json. Default mode when no API keys are set.
"""

import os
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import yfinance as yf

logger = logging.getLogger(__name__)

PAPER_BASE_URL = "https://paper-api.alpaca.markets"
SIM_PORTFOLIO_PATH = Path(__file__).parent / "simulation_portfolio.json"


def _get_live_price(ticker: str) -> Optional[float]:
    """Fetch the latest price from yfinance."""
    try:
        t = yf.Ticker(ticker)
        hist = t.history(period="1d")
        if hist is not None and not hist.empty:
            return float(hist["Close"].iloc[-1])
    except Exception as e:
        logger.warning("Failed to fetch price for %s: %s", ticker, e)
    return None


class AlpacaTrader:
    """
    Unified trading interface with Alpaca paper trading and local simulation.

    Mode is auto-detected: if ALPACA_API_KEY and ALPACA_SECRET_KEY are set,
    uses Alpaca paper trading API. Otherwise falls back to simulation mode.
    """

    def __init__(self, mode: Optional[str] = None):
        """
        Args:
            mode: 'paper', 'simulation', or None (auto-detect).
        """
        self.api_key = os.environ.get("ALPACA_API_KEY", "")
        self.secret_key = os.environ.get("ALPACA_SECRET_KEY", "")

        if mode:
            self.mode = mode.lower()
        elif self.api_key and self.secret_key:
            self.mode = "paper"
        else:
            self.mode = "simulation"

        self.api = None
        if self.mode == "paper":
            self._init_alpaca()
        else:
            self._init_simulation()

        logger.info("AlpacaTrader initialized in %s mode", self.mode.upper())

    # ------------------------------------------------------------------
    # Alpaca paper trading setup
    # ------------------------------------------------------------------

    def _init_alpaca(self):
        try:
            import alpaca_trade_api as tradeapi
            self.api = tradeapi.REST(
                self.api_key,
                self.secret_key,
                PAPER_BASE_URL,
                api_version="v2",
            )
            # Verify connection
            self.api.get_account()
            logger.info("Connected to Alpaca paper trading")
        except Exception as e:
            logger.warning("Alpaca connection failed (%s), falling back to simulation", e)
            self.mode = "simulation"
            self.api = None
            self._init_simulation()

    # ------------------------------------------------------------------
    # Simulation setup
    # ------------------------------------------------------------------

    def _init_simulation(self):
        self.sim = self._load_sim_portfolio()

    def _load_sim_portfolio(self) -> dict:
        if SIM_PORTFOLIO_PATH.exists():
            try:
                with open(SIM_PORTFOLIO_PATH, "r") as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError):
                pass
        return {
            "cash": 100_000.0,
            "positions": {},
            "trades": [],
            "initial_equity": 100_000.0,
        }

    def _save_sim_portfolio(self):
        with open(SIM_PORTFOLIO_PATH, "w") as f:
            json.dump(self.sim, f, indent=2, default=str)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_account(self) -> dict:
        """Return account summary: equity, cash, buying power."""
        if self.mode == "paper":
            acct = self.api.get_account()
            return {
                "mode": "paper",
                "equity": float(acct.equity),
                "cash": float(acct.cash),
                "buying_power": float(acct.buying_power),
                "portfolio_value": float(acct.portfolio_value),
            }

        # Simulation
        positions_value = 0.0
        for ticker, pos in self.sim["positions"].items():
            price = _get_live_price(ticker)
            if price is not None:
                positions_value += price * pos["qty"]
            else:
                positions_value += pos["avg_price"] * pos["qty"]

        equity = self.sim["cash"] + positions_value
        return {
            "mode": "simulation",
            "equity": round(equity, 2),
            "cash": round(self.sim["cash"], 2),
            "buying_power": round(self.sim["cash"], 2),
            "portfolio_value": round(equity, 2),
            "initial_equity": self.sim["initial_equity"],
        }

    def get_positions(self) -> list[dict]:
        """Return list of open positions with current market value."""
        if self.mode == "paper":
            positions = self.api.list_positions()
            return [
                {
                    "ticker": p.symbol,
                    "qty": float(p.qty),
                    "avg_price": float(p.avg_entry_price),
                    "current_price": float(p.current_price),
                    "market_value": float(p.market_value),
                    "unrealized_pnl": float(p.unrealized_pl),
                    "unrealized_pnl_pct": float(p.unrealized_plpc) * 100,
                }
                for p in positions
            ]

        # Simulation
        result = []
        for ticker, pos in self.sim["positions"].items():
            price = _get_live_price(ticker) or pos["avg_price"]
            market_value = price * pos["qty"]
            cost_basis = pos["avg_price"] * pos["qty"]
            unrealized_pnl = market_value - cost_basis
            unrealized_pnl_pct = (unrealized_pnl / cost_basis * 100) if cost_basis > 0 else 0.0
            result.append({
                "ticker": ticker,
                "qty": pos["qty"],
                "avg_price": round(pos["avg_price"], 2),
                "current_price": round(price, 2),
                "market_value": round(market_value, 2),
                "unrealized_pnl": round(unrealized_pnl, 2),
                "unrealized_pnl_pct": round(unrealized_pnl_pct, 2),
            })
        return result

    def place_order(self, ticker: str, qty: float, side: str) -> dict:
        """
        Place a market order.

        Args:
            ticker: Symbol (e.g. 'AAPL')
            qty: Number of shares (fractional allowed in sim)
            side: 'buy' or 'sell'

        Returns dict with order details.
        """
        side = side.lower()
        if side not in ("buy", "sell"):
            raise ValueError(f"Invalid side: {side}. Must be 'buy' or 'sell'.")

        if self.mode == "paper":
            order = self.api.submit_order(
                symbol=ticker,
                qty=qty,
                side=side,
                type="market",
                time_in_force="day",
            )
            return {
                "id": order.id,
                "ticker": ticker,
                "side": side,
                "qty": float(qty),
                "status": order.status,
                "mode": "paper",
            }

        # Simulation -- execute at real yfinance price
        price = _get_live_price(ticker)
        if price is None:
            return {"error": f"Could not fetch price for {ticker}", "ticker": ticker}

        now = datetime.now(timezone.utc).isoformat()

        if side == "buy":
            cost = price * qty
            if cost > self.sim["cash"]:
                return {"error": "Insufficient cash", "required": round(cost, 2),
                        "available": round(self.sim["cash"], 2)}
            self.sim["cash"] -= cost

            if ticker in self.sim["positions"]:
                existing = self.sim["positions"][ticker]
                total_qty = existing["qty"] + qty
                existing["avg_price"] = (
                    (existing["avg_price"] * existing["qty"] + price * qty) / total_qty
                )
                existing["qty"] = total_qty
            else:
                self.sim["positions"][ticker] = {
                    "qty": qty,
                    "avg_price": price,
                    "opened_at": now,
                }
        else:  # sell
            if ticker not in self.sim["positions"]:
                return {"error": f"No position in {ticker}"}
            pos = self.sim["positions"][ticker]
            if qty > pos["qty"]:
                return {"error": f"Insufficient shares. Have {pos['qty']}, selling {qty}"}
            proceeds = price * qty
            self.sim["cash"] += proceeds
            pnl = (price - pos["avg_price"]) * qty

            pos["qty"] -= qty
            if pos["qty"] <= 0.0001:
                del self.sim["positions"][ticker]

            self.sim["trades"].append({
                "ticker": ticker,
                "side": "sell",
                "qty": qty,
                "price": round(price, 2),
                "pnl": round(pnl, 2),
                "timestamp": now,
            })

        if side == "buy":
            self.sim["trades"].append({
                "ticker": ticker,
                "side": "buy",
                "qty": qty,
                "price": round(price, 2),
                "timestamp": now,
            })

        self._save_sim_portfolio()

        return {
            "ticker": ticker,
            "side": side,
            "qty": qty,
            "price": round(price, 2),
            "mode": "simulation",
            "timestamp": now,
        }

    def close_position(self, ticker: str) -> dict:
        """Close entire position for a ticker."""
        if self.mode == "paper":
            try:
                self.api.close_position(ticker)
                return {"ticker": ticker, "status": "closed", "mode": "paper"}
            except Exception as e:
                return {"ticker": ticker, "error": str(e), "mode": "paper"}

        # Simulation
        if ticker not in self.sim["positions"]:
            return {"error": f"No position in {ticker}"}
        qty = self.sim["positions"][ticker]["qty"]
        return self.place_order(ticker, qty, "sell")

    def auto_trade(self, signals: list[dict], portfolio_pct: float = 0.1) -> list[dict]:
        """
        Automatically trade based on swarm signals.

        Fires on |consensus| > 0.5. Allocates portfolio_pct of equity per trade.

        Args:
            signals: List of scan results from LiveScanner.scan()
            portfolio_pct: Fraction of portfolio equity per trade (default 10%)

        Returns list of executed order results.
        """
        account = self.get_account()
        equity = account["equity"]
        trade_budget = equity * portfolio_pct
        orders = []

        for sig in signals:
            ticker = sig.get("ticker", "")
            consensus = sig.get("consensus", 0)
            action = sig.get("action", "HOLD")
            price = sig.get("price", 0)

            if abs(consensus) <= 0.5 or price <= 0:
                continue

            if action == "BUY":
                qty = int(trade_budget / price)
                if qty < 1:
                    continue
                # Don't buy if already holding
                positions = {p["ticker"]: p for p in self.get_positions()}
                if ticker in positions:
                    logger.info("Already holding %s, skipping buy", ticker)
                    continue
                result = self.place_order(ticker, qty, "buy")
                result["reason"] = f"consensus={consensus:+.3f}"
                orders.append(result)
                logger.info("AUTO BUY %s x%d @ $%.2f (consensus=%+.3f)",
                            ticker, qty, price, consensus)

            elif action == "SELL":
                positions = {p["ticker"]: p for p in self.get_positions()}
                if ticker not in positions:
                    continue
                result = self.close_position(ticker)
                result["reason"] = f"consensus={consensus:+.3f}"
                orders.append(result)
                logger.info("AUTO SELL %s (consensus=%+.3f)", ticker, consensus)

        return orders

    def get_pnl(self) -> dict:
        """
        Get portfolio P&L summary.

        Returns total P&L, realized P&L (from closed trades), and unrealized
        P&L (from open positions).
        """
        account = self.get_account()
        positions = self.get_positions()

        unrealized_pnl = sum(p["unrealized_pnl"] for p in positions)
        realized_pnl = sum(
            t.get("pnl", 0) for t in self.sim.get("trades", [])
            if t.get("side") == "sell"
        ) if self.mode == "simulation" else 0.0

        if self.mode == "paper":
            try:
                acct = self.api.get_account()
                total_pnl = float(acct.equity) - float(acct.last_equity)
            except Exception:
                total_pnl = unrealized_pnl
        else:
            initial = self.sim.get("initial_equity", 100_000.0)
            total_pnl = account["equity"] - initial
            realized_pnl = sum(
                t.get("pnl", 0) for t in self.sim.get("trades", [])
                if t.get("side") == "sell"
            )

        total_return_pct = (
            (total_pnl / self.sim.get("initial_equity", 100_000.0)) * 100
            if self.mode == "simulation"
            else (total_pnl / max(account["equity"] - total_pnl, 1)) * 100
        )

        return {
            "mode": self.mode,
            "equity": account["equity"],
            "cash": account["cash"],
            "total_pnl": round(total_pnl, 2),
            "total_return_pct": round(total_return_pct, 2),
            "realized_pnl": round(realized_pnl, 2),
            "unrealized_pnl": round(unrealized_pnl, 2),
            "open_positions": len(positions),
            "positions": positions,
        }
