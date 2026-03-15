"""
Multi-Agent Trading Swarm System

Architecture: 6 specialized technical analysis agents each independently analyze
market data. A SwarmConsensus layer aggregates their signals using adaptive
weighted voting --weights shift based on each agent's recent prediction accuracy.

Data source: yfinance (free, no API key needed)
Indicators: 'ta' library for standardized technical analysis calculations
"""

import warnings
warnings.filterwarnings("ignore")

from dataclasses import dataclass
from abc import ABC, abstractmethod
from typing import Optional
import numpy as np
import pandas as pd
import yfinance as yf
import ta


# ---------------------------------------------------------------------------
# Agent signal data structure
# ---------------------------------------------------------------------------

@dataclass
class AgentSignal:
    """Return type for every agent's analyze() call."""
    signal: int        # -1 = sell, 0 = hold, 1 = buy
    confidence: float  # 0.0 – 1.0
    reason: str


# ---------------------------------------------------------------------------
# Base agent class
# ---------------------------------------------------------------------------

class BaseAgent(ABC):
    """All agents inherit from this. Each agent gets the full OHLCV DataFrame
    and must return an AgentSignal based solely on its own indicator logic."""

    name: str = "BaseAgent"

    @abstractmethod
    def analyze(self, df: pd.DataFrame) -> AgentSignal:
        ...


# ---------------------------------------------------------------------------
# 1. SMA Crossover Agent --Golden Cross / Death Cross (20/50 day)
# ---------------------------------------------------------------------------

class SMAAgent(BaseAgent):
    name = "SMA Crossover"

    def analyze(self, df: pd.DataFrame) -> AgentSignal:
        if len(df) < 55:
            return AgentSignal(0, 0.0, "Insufficient data for SMA calculation")

        df = df.copy()
        df["sma20"] = ta.trend.sma_indicator(df["Close"], window=20)
        df["sma50"] = ta.trend.sma_indicator(df["Close"], window=50)
        df.dropna(inplace=True)

        if len(df) < 3:
            return AgentSignal(0, 0.0, "Not enough SMA data points")

        curr_diff = df["sma20"].iloc[-1] - df["sma50"].iloc[-1]
        prev_diff = df["sma20"].iloc[-2] - df["sma50"].iloc[-2]
        pct_spread = curr_diff / df["sma50"].iloc[-1]

        # Golden cross: SMA20 just crossed above SMA50
        if curr_diff > 0 and prev_diff <= 0:
            return AgentSignal(1, 0.9, "Golden Cross --SMA20 crossed above SMA50")
        # Death cross: SMA20 just crossed below SMA50
        if curr_diff < 0 and prev_diff >= 0:
            return AgentSignal(-1, 0.9, "Death Cross --SMA20 crossed below SMA50")

        # Sustained trend: wider spread = higher confidence
        conf = min(abs(pct_spread) * 10, 0.75)
        if curr_diff > 0:
            return AgentSignal(1, conf, f"SMA20 above SMA50 by {pct_spread:.2%}")
        elif curr_diff < 0:
            return AgentSignal(-1, conf, f"SMA20 below SMA50 by {pct_spread:.2%}")
        return AgentSignal(0, 0.1, "SMA20 ~ SMA50 -- no clear trend")


# ---------------------------------------------------------------------------
# 2. RSI Agent --Overbought / Oversold with divergence detection
# ---------------------------------------------------------------------------

class RSIAgent(BaseAgent):
    name = "RSI(14)"

    def analyze(self, df: pd.DataFrame) -> AgentSignal:
        if len(df) < 20:
            return AgentSignal(0, 0.0, "Insufficient data for RSI")

        df = df.copy()
        df["rsi"] = ta.momentum.rsi(df["Close"], window=14)
        df.dropna(inplace=True)

        if len(df) < 10:
            return AgentSignal(0, 0.0, "Not enough RSI data")

        rsi = df["rsi"].iloc[-1]

        # Divergence detection: compare last 10 bars
        # Bullish divergence: price makes lower low, RSI makes higher low
        # Bearish divergence: price makes higher high, RSI makes lower high
        recent = df.tail(10)
        mid = len(recent) // 2
        first_half = recent.iloc[:mid]
        second_half = recent.iloc[mid:]

        bullish_div = (
            second_half["Close"].min() < first_half["Close"].min()
            and second_half["rsi"].min() > first_half["rsi"].min()
        )
        bearish_div = (
            second_half["Close"].max() > first_half["Close"].max()
            and second_half["rsi"].max() < first_half["rsi"].max()
        )

        if rsi < 30:
            conf = min((30 - rsi) / 30 + 0.3, 0.95)
            reason = f"RSI={rsi:.1f} oversold"
            if bullish_div:
                conf = min(conf + 0.15, 0.95)
                reason += " + bullish divergence"
            return AgentSignal(1, conf, reason)

        if rsi > 70:
            conf = min((rsi - 70) / 30 + 0.3, 0.95)
            reason = f"RSI={rsi:.1f} overbought"
            if bearish_div:
                conf = min(conf + 0.15, 0.95)
                reason += " + bearish divergence"
            return AgentSignal(-1, conf, reason)

        # Neutral zone --slight lean based on RSI direction
        if bullish_div:
            return AgentSignal(1, 0.45, f"RSI={rsi:.1f} neutral but bullish divergence detected")
        if bearish_div:
            return AgentSignal(-1, 0.45, f"RSI={rsi:.1f} neutral but bearish divergence detected")

        return AgentSignal(0, 0.2, f"RSI={rsi:.1f} neutral")


# ---------------------------------------------------------------------------
# 3. Momentum Agent --10-day price momentum + rate of change
# ---------------------------------------------------------------------------

class MomentumAgent(BaseAgent):
    name = "Momentum(10)"

    def analyze(self, df: pd.DataFrame) -> AgentSignal:
        if len(df) < 15:
            return AgentSignal(0, 0.0, "Insufficient data for momentum")

        df = df.copy()
        # Rate of change over 10 days
        df["roc"] = ta.momentum.roc(df["Close"], window=10)
        # Also look at shorter-term momentum (5 day) for acceleration
        df["roc5"] = ta.momentum.roc(df["Close"], window=5)
        df.dropna(inplace=True)

        if len(df) < 3:
            return AgentSignal(0, 0.0, "Not enough momentum data")

        roc10 = df["roc"].iloc[-1]
        roc5 = df["roc5"].iloc[-1]
        prev_roc10 = df["roc"].iloc[-2]

        # Acceleration: short-term momentum exceeding long-term = strengthening trend
        accelerating = abs(roc5) > abs(roc10) and np.sign(roc5) == np.sign(roc10)

        conf = min(abs(roc10) / 15, 0.85)  # Scale: 15% ROC → max confidence
        if accelerating:
            conf = min(conf + 0.1, 0.9)

        if roc10 > 1:
            accel_note = " (accelerating)" if accelerating else ""
            return AgentSignal(1, conf, f"10d ROC={roc10:.1f}%{accel_note} --bullish momentum")
        elif roc10 < -1:
            accel_note = " (accelerating)" if accelerating else ""
            return AgentSignal(-1, conf, f"10d ROC={roc10:.1f}%{accel_note} --bearish momentum")

        # Momentum near zero but turning
        if roc10 > 0 > prev_roc10:
            return AgentSignal(1, 0.35, f"Momentum turning positive (ROC={roc10:.1f}%)")
        if roc10 < 0 < prev_roc10:
            return AgentSignal(-1, 0.35, f"Momentum turning negative (ROC={roc10:.1f}%)")

        return AgentSignal(0, 0.15, f"Momentum flat (ROC={roc10:.1f}%)")


# ---------------------------------------------------------------------------
# 4. Bollinger Bands Agent --Price position within bands (20, 2)
# ---------------------------------------------------------------------------

class BollingerAgent(BaseAgent):
    name = "Bollinger Bands"

    def analyze(self, df: pd.DataFrame) -> AgentSignal:
        if len(df) < 25:
            return AgentSignal(0, 0.0, "Insufficient data for Bollinger Bands")

        df = df.copy()
        bb = ta.volatility.BollingerBands(df["Close"], window=20, window_dev=2)
        df["bb_upper"] = bb.bollinger_hband()
        df["bb_lower"] = bb.bollinger_lband()
        df["bb_mid"] = bb.bollinger_mavg()
        df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / df["bb_mid"]
        df.dropna(inplace=True)

        if len(df) < 3:
            return AgentSignal(0, 0.0, "Not enough BB data")

        close = df["Close"].iloc[-1]
        upper = df["bb_upper"].iloc[-1]
        lower = df["bb_lower"].iloc[-1]
        mid = df["bb_mid"].iloc[-1]
        band_range = upper - lower

        if band_range == 0:
            return AgentSignal(0, 0.1, "Bollinger band width is zero")

        # Position within band: 0 = at lower, 1 = at upper
        position = (close - lower) / band_range

        # Squeeze detection: unusually narrow bands indicate pending breakout
        avg_width = df["bb_width"].tail(50).mean() if len(df) >= 50 else df["bb_width"].mean()
        curr_width = df["bb_width"].iloc[-1]
        squeeze = curr_width < avg_width * 0.6

        if close < lower:
            conf = min(0.5 + (lower - close) / band_range, 0.85)
            reason = f"Price below lower BB (pos={position:.2f})"
            if squeeze:
                reason += " during squeeze"
            return AgentSignal(1, conf, reason)

        if close > upper:
            conf = min(0.5 + (close - upper) / band_range, 0.85)
            reason = f"Price above upper BB (pos={position:.2f})"
            if squeeze:
                reason += " during squeeze"
            return AgentSignal(-1, conf, reason)

        # Near boundaries
        if position < 0.15:
            return AgentSignal(1, 0.5, f"Price near lower BB (pos={position:.2f})")
        if position > 0.85:
            return AgentSignal(-1, 0.5, f"Price near upper BB (pos={position:.2f})")

        # Middle zone --mild directional lean based on position relative to midline
        if position < 0.4:
            return AgentSignal(1, 0.25, f"Price below BB midline (pos={position:.2f})")
        if position > 0.6:
            return AgentSignal(-1, 0.25, f"Price above BB midline (pos={position:.2f})")

        return AgentSignal(0, 0.1, f"Price at BB midline (pos={position:.2f})")


# ---------------------------------------------------------------------------
# 5. Volume Agent --OBV trend + volume vs 20-day average
# ---------------------------------------------------------------------------

class VolumeAgent(BaseAgent):
    name = "Volume/OBV"

    def analyze(self, df: pd.DataFrame) -> AgentSignal:
        if len(df) < 25 or "Volume" not in df.columns:
            return AgentSignal(0, 0.0, "Insufficient volume data")

        df = df.copy()

        # Skip if volume data is all zero (some instruments lack volume)
        if df["Volume"].sum() == 0:
            return AgentSignal(0, 0.0, "No volume data available for this ticker")

        df["obv"] = ta.volume.on_balance_volume(df["Close"], df["Volume"])
        df["vol_sma20"] = df["Volume"].rolling(20).mean()
        df.dropna(inplace=True)

        if len(df) < 10:
            return AgentSignal(0, 0.0, "Not enough volume history")

        # OBV trend: compare 5-day SMA of OBV to 20-day SMA of OBV
        obv_short = df["obv"].tail(5).mean()
        obv_long = df["obv"].tail(20).mean()
        obv_trend = "up" if obv_short > obv_long else "down"

        # Volume relative to average
        curr_vol = df["Volume"].iloc[-1]
        avg_vol = df["vol_sma20"].iloc[-1]
        vol_ratio = curr_vol / avg_vol if avg_vol > 0 else 1.0

        # Price direction over last 5 days
        price_change = (df["Close"].iloc[-1] - df["Close"].iloc[-5]) / df["Close"].iloc[-5]
        price_up = price_change > 0

        # Volume confirms price: high volume in direction of move
        high_vol = vol_ratio > 1.3

        if obv_trend == "up" and price_up:
            conf = min(0.3 + (vol_ratio - 1) * 0.3, 0.8) if high_vol else 0.4
            return AgentSignal(1, conf, f"OBV trending up, vol={vol_ratio:.1f}x avg --bullish confirmation")
        if obv_trend == "down" and not price_up:
            conf = min(0.3 + (vol_ratio - 1) * 0.3, 0.8) if high_vol else 0.4
            return AgentSignal(-1, conf, f"OBV trending down, vol={vol_ratio:.1f}x avg --bearish confirmation")

        # Divergence: OBV disagrees with price
        if obv_trend == "up" and not price_up:
            conf = 0.5 if high_vol else 0.35
            return AgentSignal(1, conf, f"OBV divergence: OBV up but price down --potential reversal")
        if obv_trend == "down" and price_up:
            conf = 0.5 if high_vol else 0.35
            return AgentSignal(-1, conf, f"OBV divergence: OBV down but price up --potential weakness")

        return AgentSignal(0, 0.15, f"Volume neutral (vol={vol_ratio:.1f}x avg)")


# ---------------------------------------------------------------------------
# 6. MACD Agent --MACD line vs signal line crossovers
# ---------------------------------------------------------------------------

class MACDAgent(BaseAgent):
    name = "MACD"

    def analyze(self, df: pd.DataFrame) -> AgentSignal:
        if len(df) < 35:
            return AgentSignal(0, 0.0, "Insufficient data for MACD")

        df = df.copy()
        macd_obj = ta.trend.MACD(df["Close"], window_slow=26, window_fast=12, window_sign=9)
        df["macd"] = macd_obj.macd()
        df["macd_signal"] = macd_obj.macd_signal()
        df["macd_hist"] = macd_obj.macd_diff()
        df.dropna(inplace=True)

        if len(df) < 3:
            return AgentSignal(0, 0.0, "Not enough MACD data")

        macd_val = df["macd"].iloc[-1]
        signal_val = df["macd_signal"].iloc[-1]
        hist = df["macd_hist"].iloc[-1]
        prev_hist = df["macd_hist"].iloc[-2]

        # Histogram crossing zero = MACD crossing signal line
        cross_up = hist > 0 and prev_hist <= 0
        cross_down = hist < 0 and prev_hist >= 0

        if cross_up:
            return AgentSignal(1, 0.85, "MACD bullish crossover (MACD crossed above signal)")
        if cross_down:
            return AgentSignal(-1, 0.85, "MACD bearish crossover (MACD crossed below signal)")

        # Sustained position above/below signal
        # Histogram magnitude relative to price gives confidence
        price = df["Close"].iloc[-1]
        hist_pct = abs(hist) / price * 100 if price > 0 else 0
        conf = min(0.2 + hist_pct * 5, 0.7)

        # Histogram expanding or contracting
        expanding = abs(hist) > abs(prev_hist)

        if hist > 0:
            note = "expanding" if expanding else "contracting"
            return AgentSignal(1, conf, f"MACD above signal ({note}, hist={hist:.3f})")
        elif hist < 0:
            note = "expanding" if expanding else "contracting"
            return AgentSignal(-1, conf, f"MACD below signal ({note}, hist={hist:.3f})")

        return AgentSignal(0, 0.1, "MACD at signal line --indeterminate")


# ---------------------------------------------------------------------------
# Swarm Consensus --Adaptive weighted voting
# ---------------------------------------------------------------------------

class SwarmConsensus:
    """
    Aggregates signals from all agents using weighted voting.

    Weights are adaptive: each agent starts with equal weight (1.0), and weights
    are updated based on the agent's accuracy over the last N trades. An agent
    that correctly predicted direction gets its weight increased; incorrect
    predictions decrease weight. This creates a self-tuning ensemble.
    """

    def __init__(self, agents: list[BaseAgent], lookback: int = 20):
        self.agents = agents
        self.lookback = lookback
        # Start with equal weights
        self.weights = {agent.name: 1.0 for agent in agents}
        # Track each agent's recent prediction accuracy for weight updates
        self.history: dict[str, list[bool]] = {agent.name: [] for agent in agents}

    def get_signals(self, df: pd.DataFrame) -> dict[str, AgentSignal]:
        """Run all agents and return their individual signals."""
        signals = {}
        for agent in self.agents:
            try:
                signals[agent.name] = agent.analyze(df)
            except Exception as e:
                signals[agent.name] = AgentSignal(0, 0.0, f"Error: {e}")
        return signals

    def consensus(self, df: pd.DataFrame) -> tuple[dict[str, AgentSignal], float, float]:
        """
        Run all agents, compute weighted consensus.

        Returns: (signals_dict, consensus_score, aggregate_confidence)
        - consensus_score: -1.0 to +1.0 (weighted average of agent signals)
        - aggregate_confidence: 0.0 to 1.0 (how much the swarm agrees)
        """
        signals = self.get_signals(df)

        # Weighted vote: each agent's signal * confidence * weight
        weighted_sum = 0.0
        total_weight = 0.0
        for agent_name, sig in signals.items():
            w = self.weights[agent_name]
            weighted_sum += sig.signal * sig.confidence * w
            total_weight += sig.confidence * w

        # Consensus score normalized to [-1, 1]
        if total_weight > 0:
            consensus_score = weighted_sum / total_weight
        else:
            consensus_score = 0.0

        # Aggregate confidence: how strongly the swarm agrees
        # High when agents agree with high individual confidence
        if len(signals) > 0:
            agreement = np.mean([
                1.0 if (sig.signal * np.sign(consensus_score)) > 0 else 0.0
                for sig in signals.values()
                if sig.signal != 0
            ]) if any(s.signal != 0 for s in signals.values()) else 0.0
            avg_conf = np.mean([s.confidence for s in signals.values()])
            aggregate_confidence = agreement * avg_conf
        else:
            aggregate_confidence = 0.0

        return signals, consensus_score, aggregate_confidence

    def update_weights(self, agent_name: str, was_correct: bool):
        """
        Update an agent's weight based on prediction outcome.
        Uses the last `lookback` predictions to compute accuracy,
        then sets weight = 0.5 + accuracy (range: 0.5 – 1.5).
        """
        self.history[agent_name].append(was_correct)
        # Keep only last N entries
        self.history[agent_name] = self.history[agent_name][-self.lookback:]
        recent = self.history[agent_name]
        if len(recent) >= 3:
            accuracy = sum(recent) / len(recent)
            self.weights[agent_name] = 0.5 + accuracy  # range [0.5, 1.5]


# ---------------------------------------------------------------------------
# Market Data Fetcher
# ---------------------------------------------------------------------------

def fetch_data(ticker: str, period: str = "1y") -> Optional[pd.DataFrame]:
    """Download OHLCV data from yfinance. Returns None on failure."""
    try:
        t = yf.Ticker(ticker)
        df = t.history(period=period)
        if df is None or df.empty:
            return None
        # Standardize column names
        df.columns = [c.strip() for c in df.columns]
        return df
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Backtest Engine --Walk-forward, no look-ahead bias
# ---------------------------------------------------------------------------

class BacktestEngine:
    """
    Walk-forward backtest: at each bar, the swarm only sees data up to that
    point (no future data). Trades are executed at next bar's open to avoid
    look-ahead bias.

    Position sizing: all-in / all-out for simplicity.
    Transaction costs: 0.1% per trade (round trip).
    """

    TRADE_COST = 0.001  # 0.1% per trade

    def __init__(self, swarm: SwarmConsensus, buy_threshold: float = 0.25,
                 sell_threshold: float = -0.25):
        self.swarm = swarm
        self.buy_threshold = buy_threshold
        self.sell_threshold = sell_threshold

    def run(self, df: pd.DataFrame, warmup: int = 60) -> dict:
        """
        Run walk-forward backtest.

        Args:
            df: Full OHLCV DataFrame
            warmup: Minimum bars before first trade (agents need history)

        Returns dict with: trades, win_rate, sharpe_ratio, max_drawdown,
                          total_return, agent_weights
        """
        if len(df) < warmup + 10:
            return {"error": "Insufficient data for backtest"}

        trades = []
        equity_curve = [1.0]
        position = 0  # 0 = flat, 1 = long
        entry_price = 0.0
        entry_idx = 0
        prev_close = df["Close"].iloc[warmup - 1]

        for i in range(warmup, len(df) - 1):
            # Agent only sees data up to current bar (walk-forward)
            window = df.iloc[:i + 1]
            _, score, confidence = self.swarm.consensus(window)

            current_price = df["Close"].iloc[i]
            next_open = df["Open"].iloc[i + 1] if i + 1 < len(df) else current_price

            if position == 0 and score >= self.buy_threshold and confidence > 0.2:
                # Enter long at next bar's open
                position = 1
                entry_price = next_open * (1 + self.TRADE_COST)
                entry_idx = i + 1

            elif position == 1 and score <= self.sell_threshold:
                # Exit at next bar's open
                exit_price = next_open * (1 - self.TRADE_COST)
                pnl = (exit_price - entry_price) / entry_price
                trades.append({
                    "entry_date": str(df.index[entry_idx].date()) if hasattr(df.index[entry_idx], "date") else str(df.index[entry_idx]),
                    "exit_date": str(df.index[i + 1].date()) if hasattr(df.index[i + 1], "date") else str(df.index[i + 1]),
                    "entry_price": round(entry_price, 2),
                    "exit_price": round(exit_price, 2),
                    "pnl_pct": round(pnl * 100, 2),
                    "score_at_exit": round(score, 3),
                })

                # Update agent weights based on trade outcome
                was_profitable = pnl > 0
                signals = self.swarm.get_signals(window)
                for agent_name, sig in signals.items():
                    if sig.signal != 0:
                        # Agent was "correct" if its signal direction matched the trade outcome
                        agent_correct = (sig.signal > 0) == was_profitable
                        self.swarm.update_weights(agent_name, agent_correct)

                position = 0
                entry_price = 0.0

            # Track equity using daily returns (not cumulative unrealized)
            if position == 1:
                daily_ret = (current_price - prev_close) / prev_close
                equity_curve.append(equity_curve[-1] * (1 + daily_ret))
            else:
                equity_curve.append(equity_curve[-1])

            prev_close = current_price

        # Close any open position at last price
        if position == 1:
            exit_price = df["Close"].iloc[-1] * (1 - self.TRADE_COST)
            pnl = (exit_price - entry_price) / entry_price
            trades.append({
                "entry_date": str(df.index[entry_idx].date()) if hasattr(df.index[entry_idx], "date") else str(df.index[entry_idx]),
                "exit_date": str(df.index[-1].date()) if hasattr(df.index[-1], "date") else str(df.index[-1]),
                "entry_price": round(entry_price, 2),
                "exit_price": round(exit_price, 2),
                "pnl_pct": round(pnl * 100, 2),
                "score_at_exit": 0,
            })

        # --- Compute statistics ---
        equity = np.array(equity_curve)
        total_return = (equity[-1] / equity[0] - 1) * 100

        # Win rate
        if trades:
            wins = sum(1 for t in trades if t["pnl_pct"] > 0)
            win_rate = wins / len(trades) * 100
        else:
            win_rate = 0.0

        # Sharpe ratio (annualized, assuming 252 trading days)
        daily_returns = np.diff(equity) / equity[:-1]
        if len(daily_returns) > 1 and np.std(daily_returns) > 0:
            sharpe = np.mean(daily_returns) / np.std(daily_returns) * np.sqrt(252)
        else:
            sharpe = 0.0

        # Max drawdown
        peak = np.maximum.accumulate(equity)
        drawdown = (peak - equity) / peak
        max_dd = np.max(drawdown) * 100

        return {
            "total_trades": len(trades),
            "win_rate": round(win_rate, 1),
            "total_return": round(total_return, 2),
            "sharpe_ratio": round(sharpe, 2),
            "max_drawdown": round(max_dd, 2),
            "trades": trades,
            "agent_weights": dict(self.swarm.weights),
            "equity_curve": equity.tolist(),
        }


# ---------------------------------------------------------------------------
# Live Scanner --Scans tickers and ranks by swarm consensus
# ---------------------------------------------------------------------------

class LiveScanner:
    """Scans a list of tickers and returns ranked buy/sell signals."""

    def __init__(self, swarm: SwarmConsensus):
        self.swarm = swarm

    def scan(self, tickers: list[str], period: str = "6mo") -> list[dict]:
        """
        Scan tickers and return results sorted by absolute consensus strength.
        """
        results = []
        for ticker in tickers:
            df = fetch_data(ticker, period=period)
            if df is None or len(df) < 30:
                results.append({
                    "ticker": ticker,
                    "consensus": 0.0,
                    "confidence": 0.0,
                    "action": "ERROR",
                    "agents": {},
                    "price": 0.0,
                    "error": "Failed to fetch data",
                })
                continue

            signals, score, confidence = self.swarm.consensus(df)
            price = df["Close"].iloc[-1]

            if score >= 0.25:
                action = "BUY"
            elif score <= -0.25:
                action = "SELL"
            else:
                action = "HOLD"

            agent_details = {}
            for name, sig in signals.items():
                label = "BUY" if sig.signal > 0 else ("SELL" if sig.signal < 0 else "HOLD")
                agent_details[name] = {
                    "signal": sig.signal,
                    "confidence": round(sig.confidence, 2),
                    "label": label,
                    "reason": sig.reason,
                }

            results.append({
                "ticker": ticker,
                "consensus": round(score, 3),
                "confidence": round(confidence, 3),
                "action": action,
                "agents": agent_details,
                "price": round(price, 2),
            })

        # Sort by absolute consensus strength (strongest signals first)
        results.sort(key=lambda x: abs(x.get("consensus", 0)), reverse=True)
        return results


# ---------------------------------------------------------------------------
# Factory: create a fully-initialized swarm with all 6 agents
# ---------------------------------------------------------------------------

def create_swarm() -> SwarmConsensus:
    agents = [
        SMAAgent(),
        RSIAgent(),
        MomentumAgent(),
        BollingerAgent(),
        VolumeAgent(),
        MACDAgent(),
    ]
    return SwarmConsensus(agents)
