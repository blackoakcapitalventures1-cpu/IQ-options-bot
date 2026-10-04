#!/usr/bin/env python3
"""
IQ Option x Claude: market analysis + signal generation bot.

Flow each cycle:
  1. Wait for a new candle to close
  2. Pull candles from IQ Option, compute indicators locally
  3. Send a compact market snapshot to Claude, which returns a JSON signal
  4. Apply risk rules; optionally place the trade (practice account by default)
  5. Log everything to CSV

Install:
  pip install anthropic pandas numpy
  pip install git+https://github.com/Lu-Yi-Hsun/iqoptionapi.git   # unofficial API

Env vars:
  IQ_EMAIL, IQ_PASSWORD, ANTHROPIC_API_KEY

Examples:
  python iq_claude_bot.py --assets EURUSD GBPUSD --dry-run
  python iq_claude_bot.py --assets EURUSD --stake 1 --expiry 1          # PRACTICE account
  python iq_claude_bot.py --assets EURUSD --live                         # REAL money (asks to confirm)

WARNING: Binary/digital options carry a high risk of loss. Payouts below 100% mean you
need a win rate well above 50% to break even. An LLM has no inherent predictive edge;
treat its output as a structured second opinion, not a guarantee. Test on PRACTICE first.
This is not financial advice. Check that your use of unofficial APIs is permitted by
IQ Option's terms and your local regulations.
"""

import argparse
import csv
import json
import logging
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import anthropic
from iqoptionapi.stable_api import IQ_Option

MODEL = "claude-sonnet-5-5"
LOG_FILE = Path("signals_log.csv")

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("iq-claude")
logging.getLogger("iqoptionapi").setLevel(logging.ERROR)


# --------------------------------------------------------------------------- #
# Indicators
# --------------------------------------------------------------------------- #
def ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False).mean()


def rsi(s: pd.Series, n: int = 14) -> pd.Series:
    d = s.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    rs = up / dn.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def macd(s: pd.Series):
    line = ema(s, 12) - ema(s, 26)
    sig = ema(line, 9)
    return line, sig, line - sig


def bollinger(s: pd.Series, n: int = 20, k: float = 2.0):
    mid = s.rolling(n).mean()
    sd = s.rolling(n).std()
    return mid + k * sd, mid, mid - k * sd


def atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    pc = df["close"].shift()
    tr = pd.concat(
        [df["max"] - df["min"], (df["max"] - pc).abs(), (df["min"] - pc).abs()], axis=1
    ).max(axis=1)
    return tr.ewm(alpha=1 / n, adjust=False).mean()


def build_snapshot(candles: list, asset: str, tf_sec: int) -> dict:
    df = pd.DataFrame(candles)
    df = df.iloc[:-1].copy()  # drop the still-forming candle
    c = df["close"]
    df["ema9"], df["ema21"], df["ema50"] = ema(c, 9), ema(c, 21), ema(c, 50)
    df["rsi"] = rsi(c)
    df["macd"], df["macd_sig"], df["macd_hist"] = macd(c)
    df["bb_up"], df["bb_mid"], df["bb_lo"] = bollinger(c)
    df["atr"] = atr(df)
    last = df.iloc[-1]

    recent = df.tail(12)[["open", "max", "min", "close"]].round(5)
    recent_rows = [
        f"O={r.open} H={r.max} L={r.min} C={r.close}" for r in recent.itertuples()
    ]
    bb_pos = (last.close - last.bb_lo) / (last.bb_up - last.bb_lo + 1e-12)

    return {
        "asset": asset,
        "timeframe_sec": tf_sec,
        "last_close": round(float(last.close), 5),
        "ema9": round(float(last.ema9), 5),
        "ema21": round(float(last.ema21), 5),
        "ema50": round(float(last.ema50), 5),
        "rsi14": round(float(last.rsi), 2),
        "macd": round(float(last.macd), 6),
        "macd_signal": round(float(last.macd_sig), 6),
        "macd_hist": round(float(last.macd_hist), 6),
        "bollinger_position_0_to_1": round(float(bb_pos), 3),
        "atr14": round(float(last.atr), 6),
        "atr_pct_of_price": round(float(last.atr / last.close * 100), 4),
        "last_12_candles_oldest_to_newest": recent_rows,
    }


# --------------------------------------------------------------------------- #
# Claude
# --------------------------------------------------------------------------- #
SYSTEM_PROMPT = """You are a disciplined technical analyst assisting an options trading bot.
You receive a snapshot of closed candles and indicators for one asset. Decide whether there is
a reasonably clear short-term directional edge over the stated expiry.

Rules:
- Default to NO_TRADE. Only signal CALL (price higher at expiry) or PUT (price lower) when
  several independent factors agree (trend alignment, momentum, location vs. Bollinger bands,
  recent candle structure).
- Be skeptical of choppy/ranging conditions, conflicting indicators, and extreme RSI that
  could mean either reversal or continuation.
- Confidence is your honest 0-100 estimate that the signal is correct. Do not inflate it.
- Respond with ONLY a JSON object, no prose, no markdown fences:
  {"signal": "CALL" | "PUT" | "NO_TRADE", "confidence": <int 0-100>,
   "market_regime": "trending_up" | "trending_down" | "ranging" | "volatile",
   "reasoning": "<max 2 sentences>"}"""


def ask_claude(client: anthropic.Anthropic, snap: dict, expiry_min: int) -> dict:
    user_msg = (
        f"Expiry: {expiry_min} minute(s).\n"
        f"Snapshot:\n{json.dumps(snap, indent=2)}"
    )
    resp = client.messages.create(
        model=MODEL,
        max_tokens=400,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_msg}],
    )
    text = "".join(b.text for b in resp.content if b.type == "text")
    return parse_signal(text)


def parse_signal(text: str) -> dict:
    fallback = {"signal": "NO_TRADE", "confidence": 0, "market_regime": "unknown",
                "reasoning": "unparseable response"}
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return fallback
    try:
        d = json.loads(m.group(0))
        sig = str(d.get("signal", "NO_TRADE")).upper()
        if sig not in ("CALL", "PUT", "NO_TRADE"):
            return fallback
        d["signal"] = sig
        d["confidence"] = max(0, min(100, int(d.get("confidence", 0))))
        d.setdefault("market_regime", "unknown")
        d.setdefault("reasoning", "")
        return d
    except (json.JSONDecodeError, ValueError, TypeError):
        return fallback


# --------------------------------------------------------------------------- #
# Risk management
# --------------------------------------------------------------------------- #
class RiskManager:
    def __init__(self, args, start_balance: float):
        self.a = args
        self.start_balance = start_balance
        self.day = datetime.now(timezone.utc).date()
        self.trades_today = 0
        self.pnl_today = 0.0
        self.loss_streak = 0

    def _roll_day(self):
        today = datetime.now(timezone.utc).date()
        if today != self.day:
            self.day, self.trades_today, self.pnl_today, self.loss_streak = today, 0, 0.0, 0

    def can_trade(self) -> tuple[bool, str]:
        self._roll_day()
        if self.trades_today >= self.a.max_trades:
            return False, "max trades per day reached"
        if self.pnl_today <= -abs(self.a.max_daily_loss):
            return False, "daily loss limit hit"
        if self.loss_streak >= self.a.max_loss_streak:
            return False, "max consecutive losses reached"
        return True, "ok"

    def stake(self, balance: float) -> float:
        if self.a.stake_pct > 0:
            s = balance * self.a.stake_pct / 100
        else:
            s = self.a.stake
        return round(max(1.0, min(s, self.a.max_stake)), 2)

    def record(self, profit: float):
        self.trades_today += 1
        self.pnl_today += profit
        self.loss_streak = self.loss_streak + 1 if profit < 0 else 0


# --------------------------------------------------------------------------- #
# Logging
# --------------------------------------------------------------------------- #
def log_row(row: dict):
    new = not LOG_FILE.exists()
    with LOG_FILE.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if new:
            w.writeheader()
        w.writerow(row)


# --------------------------------------------------------------------------- #
# Main loop
# --------------------------------------------------------------------------- #
def main():
    p = argparse.ArgumentParser(description="IQ Option + Claude signal bot")
    p.add_argument("--assets", nargs="+", default=["EURUSD"])
    p.add_argument("--timeframe", type=int, default=60, help="candle size in seconds")
    p.add_argument("--expiry", type=int, default=1, help="expiry in minutes (1-5)")
    p.add_argument("--candles", type=int, default=120)
    p.add_argument("--stake", type=float, default=1.0, help="fixed stake")
    p.add_argument("--stake-pct", type=float, default=0.0, help="stake as %% of balance (overrides --stake)")
    p.add_argument("--max-stake", type=float, default=10.0)
    p.add_argument("--min-confidence", type=int, default=70)
    p.add_argument("--min-payout", type=float, default=0.80)
    p.add_argument("--max-trades", type=int, default=20)
    p.add_argument("--max-daily-loss", type=float, default=10.0)
    p.add_argument("--max-loss-streak", type=int, default=3)
    p.add_argument("--dry-run", action="store_true", help="signals only, never trade")
    p.add_argument("--live", action="store_true", help="use REAL account (default PRACTICE)")
    args = p.parse_args()

    email, pwd = os.getenv("IQ_EMAIL"), os.getenv("IQ_PASSWORD")
    if not (email and pwd and os.getenv("ANTHROPIC_API_KEY")):
        sys.exit("Set IQ_EMAIL, IQ_PASSWORD and ANTHROPIC_API_KEY environment variables.")

    if args.live and not args.dry_run:
        if input("REAL MONEY mode. Type 'I ACCEPT THE RISK' to continue: ") != "I ACCEPT THE RISK":
            sys.exit("Aborted.")

    iq = IQ_Option(email, pwd)
    ok, reason = iq.connect()
    if not ok:
        sys.exit(f"IQ Option connection failed: {reason}")
    iq.change_balance("REAL" if args.live else "PRACTICE")
    balance = iq.get_balance()
    log.info("Connected. Account=%s balance=%.2f", "REAL" if args.live else "PRACTICE", balance)

    client = anthropic.Anthropic()
    risk = RiskManager(args, balance)
    instrument = "turbo" if args.expiry <= 5 else "binary"

    try:
        while True:
            # Wait for the next candle boundary (+2s so the candle is finalized)
            now = time.time()
            time.sleep(args.timeframe - (now % args.timeframe) + 2)

            if not iq.check_connect():
                log.warning("Connection lost, reconnecting...")
                iq.connect()
                iq.change_balance("REAL" if args.live else "PRACTICE")

            ok_trade, why = risk.can_trade()
            if not ok_trade and not args.dry_run:
                log.warning("Trading halted: %s. Stopping.", why)
                break

            payouts = iq.get_all_profit()

            for asset in args.assets:
                try:
                    candles = iq.get_candles(asset, args.timeframe, args.candles, time.time())
                    if len(candles) < 60:
                        log.warning("%s: not enough candles", asset)
                        continue

                    snap = build_snapshot(candles, asset, args.timeframe)
                    sig = ask_claude(client, snap, args.expiry)
                    payout = payouts.get(asset, {}).get(instrument, 0) or 0

                    log.info("%s | %s (%d%%) | %s | payout=%.0f%% | %s",
                             asset, sig["signal"], sig["confidence"], sig["market_regime"],
                             payout * 100, sig["reasoning"])

                    row = {
                        "utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                        "asset": asset, "signal": sig["signal"], "confidence": sig["confidence"],
                        "regime": sig["market_regime"], "payout": payout,
                        "price": snap["last_close"], "reasoning": sig["reasoning"],
                        "traded": False, "stake": "", "result": "",
                    }

                    actionable = (
                        sig["signal"] in ("CALL", "PUT")
                        and sig["confidence"] >= args.min_confidence
                        and payout >= args.min_payout
                    )
                    if actionable and not args.dry_run and ok_trade:
                        balance = iq.get_balance()
                        stake = risk.stake(balance)
                        status, order_id = iq.buy(stake, asset, sig["signal"].lower(), args.expiry)
                        if status:
                            log.info("Order %s placed: %s %s $%.2f", order_id, asset, sig["signal"], stake)
                            profit = iq.check_win_v3(order_id)  # blocks until expiry
                            risk.record(profit)
                            log.info("Result: %+.2f | day PnL %+.2f | streak %d",
                                     profit, risk.pnl_today, risk.loss_streak)
                            row.update(traded=True, stake=stake, result=round(profit, 2))
                            ok_trade, why = risk.can_trade()
                            log_row(row)
                            break  # one trade at a time; re-sync to next candle
                        else:
                            log.error("Order rejected for %s: %s", asset, order_id)
                    log_row(row)

                except Exception as e:  # keep the bot alive on per-asset errors
                    log.exception("Error processing %s: %s", asset, e)

    except KeyboardInterrupt:
        log.info("Stopped by user.")
    finally:
        log.info("Session PnL: %+.2f over %d trades", risk.pnl_today, risk.trades_today)


if __name__ == "__main__":
    main()
