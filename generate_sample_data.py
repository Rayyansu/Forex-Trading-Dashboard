"""Generate a realistic demo journal so the dashboard has something to chew on.

The generator intentionally bakes in behavioural signal, not white noise:
  * FOMO / Revengeful states have a materially worse hit rate and fatter losses.
  * A tilt mechanic raises the chance of an oversized, rule-breaking trade
    immediately after a loss.
  * One weekday and one hour band are deliberately unprofitable.
  * Execution rating is loosely (not perfectly) correlated with realised R.

Run:  python generate_sample_data.py [seed]
"""
from __future__ import annotations

import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from schema import MISTAKE_VOCAB, SETUP_VOCAB, pip_size

ROOT = Path(__file__).resolve().parent

DEFAULT_SEED = 20260809
RNG = np.random.default_rng(DEFAULT_SEED)


def set_seed(seed: int) -> None:
    """Rebind the module RNG so a specific demo journal can be reproduced."""
    global RNG
    RNG = np.random.default_rng(seed)

INSTRUMENTS = {
    # symbol: (typical price, daily vol in price units, money per price unit @0.10 lot)
    "XAUUSD": (2350.0, 22.0, 10.0),
    "EURUSD": (1.0850, 0.0060, 10000.0),
    "GBPJPY": (196.50, 1.20, 65.0),
    "USDJPY": (156.80, 0.90, 65.0),
    "US500": (5450.0, 45.0, 1.0),
    "TASI": (11800.0, 90.0, 0.5),
}

SETUPS = SETUP_VOCAB[:8]
SETUP_QUALITY = {  # base edge per setup: probability of hitting target
    "Order Block": 0.58, "Breaker Block": 0.54, "FVG Fill": 0.50,
    "Liquidity Sweep": 0.61, "BOS Continuation": 0.56, "CHoCH Reversal": 0.47,
    "Momentum": 0.52, "Breakout": 0.44,
}
GOOD_STATES = ["Confident", "Calm", "Patient", "Neutral"]
BAD_STATES = ["FOMO", "Anxious", "Frustrated", "Revengeful", "Greedy", "Bored"]

NOTE_BANK_GOOD = [
    "Waited for the 4H order block to be swept before entering. Entry on the 15m CHoCH.",
    "Clean liquidity grab below Asia low, entered on displacement. Managed to partial at 2R.",
    "Followed the plan exactly. No adjustment to stop, target hit while I was away from screen.",
    "Higher timeframe bias was clear, only took the setup in line with the daily trend.",
    "Small size because spread was wide before the news. Still respected the invalidation.",
    "Patience paid off — sat on hands for two hours, one A+ setup, took it, done for the day.",
]
NOTE_BANK_BAD = [
    "Entered before the sweep completed. Impatient, price took my stop then went my way.",
    "Revenge entry within minutes of the previous loss. Doubled size to get it back. Stupid.",
    "Moved the stop further away because I 'knew' it would come back. It did not.",
    "Chased the candle after it already ran 40 points. No structure, pure FOMO.",
    "Closed at +0.4R out of fear while the target was still 2R away. Price hit target later.",
    "Traded through the news release without checking the calendar. Slipped badly.",
    "Fifth trade of the day, no setup, just boredom. Should have shut the platform.",
    "Ignored the daily bias and took the counter-trend setup because the 5m looked pretty.",
]


def session_hour() -> int:
    """Entry hours concentrated around London and New York opens."""
    pool = [7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 20, 22]
    weights = np.array([6, 12, 14, 10, 7, 8, 13, 12, 9, 5, 3, 2], dtype=float)
    return int(RNG.choice(pool, p=weights / weights.sum()))


def make_trades(n: int = 240, start: datetime | None = None,
                seed: int = DEFAULT_SEED) -> pd.DataFrame:
    set_seed(seed)
    start = start or (datetime.now() - timedelta(days=180))
    rows: list[dict] = []
    day, tilt, last_was_loss = start, 0, False

    while len(rows) < n:
        day += timedelta(days=1)
        if day.weekday() >= 5:          # no weekend trading
            continue
        # Trade count per day: usually 1-2, occasionally a revenge-driven cluster.
        count = int(RNG.choice([0, 1, 2, 3, 5], p=[0.18, 0.34, 0.28, 0.14, 0.06]))
        for _ in range(count):
            if len(rows) >= n:
                break

            symbol = str(RNG.choice(list(INSTRUMENTS), p=[0.30, 0.20, 0.14, 0.10, 0.16, 0.10]))
            price0, vol, mpu = INSTRUMENTS[symbol]
            setup = str(RNG.choice(SETUPS))
            hour = session_hour()

            # --- Emotional state: tilt makes bad states far more likely ------ #
            bad_prob = 0.22 + 0.35 * min(tilt, 2) / 2
            state = str(RNG.choice(BAD_STATES if RNG.random() < bad_prob else GOOD_STATES))
            in_bad_state = state in BAD_STATES

            # --- Edge model -------------------------------------------------- #
            p_win = SETUP_QUALITY[setup]
            p_win += 0.05 if not in_bad_state else -0.17
            if day.weekday() == 2:          # deliberately weak Wednesday
                p_win -= 0.09
            if hour in (20, 22):            # late-session degradation
                p_win -= 0.11
            p_win = float(np.clip(p_win, 0.15, 0.82))

            direction = "Buy" if RNG.random() < 0.52 else "Sell"
            sign = 1 if direction == "Buy" else -1
            entry = float(price0 * (1 + RNG.normal(0, 0.012)))

            # Stop distance ~ 0.4-1.1 daily vol; target set by planned R:R.
            stop_dist = float(vol * RNG.uniform(0.35, 1.05))
            planned_rr = float(RNG.uniform(1.2, 3.2) if not in_bad_state
                               else RNG.uniform(0.7, 1.8))
            sl = entry - sign * stop_dist
            tp = entry + sign * stop_dist * planned_rr

            # Position size: oversized when tilted.
            lots = float(np.round(RNG.uniform(0.05, 0.22) * (1.7 if tilt >= 2 else 1.0), 2))
            size_factor = lots / 0.10

            win = RNG.random() < p_win
            if win:
                # Winners rarely capture the full target: early exits are common,
                # more so in a poor state.
                capture = RNG.uniform(0.45, 1.0) if not in_bad_state else RNG.uniform(0.25, 0.75)
                r_realised = planned_rr * capture
            else:
                # Losers occasionally exceed 1R: moved stops and slippage.
                r_realised = -RNG.uniform(0.85, 1.02)
                if in_bad_state and RNG.random() < 0.28:
                    r_realised = -RNG.uniform(1.2, 2.0)

            exit_price = entry + sign * stop_dist * r_realised
            gross = r_realised * stop_dist * mpu * size_factor
            commission = round(3.5 * size_factor, 2)
            swap = round(float(RNG.normal(-0.4, 1.1)) * size_factor, 2)

            # --- Mistake tags follow the story ------------------------------- #
            tags: list[str] = []
            if state == "Revengeful":
                tags.append("Revenge Trade")
            if state == "FOMO":
                tags += ["Chased Price", "Late Entry"]
            if state == "Bored":
                tags.append("No Setup / Impulse")
            if tilt >= 2:
                tags.append("Oversized Position")
            if not win and r_realised < -1.15:
                tags.append("Moved Stop Loss")
            if win and r_realised < planned_rr * 0.5:
                tags.append("Early Exit")
            if hour in (20, 22) and not win:
                tags.append("Traded During News")
            if count >= 4:
                tags.append("Overtrading")
            tags = sorted(set(tags))[:3]

            rating = int(np.clip(round(RNG.normal(4.1 if not in_bad_state else 2.2, 0.8)), 1, 5))
            note = str(RNG.choice(NOTE_BANK_BAD if (in_bad_state or tags)
                                  else NOTE_BANK_GOOD))

            open_time = day.replace(hour=hour, minute=int(RNG.integers(0, 59)),
                                    second=0, microsecond=0)
            hold = float(np.clip(RNG.lognormal(3.9, 0.9), 6, 1400))  # minutes
            close_time = open_time + timedelta(minutes=hold)

            decimals = 5 if pip_size(symbol) == 0.0001 else 2
            rows.append({
                "ticket": f"{10_000_000 + len(rows) * 7 + int(RNG.integers(1, 6))}",
                "open_time": open_time.strftime("%Y-%m-%d %H:%M:%S"),
                "close_time": close_time.strftime("%Y-%m-%d %H:%M:%S"),
                "symbol": symbol,
                "direction": direction,
                "lots": lots,
                "entry_price": round(entry, decimals),
                "exit_price": round(exit_price, decimals),
                "stop_loss": round(sl, decimals),
                "take_profit": round(tp, decimals),
                "gross_pnl": round(gross, 2),
                "commission": commission,
                "swap": swap,
                "net_pnl": "",
                "pips": "",
                "setup": setup,
                "emotion": state,
                "execution_rating": rating,
                "mistakes": "; ".join(tags),
                "notes": note,
            })

            last_was_loss = not win
            tilt = min(tilt + 1, 3) if last_was_loss else 0

    return pd.DataFrame(rows)


if __name__ == "__main__":
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_SEED
    out = ROOT / "data" / "sample_trades.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df = make_trades(240, seed=seed)
    df.to_csv(out, index=False)
    print(f"Wrote {len(df)} trades (seed {seed}) → {out}")
