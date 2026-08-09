"""Trade schema, ingestion and feature engineering.

Responsibilities
----------------
1. Define the canonical column set every downstream module can rely on.
2. Map messy broker exports (MT4/MT5/cTrader/TradingView) onto that schema.
3. Engineer the derived fields the analytics need: net P&L, pips, R-multiple,
   planned RR, outcome, and calendar buckets.

Quant conventions used here (documented once, applied everywhere):
  * `commission` is always treated as a cost -> subtracted as abs().
  * `swap` is signed -> it can legitimately be a credit.
  * `net_pnl = gross_pnl - |commission| + swap` when net is not supplied.
  * Risk is derived from the *implied money-per-price-unit* of each trade so the
    journal stays instrument-agnostic (works for FX, gold, indices, TASI equities)
    without needing a contract-specification table.
"""
from __future__ import annotations

import io
import re
from typing import Iterable

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------- #
# Canonical schema
# --------------------------------------------------------------------------- #
TIME_COLS = ["open_time", "close_time"]
NUM_COLS = [
    "lots", "entry_price", "exit_price", "stop_loss", "take_profit",
    "gross_pnl", "commission", "swap", "net_pnl", "pips", "execution_rating",
]
TEXT_COLS = ["ticket", "symbol", "direction", "setup", "emotion", "mistakes", "notes"]
CANONICAL_COLUMNS = ["ticket"] + TIME_COLS + ["symbol", "direction"] + [
    "lots", "entry_price", "exit_price", "stop_loss", "take_profit",
    "gross_pnl", "commission", "swap", "net_pnl", "pips",
    "setup", "emotion", "execution_rating", "mistakes", "notes",
]

# Controlled vocabularies -- editable by the user in the UI.
SETUP_VOCAB = [
    "Order Block", "Breaker Block", "FVG Fill", "Liquidity Sweep", "BOS Continuation",
    "CHoCH Reversal", "Momentum", "Breakout", "Swing", "Mean Reversion", "News Fade",
]
EMOTION_VOCAB = [
    "Confident", "Calm", "Patient", "Neutral", "Hesitant",
    "Anxious", "FOMO", "Greedy", "Frustrated", "Revengeful", "Bored",
]
MISTAKE_VOCAB = [
    "Moved Stop Loss", "Revenge Trade", "No Setup / Impulse", "Oversized Position",
    "Early Exit", "Late Entry", "Chased Price", "Ignored Higher Timeframe",
    "Traded During News", "Overtrading", "Ignored Risk Limit", "Widened Target",
]

# Pip / point size per instrument family. Extend freely.
PIP_SIZE = {
    "JPY": 0.01,       # any *JPY pair
    "XAU": 0.10,       # gold: 1 pip = 0.10
    "XAG": 0.01,
    "BTC": 1.0,
    "ETH": 0.1,
    "INDEX": 1.0,      # US500, US30, NAS100, TASI ...
    "DEFAULT": 0.0001,  # standard FX
}
_INDEX_HINTS = ("US30", "US500", "SPX", "NAS", "GER", "DAX", "UK100", "JP225",
                "TASI", "HSI", "DJ", "NDX", "STOXX")

# --------------------------------------------------------------------------- #
# Alias mapping (broker exports -> canonical)
# --------------------------------------------------------------------------- #
ALIASES: dict[str, str] = {}


def _register(canonical: str, aliases: Iterable[str]) -> None:
    for a in aliases:
        ALIASES[_key(a)] = canonical


def _key(s: str) -> str:
    """Normalise a header for fuzzy matching: lowercase, alphanumerics only."""
    return re.sub(r"[^a-z0-9]", "", str(s).lower())


_register("ticket", ["ticket", "ticket no", "ticket number", "id", "order", "deal",
                     "trade id", "order id", "#"])
_register("open_time", ["open time", "open", "entry time", "time open", "open date",
                        "entry date", "datetime", "date", "opened"])
_register("close_time", ["close time", "close", "exit time", "time close",
                         "close date", "exit date", "closed"])
_register("symbol", ["symbol", "item", "instrument", "pair", "asset", "market", "ticker"])
_register("direction", ["type", "direction", "side", "buy sell", "trade type", "action",
                        "long short"])
_register("lots", ["lots", "lot", "volume", "size", "lot size", "qty", "quantity",
                   "contracts", "units"])
_register("entry_price", ["entry price", "open price", "entry", "price open", "price",
                          "avg entry"])
_register("exit_price", ["exit price", "close price", "exit", "price close", "avg exit"])
_register("stop_loss", ["stop loss", "sl", "s/l", "stop"])
_register("take_profit", ["take profit", "tp", "t/p", "target"])
_register("gross_pnl", ["gross pnl", "gross profit", "profit", "pnl", "p&l", "gross",
                        "p/l", "result", "gross p&l"])
_register("commission", ["commission", "commissions", "fees", "fee", "cost", "charges"])
_register("swap", ["swap", "rollover", "interest", "financing"])
_register("net_pnl", ["net pnl", "net profit", "net", "net result", "net p&l", "realized"])
_register("pips", ["pips", "points", "pips gained", "pip"])
_register("setup", ["setup", "strategy", "system", "model", "playbook", "pattern"])
_register("emotion", ["emotion", "feeling", "mood", "psychology", "state"])
_register("execution_rating", ["execution rating", "rating", "grade", "execution",
                               "quality", "score", "discipline"])
_register("mistakes", ["mistakes", "mistake", "tags", "tag", "errors", "error",
                       "mistake tags"])
_register("notes", ["notes", "note", "comment", "comments", "journal", "remarks",
                    "reason", "review"])


# --------------------------------------------------------------------------- #
# Ingestion
# --------------------------------------------------------------------------- #
def empty_frame() -> pd.DataFrame:
    """An empty, correctly-typed journal -- used as the cold-start state."""
    df = pd.DataFrame({c: pd.Series(dtype="object") for c in CANONICAL_COLUMNS})
    for c in NUM_COLS:
        df[c] = pd.Series(dtype="float64")
    for c in TIME_COLS:
        df[c] = pd.Series(dtype="datetime64[ns]")
    return df


def map_columns(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, str]]:
    """Rename incoming columns onto the canonical schema.

    Returns the renamed frame plus the mapping actually applied, so the UI can
    show the user exactly how their file was interpreted.
    """
    mapping: dict[str, str] = {}
    used: set[str] = set()
    for col in df.columns:
        key = _key(col)
        # pandas renames duplicate headers ("Price", "Price.1"). MT5 statements
        # rely on that duplication, so fall back to the de-suffixed key.
        target = ALIASES.get(key) or ALIASES.get(re.sub(r"\d+$", "", key))
        # First alias wins: MT5 exports repeat "Price" for entry and exit.
        if target and target not in used:
            mapping[col] = target
            used.add(target)
        elif target == "entry_price" and "exit_price" not in used:
            # Second "Price" column in an MT5 statement is the close price.
            mapping[col] = "exit_price"
            used.add("exit_price")
    return df.rename(columns=mapping), mapping


def _normalise_direction(value) -> str:
    s = str(value).strip().lower()
    if s in {"1", "1.0", "sell", "s", "short", "sell limit", "sell stop"} or "sell" in s or "short" in s:
        return "Sell"
    if s in {"0", "0.0", "buy", "b", "long"} or "buy" in s or "long" in s:
        return "Buy"
    return "Buy"


def pip_size(symbol: str) -> float:
    """Resolve pip/point size for an instrument."""
    s = str(symbol).upper()
    if any(h in s for h in _INDEX_HINTS):
        return PIP_SIZE["INDEX"]
    for k in ("JPY", "XAU", "XAG", "BTC", "ETH"):
        if k in s:
            return PIP_SIZE[k]
    return PIP_SIZE["DEFAULT"]


def normalise(df_raw: pd.DataFrame) -> pd.DataFrame:
    """Coerce any mapped frame into the canonical schema with correct dtypes."""
    df, _ = map_columns(df_raw.copy())

    for col in CANONICAL_COLUMNS:
        if col not in df.columns:
            df[col] = np.nan
    df = df[CANONICAL_COLUMNS].copy()

    for col in TIME_COLS:
        df[col] = pd.to_datetime(df[col], errors="coerce", format="mixed")
    for col in NUM_COLS:
        df[col] = pd.to_numeric(
            df[col].astype(str).str.replace(r"[^\d\.\-\+eE]", "", regex=True),
            errors="coerce",
        )
    for col in TEXT_COLS:
        if col in ("ticket",):
            df[col] = df[col].astype(str).str.strip().replace({"nan": ""})
        else:
            df[col] = df[col].fillna("").astype(str).str.strip()

    df["direction"] = df["direction"].map(_normalise_direction)
    df["symbol"] = df["symbol"].str.upper().replace({"": "UNKNOWN"})
    df["commission"] = df["commission"].fillna(0.0)
    df["swap"] = df["swap"].fillna(0.0)
    df["execution_rating"] = df["execution_rating"].clip(1, 5)

    # A trade with no close time is still open -> park it at the open time so the
    # equity curve stays monotonic in time rather than dropping the row.
    df["close_time"] = df["close_time"].fillna(df["open_time"])
    df = df.dropna(subset=["open_time"]).reset_index(drop=True)

    # Ticket fallback so every row is addressable.
    blank = df["ticket"].eq("") | df["ticket"].isna()
    df.loc[blank, "ticket"] = [f"T{i:05d}" for i in range(1, int(blank.sum()) + 1)]
    return df


# --------------------------------------------------------------------------- #
# Derived analytics fields
# --------------------------------------------------------------------------- #
def enrich(df: pd.DataFrame) -> pd.DataFrame:
    """Add every derived column the analytics layer expects.

    The interesting piece is the risk model. Rather than maintaining contract
    specs per symbol, we back out the *implied money value of one price unit*
    from each closed trade:

        money_per_unit = |gross_pnl| / |exit_price - entry_price|

    Any trade missing that information inherits the median value of its own
    symbol (then the global median). Risk in account currency is then:

        risk = |entry_price - stop_loss| * money_per_unit
        R    = net_pnl / risk

    This makes R-multiples comparable across XAUUSD, EURUSD, US500 and TASI
    without a single hard-coded tick value.
    """
    if df.empty:
        out = df.copy()
        for c in ["net_pnl", "pips", "r_multiple", "planned_rr", "risk_amount",
                  "money_per_unit", "duration_min"]:
            out[c] = pd.Series(dtype="float64")
        for c in ["outcome", "date", "dow", "hour", "month", "week", "session"]:
            out[c] = pd.Series(dtype="object")
        out["mistake_list"] = pd.Series(dtype="object")
        return out

    d = df.copy()

    # --- Net P&L ---------------------------------------------------------- #
    computed_net = d["gross_pnl"].fillna(0.0) - d["commission"].abs().fillna(0.0) \
        + d["swap"].fillna(0.0)
    d["net_pnl"] = d["net_pnl"].where(d["net_pnl"].notna(), computed_net)
    # If only net was supplied, reconstruct gross so profit-factor stays honest.
    d["gross_pnl"] = d["gross_pnl"].where(
        d["gross_pnl"].notna(),
        d["net_pnl"] + d["commission"].abs().fillna(0.0) - d["swap"].fillna(0.0),
    )
    d["fees"] = d["commission"].abs().fillna(0.0) - d["swap"].fillna(0.0)

    # --- Direction sign: +1 long, -1 short -------------------------------- #
    sign = np.where(d["direction"].eq("Sell"), -1.0, 1.0)
    price_move = (d["exit_price"] - d["entry_price"]) * sign

    # --- Pips ------------------------------------------------------------- #
    psize = d["symbol"].map(pip_size).astype(float)
    computed_pips = price_move / psize
    d["pips"] = d["pips"].where(d["pips"].notna(), computed_pips)

    # If the export gave pips but no exit price, reconstruct the exit so the
    # risk model below still has something to work with.
    missing_exit = d["exit_price"].isna() & d["pips"].notna() & d["entry_price"].notna()
    d.loc[missing_exit, "exit_price"] = (
        d.loc[missing_exit, "entry_price"]
        + sign[missing_exit.to_numpy()] * d.loc[missing_exit, "pips"] * psize[missing_exit]
    )

    # --- Implied money per price unit ------------------------------------- #
    denom = (d["exit_price"] - d["entry_price"]).abs()
    mpu = (d["gross_pnl"].abs() / denom).replace([np.inf, -np.inf], np.nan)
    mpu = mpu.where(denom > 0)
    # Fill by symbol median, then the global median. Deliberately NOT filled with
    # a placeholder: an invented tick value would produce plausible-looking but
    # meaningless R-multiples, which is worse than showing nothing.
    by_symbol = mpu.groupby(d["symbol"]).transform("median")
    mpu = mpu.fillna(by_symbol).fillna(mpu.median())
    d["money_per_unit"] = mpu

    # --- Risk, R-multiple, planned RR ------------------------------------- #
    stop_dist = (d["entry_price"] - d["stop_loss"]).abs()
    d["risk_amount"] = (stop_dist * d["money_per_unit"]).where(stop_dist > 0)
    d["r_multiple"] = (d["net_pnl"] / d["risk_amount"]).replace([np.inf, -np.inf], np.nan)

    target_dist = (d["take_profit"] - d["entry_price"]).abs()
    d["planned_rr"] = (target_dist / stop_dist).where(stop_dist > 0) \
        .replace([np.inf, -np.inf], np.nan)

    # --- Outcome classification ------------------------------------------- #
    d["outcome"] = np.select(
        [d["net_pnl"] > 0, d["net_pnl"] < 0], ["Win", "Loss"], default="Breakeven"
    )

    # --- Calendar buckets (based on entry time = the decision moment) ------ #
    d["duration_min"] = (d["close_time"] - d["open_time"]).dt.total_seconds() / 60.0
    d["date"] = d["close_time"].dt.date
    d["dow"] = d["open_time"].dt.day_name()
    d["hour"] = d["open_time"].dt.hour
    d["month"] = d["open_time"].dt.to_period("M").astype(str)
    d["week"] = d["open_time"].dt.to_period("W").astype(str)
    d["session"] = pd.cut(
        d["hour"], bins=[-1, 6, 11, 16, 23],
        labels=["Asia", "London AM", "New York", "Late NY / Asia open"],
    ).astype(str)

    # --- Mistake tags ------------------------------------------------------ #
    d["mistake_list"] = d["mistakes"].map(split_tags)

    d = d.sort_values("close_time").reset_index(drop=True)
    return d


def split_tags(value) -> list[str]:
    """Split a free-text tag cell into a clean list."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return []
    parts = re.split(r"[;,|]", str(value))
    return [p.strip() for p in parts if p.strip()]


def load_csv(path_or_buffer) -> pd.DataFrame:
    """Read a CSV/TSV export and return an enriched canonical frame."""
    df = pd.read_csv(path_or_buffer, sep=None, engine="python")
    return enrich(normalise(promote_header(df)))


# --------------------------------------------------------------------------- #
# Tolerant file reading
# --------------------------------------------------------------------------- #
# Real broker exports are messier than a clean CSV: Windows-Arabic encodings,
# semicolon delimiters from European locales, a title/account block above the
# real header row, and MT4/MT5's default HTML or XLSX report formats. Each of
# those looks to the user like "the upload does not work", so all of them are
# handled here rather than rejected.

ENCODINGS = ("utf-8-sig", "utf-8", "utf-16", "cp1256", "cp1252", "latin-1")


def header_score(values) -> int:
    """How many cells in a row look like known column headers."""
    return sum(1 for v in values if _key(v) in ALIASES
               or ALIASES.get(re.sub(r"\d+$", "", _key(v))))


def promote_header(df: pd.DataFrame, max_scan: int = 12) -> pd.DataFrame:
    """Find the real header row when the export starts with a title block.

    MT4/MT5 statements often open with account name, broker and date lines. If
    the current columns don't look like headers but a row further down does,
    that row is promoted and everything above it is dropped.
    """
    if df.empty:
        return df
    if header_score(df.columns) >= 3:
        return df

    best_row, best = None, 2  # require a clearly better candidate
    for i in range(min(max_scan, len(df))):
        score = header_score(df.iloc[i].tolist())
        if score > best:
            best_row, best = i, score

    if best_row is None:
        return df

    out = df.iloc[best_row + 1:].copy()
    out.columns = [str(c).strip() for c in df.iloc[best_row].tolist()]
    return out.dropna(axis=1, how="all").reset_index(drop=True)


def sniff_header(text: str, max_scan: int = 30) -> tuple[str, int, int]:
    """Locate the delimiter and the header line in raw text.

    Delimiter auto-detection has to happen *after* the header is found, not
    before: a statement whose first line reads "Trade History Report" will
    otherwise be sniffed as whitespace-delimited and parsed into nonsense. So
    every (delimiter, line) pair is scored by how many known column names the
    split produces, and the best pair wins.

    Returns (delimiter, line_index_of_header, score).
    """
    lines = text.splitlines()[:max_scan]
    best = (",", 0, 0)
    for sep in (",", ";", "\t", "|"):
        for i, line in enumerate(lines):
            cells = line.split(sep)
            if len(cells) < 3:
                continue
            score = header_score(cells)
            if score > best[2]:
                best = (sep, i, score)
    return best


def _read_csv_bytes(data: bytes) -> pd.DataFrame:
    """Try every plausible encoding, then locate the header, then parse."""
    errors: list[str] = []
    for enc in ENCODINGS:
        try:
            text = data.decode(enc)
        except (UnicodeDecodeError, LookupError):
            continue
        if not text.strip():
            continue

        sep, header_line, score = sniff_header(text)
        attempts = ([(sep, header_line)] if score >= 3 else []) + \
                   [(None, 0), (",", 0), (";", 0), ("\t", 0), ("|", 0)]

        for use_sep, skip in attempts:
            try:
                df = pd.read_csv(io.StringIO(text), sep=use_sep, engine="python",
                                 skiprows=skip, skip_blank_lines=True)
            except Exception as exc:  # noqa: BLE001 - try the next combination
                errors.append(f"{enc}/{use_sep or 'auto'}: {exc}")
                continue
            if df.shape[1] >= 3 and len(df) > 0:
                return df

    raise ValueError(
        "Could not parse the file as a table with any recognisable columns. "
        f"Tried encodings: {', '.join(ENCODINGS)}. "
        f"Last errors: {'; '.join(errors[-2:]) or 'none'}"
    )


def _read_html_bytes(data: bytes) -> pd.DataFrame:
    """Pick the table in an MT4/MT5 HTML statement that looks most like trades."""
    last: Exception | None = None
    for flavor in ("lxml", "bs4"):
        for enc in ENCODINGS:
            try:
                tables = pd.read_html(io.BytesIO(data), flavor=flavor, encoding=enc)
            except Exception as exc:  # noqa: BLE001 - flavour may not be installed
                last = exc
                continue
            if tables:
                # Score every table by how many real headers it can expose.
                scored = [(header_score(promote_header(t).columns), t) for t in tables]
                score, best = max(scored, key=lambda pair: pair[0])
                if score >= 3:
                    return best
                return max(tables, key=lambda t: t.shape[0] * t.shape[1])
    raise ValueError(
        "Could not read the HTML statement. Install `lxml` or `beautifulsoup4` "
        f"+ `html5lib`, or re-export as CSV. ({last})"
    )


def read_table(filename: str, data: bytes) -> pd.DataFrame:
    """Read an uploaded broker export of any common format into a raw frame.

    Accepts CSV/TSV/TXT, Excel and HTML statements. Returns the frame *before*
    normalisation so the caller can show the user which columns were detected.
    """
    ext = str(filename).lower().rsplit(".", 1)[-1] if "." in str(filename) else ""

    if ext in {"xlsx", "xlsm", "xls"}:
        try:
            raw = pd.read_excel(io.BytesIO(data))
        except ImportError as exc:
            raise ValueError(
                "Reading Excel needs `openpyxl` — add it to requirements.txt, "
                "or re-export the statement as CSV."
            ) from exc
    elif ext in {"htm", "html"}:
        raw = _read_html_bytes(data)
    else:
        raw = _read_csv_bytes(data)

    # Flatten the MultiIndex columns pandas builds from merged HTML/Excel headers.
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = [" ".join(str(p) for p in col if str(p) != "nan").strip()
                       for col in raw.columns]

    raw = promote_header(raw)
    raw = raw.dropna(axis=1, how="all").dropna(axis=0, how="all")
    return raw.reset_index(drop=True)


def to_export_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Strip derived columns so exports round-trip cleanly back into the app."""
    cols = [c for c in CANONICAL_COLUMNS if c in df.columns]
    return df[cols].copy()
