"""End-to-end app tests using Streamlit's own headless runner.

Why this file exists: importing `app.py` outside Streamlit ("bare mode") checks
syntax and logic but skips the script-run context entirely -- which is exactly
where Streamlit assigns element ids. A `StreamlitDuplicateElementId` therefore
cannot be reproduced that way. `AppTest` runs the script with a real context, so
it catches it.

Run:
    python test_app.py
    pytest -q test_app.py
"""
from __future__ import annotations

import pandas as pd
from streamlit.testing.v1 import AppTest

from schema import enrich, normalise
import storage

PAGES = [
    "Dashboard",
    "Deep analytics",
    "Psychology & AI review",
    "Trade log",
    "Add trade",
    "Data & settings",
]


def run_page(page: str, trades: pd.DataFrame | None = None) -> AppTest:
    """Run app.py to completion on one page and return the finished app."""
    at = AppTest.from_file("app.py", default_timeout=120)
    if trades is not None:
        # init_state() only populates when the key is absent, so seeding here
        # lets us drive the app with any journal we like -- including none.
        at.session_state["trades"] = trades
    at.run()
    assert not at.exception, f"[{page}] initial run: {at.exception}"

    if page != PAGES[0]:
        at.sidebar.radio[0].set_value(page).run()
        assert not at.exception, f"[{page}] after navigation: {at.exception}"
    return at


def test_all_pages_with_sample_data():
    """The default experience: bundled demo journal, every page."""
    df = storage.load_sample()
    assert not df.empty, "sample_trades.csv is missing or unreadable"
    for page in PAGES:
        run_page(page, df)


def test_all_pages_with_empty_journal():
    """The failure mode that broke production.

    With no trades, several charts fall back to the same "no data" placeholder
    figure. Identical arguments produce identical element ids, so without an
    explicit key Streamlit raises StreamlitDuplicateElementId. This test only
    passes while every chart carries a unique key.
    """
    empty = enrich(normalise(pd.DataFrame()))
    for page in PAGES:
        run_page(page, empty)


def test_all_pages_with_single_trade():
    """Degenerate sample: no losses, no streaks, undefined profit factor."""
    one = storage.load_sample().head(1)
    for page in PAGES:
        run_page(page, one)


def test_chart_keys_are_unique():
    """Static guard: a copy-pasted call site should fail here, not in production."""
    import re

    source = open("app.py", encoding="utf-8").read()
    keys = re.findall(r'chart\([^\n]*?key="([^"]+)"', source, flags=re.S)
    assert keys, "no keyed chart calls found - did chart() change?"
    duplicates = {k for k in keys if keys.count(k) > 1}
    assert not duplicates, f"duplicate chart keys: {sorted(duplicates)}"


def test_offline_review_renders():
    """The AI tab must produce a report with no provider configured."""
    import ai_review
    import metrics as M

    df = storage.load_sample()
    report = ai_review.heuristic_report(df, M.compute_kpis(df, 10_000))
    assert "الخلاصة" in report and len(report) > 400


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"  PASS  {t.__name__}")
        except AssertionError as exc:
            failed += 1
            print(f"  FAIL  {t.__name__}\n        {exc}")
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print(f"  ERROR {t.__name__}  {type(exc).__name__}: {exc}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    raise SystemExit(1 if failed else 0)
