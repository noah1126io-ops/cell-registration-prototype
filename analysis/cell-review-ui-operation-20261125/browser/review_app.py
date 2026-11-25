"""Minimal Streamlit page with only the Cell Review tab (real-browser checks of 2026-11-25).

env: CRO_ROOT (review root with a precomputed auto result), CRO_SAMPLE (default 031-6_2), CRO_TABS=1 (put the tab second
behind a dummy tab, to check that hidden shortcuts do nothing), CRO_CELLS=1 (show the IF whole-cell outlines).
"""
import os
import sys

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
sys.path.insert(0, REPO)
os.environ["CELLREG_CELL_REVIEW_ROOT"] = os.environ["CRO_ROOT"]
import streamlit as st  # noqa: E402

st.set_page_config(page_title="cell review e2e", layout="wide")
from src.registration_explorer import catalog, results  # noqa: E402
from src.registration_explorer import cell_review_ui as U  # noqa: E402

sample = os.environ.get("CRO_SAMPLE", "031-6_2")
S = catalog.load_manifest()[sample]
avail = catalog.availability(S)
recs = [r for r in results.load_records(S) if r.M is not None]
rec = next((r for r in recs if r.method == "WF"), recs[0])
if os.environ.get("CRO_CELLS") == "1":
    st.session_state.setdefault("regx-crv-cellm-IF", "S1_cpsam_if_mem_full")
if os.environ.get("CRO_TABS") == "1":
    t0, t1 = st.tabs(["別のタブ", U.TAB_LABEL])
    with t0:
        st.write("dummy tab")
        st.text_input("dummy input", key="dummy-input")
    with t1:
        U.render(S, avail, rec, "IF")
else:
    U.render(S, avail, rec, "IF")
