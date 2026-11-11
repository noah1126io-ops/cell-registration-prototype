"""Render research_ui.stage_panel for the core-1.2 artifacts (Streamlit AppTest) and print what the user would see."""
import os
from streamlit.testing.v1 import AppTest

os.environ["CELLREG_RESEARCH_MODE"] = "1"


def app():
    import json
    import sys
    import streamlit as st
    repo = "/mnt/mib-dbia08_1/users/noa.ganeko/projects/cell-registration-prototype-gpu"
    sys.path.insert(0, repo)
    from src.registration_explorer import research_ui, results
    for s in ("031-8_6", "031-6_2"):
        art = json.load(open(f"{repo}/analysis/landmark-reference-failure-analysis-20261110/cache/runs/auto_{s}/artifact.json"))
        rec = results.record_from_engine_artifact(art, f"wf:auto_{s}")
        st.markdown(f"### {s} SAFETY={rec.safety_label()}")
        research_ui.stage_panel(rec)


if __name__ == "__main__":
    at = AppTest.from_function(app, default_timeout=120).run()
    print("exceptions:", [e.value for e in at.exception])
    for el in list(at.markdown) + list(at.info) + list(at.warning) + list(at.success) + list(at.caption):
        print(type(el).__name__, "|", str(el.value)[:220])
    for t in at.table:
        print(t.value.to_string()[:1200])
