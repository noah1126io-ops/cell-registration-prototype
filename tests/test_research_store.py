import json
import sqlite3

import numpy as np
import pytest

from src import research_store as rs


@pytest.fixture
def store(tmp_path):
    return rs.Store(tmp_path / "r.sqlite")


def test_records_survive_reopen_and_are_scoped_by_sample(store, tmp_path):
    sid = store.create_session("s1", "A")
    store.add_review(sample_id="A", session_id=sid, run_id="run1", candidate_id="points#0", judgement="incorrect",
                     problems=["region", "rotation"], comment="違う領域", view={"M": [[1, 0], [0, 1]], "t": [0, 0]})
    store.add_review(sample_id="B", judgement="correct")
    again = rs.Store(tmp_path / "r.sqlite")          # app restart
    a = again.reviews("A")
    assert len(a) == 1 and json.loads(a[0]["problems_json"]) == ["region", "rotation"] and a[0]["session_id"] == sid
    assert len(again.reviews("B")) == 1 and again.sessions("A")[0]["name"] == "s1"
    assert again.integrity_check() == "ok" and again.schema_version() == rs.SCHEMA_VERSION


def test_append_only_tables_cannot_be_modified(store):
    tid = store.add_transform(sample_id="A", kind="manual", fixed="HE", moving="IF", M=np.eye(2), t=[0, 0], status="MANUAL_UNVERIFIED")
    con = sqlite3.connect(store.path)
    with pytest.raises(sqlite3.DatabaseError):
        con.execute("UPDATE transforms SET status='ACCEPTED' WHERE id=?", (tid,))
    with pytest.raises(sqlite3.DatabaseError):
        con.execute("DELETE FROM transforms WHERE id=?", (tid,))
    con.close()
    with pytest.raises(ValueError):                  # a manual transform can never be stored as anything but unverified
        store.add_transform(sample_id="A", kind="manual", fixed="HE", moving="IF", M=np.eye(2), t=[0, 0], status="ACCEPTED")
    with pytest.raises(ValueError):
        store.add_review(sample_id="A", judgement="maybe")
    with pytest.raises(ValueError):
        store.add_review(sample_id="A", judgement="correct", problems=["colour"])


def test_run_artifact_cannot_be_replaced(store, tmp_path):
    art = {"decision": {"status": "NOT_ACCEPTED", "accepted": False, "selected_candidate": "image#0", "failure_stage": "C1"},
           "input": {"fixed": {"points_sha256": "x"}}, "software": {"core": "t"}}
    p1 = tmp_path / "a1.json"; p1.write_text(json.dumps(art))
    p2 = tmp_path / "a2.json"; p2.write_text(json.dumps(art))
    store.register_run(run_id="r1", session_id=None, sample_id="A", mode="automatic", localization="auto", fixed="HE", moving="IF",
                       run_dir=tmp_path, params={})
    store.update_run("r1", state="completed", artifact_path=p1)
    row = store.run("r1")
    assert row["status"] == "NOT_ACCEPTED" and row["accepted"] == 0 and row["failure_stage"] == "C1" and len(row["artifact_sha256"]) == 64
    with pytest.raises(ValueError):
        store.update_run("r1", artifact_path=p2)


def test_superseded_annotations_hidden_and_export(store):
    a1 = store.add_annotation(sample_id="A", kind="search_roi", frame="A:HE", geometry=[[0, 0], [10, 0], [10, 10]], payload={"frame_key": "HE"}, label="v1")
    store.add_annotation(sample_id="A", kind="search_roi", frame="A:HE", geometry=[[0, 0], [20, 0], [20, 20]], payload={"frame_key": "HE"},
                         label="v2", supersedes=a1)
    rows = store.annotations("A")
    assert [r["label"] for r in rows] == ["v2"]
    data = store.export_json("A")
    assert data["schema"].startswith("cellreg-research-store-export") and len(data["annotations"]) == 1 and data["events"]
    csvs = store.export_csv("A")
    assert set(csvs) == {"runs.csv", "transforms.csv", "reviews.csv", "annotations.csv", "events.csv"} and "v2" in csvs["annotations.csv"]


def test_constraints_from_annotations_explicit_and_direction_aware(store):
    M = 1.02 * np.array([[np.cos(0.1), -np.sin(0.1)], [np.sin(0.1), np.cos(0.1)]]); t = np.array([30.0, 40.0])
    ex = store.add_annotation(sample_id="A", kind="exclusion", frame="A:HE", payload={"M": M.tolist(), "t": t.tolist(), "fixed": "HE", "moving": "IF"})
    roi = store.add_annotation(sample_id="A", kind="search_roi", frame="A:HE", geometry=[[0, 0], [5, 0], [5, 5]], payload={"frame_key": "HE"})
    lm = store.add_annotation(sample_id="A", kind="landmarks", frame="A:HE", geometry=[{"fixed": [1, 2], "moving": [3, 4]}, {"fixed": [5, 6], "moving": [7, 8]}],
                              payload={"fixed": "HE", "moving": "IF"})
    c = rs.constraints_from_annotations(store, [ex, roi, lm], "HE", "IF")
    assert c.used() and c.search_roi and len(c.exclude_poses) == 1 and c.landmarks[0]["fixed"] == [1, 2] and c.annotation_ids == [ex, roi, lm]
    # other direction: exclusion pose inverted, landmarks swapped, a fixed-frame ROI is refused
    c2 = rs.constraints_from_annotations(store, [ex, lm], "IF", "HE")
    Mi = np.asarray(c2.exclude_poses[0]["M"]); ti = np.asarray(c2.exclude_poses[0]["t"])
    p = np.array([[10.0, 20.0]])
    assert np.allclose((p @ M.T + t) @ Mi.T + ti, p) and c2.landmarks[0] == {"fixed": [3, 4], "moving": [1, 2]}
    with pytest.raises(ValueError):
        rs.constraints_from_annotations(store, [roi], "IF", "HE")
    assert not rs.constraints_from_annotations(store, [], "HE", "IF").used()     # nothing is applied unless selected
