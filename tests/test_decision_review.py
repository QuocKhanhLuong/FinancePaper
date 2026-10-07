"""Independent math and expert-preview separation, not human validation."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from financepaper.evaluation.decision_review import (
    audit_fixture, build_preview, enumerated_answer, exact_shapley, read_public, validate_public,
)
from financepaper.evaluation.decision_vignettes import CurrentCase, export_synthetic_bundle, render_case


def example():
    return render_case(CurrentCase("public_id", .5, 1., .25, None, .7, (.5, 0., .25), (8, 4, 8)), 4)


def test_independent_enumeration_and_entailment_boundaries():
    np.testing.assert_allclose(exact_shapley(1, 2, 1, .4), [.5, 1, .5, .4], atol=1e-12)
    np.testing.assert_allclose(exact_shapley(0, 0, 0, 0), [0, 0, 0, 0])
    assert enumerated_answer(.999, None) == "review"
    assert enumerated_answer(1, None) == "supported"
    assert enumerated_answer(.25, 0) == "supported"
    assert enumerated_answer(.25, 1) == "review"


@pytest.mark.parametrize("mutation", [
    lambda row: row.update(answer="supported"),
    lambda row: row["available_record"].update(full_H=1),
    lambda row: row.update(risk_probability=float("nan")),
    lambda row: row.update(explanation=[{"answer": "review"}]),
    lambda row: row["available_record"].update(requirement_H=3),
    lambda row: row.pop("completion_caveat"),
])
def test_strict_schema_rejects_private_and_invalid_inputs(mutation):
    row = example()
    mutation(row)
    with pytest.raises(ValueError):
        validate_public(row)


def test_preview_escapes_payload_and_never_reads_private_files(tmp_path, monkeypatch):
    row = example()
    row["task"] = '</script><script>window.POISON=true</script>'
    source = tmp_path / "slot_000.jsonl"
    source.write_text(json.dumps(row) + "\n")
    original = Path.read_text
    allowed = {source.resolve(), Path("templates/decision_review.html").resolve()}
    def guard(path, *args, **kwargs):
        assert path.resolve() in allowed, f"Unexpected file read: {path}"
        return original(path, *args, **kwargs)
    monkeypatch.setattr(Path, "read_text", guard)
    result = build_preview(source, Path("templates/decision_review.html"))
    assert '</script><script>window.POISON' not in result
    assert "\\u003c/script\\u003e" in result
    assert 'connect-src \'none\'' in result
    assert "localStorage" not in result and "fetch(" not in result and "<form" not in result
    assert 'type="application/json"' in result


def test_duplicate_case_is_refused(tmp_path):
    source = tmp_path / "slot.jsonl"
    source.write_text((json.dumps(example()) + "\n") * 2)
    with pytest.raises(ValueError, match="repeated case"):
        read_public(source)


def test_whole_fixture_audit_detects_tampered_labels_and_renderings(tmp_path):
    bundle = tmp_path / "bundle"
    export_synthetic_bundle(bundle)
    output = tmp_path / "audit"; output.mkdir()
    audit_fixture(bundle, 20261007, output)
    audit = json.loads((output / "audit.json").read_text())
    assert audit["pool_cases_checked"] == 245
    assert audit["public_renderings_checked"] == 256
    rows = pd.read_csv(output / "rule_baselines.csv")
    assert (rows[rows.baseline == "current_record_rule"].accuracy == 1).all()
    assert audit["human_study"] == "NOT_RUN"
    public = bundle / "expert/slot_000.jsonl"
    original = public.read_text()
    first, *rest = original.splitlines()
    record = json.loads(first); record["risk_probability"] = .123
    public.write_text(json.dumps(record) + "\n" + "\n".join(rest) + "\n")
    with pytest.raises(ValueError, match="differs"):
        audit_fixture(bundle, 20261007, output)
    public.write_text(original)
    private = bundle / "private/evaluation.json"
    original_private = private.read_text()
    records = json.loads(original_private)
    records[0]["answer"] = "POISON"
    private.write_text(json.dumps(records))
    with pytest.raises(ValueError, match="private labels"):
        audit_fixture(bundle, 20261007, output)
    private.write_text(original_private)
    # A repeated case in the allocation must be rejected before public export.
    allocation = bundle / "private/allocation.csv"
    frame = pd.read_csv(allocation)
    pd.concat([frame, frame.iloc[:1]]).to_csv(allocation, index=False)
    with pytest.raises(ValueError, match="repeated case"):
        audit_fixture(bundle, 20261007, output)
