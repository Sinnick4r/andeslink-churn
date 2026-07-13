# tests del registro JSON

import json

from app.inference_recorder import record_inference


def test_record_inference_writes_jsonl(tmp_path) -> None:
    output = tmp_path / "inferences.jsonl"

    saved = record_inference(
        path=output,
        features={"tenure_months": 12, "contract_type": "mensual"},
        churn=1,
        probability=0.7234567,
        threshold=0.441444,
        model_version="v2",
        request_id="test-request-001",
    )

    assert saved is True
    record = json.loads(output.read_text(encoding="utf-8").strip())
    assert record["request_id"] == "test-request-001"
    assert record["probability"] == 0.723457
    assert record["tenure_months"] == 12
    assert record["contract_type"] == "mensual"


def test_record_inference_appends_records(tmp_path) -> None:
    output = tmp_path / "inferences.jsonl"

    for index in range(2):
        assert record_inference(
            path=output,
            features={"tenure_months": 12 + index},
            churn=index,
            probability=0.25 + index * 0.5,
            threshold=0.441444,
            model_version="v2",
            request_id=f"test-request-{index}",
        )

    lines = output.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    assert json.loads(lines[1])["request_id"] == "test-request-1"


def test_record_inference_failure_is_non_blocking(tmp_path) -> None:
    invalid_path = tmp_path / "directory"
    invalid_path.mkdir()

    saved = record_inference(
        path=invalid_path,
        features={"tenure_months": 12},
        churn=0,
        probability=0.25,
        threshold=0.441444,
        model_version="v2",
        request_id="test-request-error",
    )

    assert saved is False
