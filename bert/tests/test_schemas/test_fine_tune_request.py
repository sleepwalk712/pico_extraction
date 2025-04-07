from app.models.schemas.fine_tune_request import FineTuneRequest


def test_fine_tune_request_schema():
    test_data = {
        "texts": [["Text", "1"], ["Text", "2"]],
        "labels": [[1, 2], [0, 0]],
        "ml_model_path": "ner_model",
    }

    request = FineTuneRequest(**test_data)

    assert request.texts == [["Text", "1"], ["Text", "2"]]
    assert request.labels == [[1, 2], [0, 0]]
    assert request.epochs == 3
    assert request.ml_model_path == "ner_model"
