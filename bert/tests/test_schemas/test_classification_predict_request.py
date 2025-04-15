import pytest
from pydantic import ValidationError

from app.models.schemas import ClassificationPredictRequest


def test_classification_predict_request_valid():
    data = {
        "ml_model_path": "/fake/path",
        "text": "This is a test prediction",
        "num_labels": 3
    }
    request = ClassificationPredictRequest(**data)
    assert request.ml_model_path == "/fake/path"
    assert request.text == "This is a test prediction"
    assert request.num_labels == 3


def test_classification_predict_request_missing_ml_model_path():
    data = {
        "text": "This is a test prediction",
        "num_labels": 3
    }
    with pytest.raises(ValidationError):
        ClassificationPredictRequest(**data)
