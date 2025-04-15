import pytest
from pydantic import ValidationError

from app.models.schemas import ClassificationFineTuneRequest


def test_classification_fine_tune_request_valid():
    data = {
        "ml_model_path": "/fake/path",
        "texts": ["text1", "text2"],
        "labels": [1, 0],
        "epochs": 2,
        "num_labels": 3
    }
    request = ClassificationFineTuneRequest(**data)
    assert request.ml_model_path == "/fake/path"
    assert len(request.texts) == 2
    assert request.epochs == 2


def test_classification_fine_tune_request_missing_ml_model_path():
    data = {
        "texts": ["text1", "text2"],
        "labels": [1, 0],
        "epochs": 2,
        "num_labels": 3
    }
    with pytest.raises(ValidationError):
        ClassificationFineTuneRequest(**data)
