import pytest
from fastapi.testclient import TestClient

from main import app
from app.api.v1 import classification


class FakeClassificationService:
    def __init__(self, model_path: str, num_labels: int = 2):
        self.model_path = model_path
        self.num_labels = num_labels

    def predict(self, text: str) -> int:
        return 42

    def fine_tune(self, texts, labels, epochs, validation_split=0.2):
        pass


@pytest.fixture(autouse=True)
def override_classification_service(monkeypatch):
    monkeypatch.setattr(
        classification,
        "ClassificationService",
        FakeClassificationService
    )


client = TestClient(app)


def test_classification_predict_success():
    payload = {
        "ml_model_path": "/fake/path",
        "text": "This is a prediction test",
        "num_labels": 3
    }
    response = client.post("/v1/classification/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["prediction"] == 42


def test_classification_predict_missing_ml_model_path():
    payload = {
        "text": "This is a prediction test",
        "num_labels": 3
    }
    response = client.post("/v1/classification/predict", json=payload)
    assert response.status_code == 422


def test_classification_fine_tune_success():
    payload = {
        "ml_model_path": "/fake/path",
        "texts": ["text1", "text2"],
        "labels": [1, 0],
        "epochs": 1,
        "num_labels": 2
    }
    response = client.post("/v1/classification/fine_tune", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["message"] == "Fine-tuning completed successfully"


def test_classification_fine_tune_missing_ml_model_path():
    payload = {
        "texts": ["text1", "text2"],
        "labels": [1, 0],
        "epochs": 1,
        "num_labels": 2
    }
    response = client.post("/v1/classification/fine_tune", json=payload)
    assert response.status_code == 422
