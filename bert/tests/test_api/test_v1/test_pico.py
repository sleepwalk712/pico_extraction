import os
import shutil

from fastapi.testclient import TestClient

from main import app


client = TestClient(app)

MODEL_PATH = "test_ner_service"


def test_predict_endpoint():
    request_data = {
        "text": "Test text for prediction",
        "ml_model_path": MODEL_PATH,
    }
    response = client.post("/v1/pico/predict", json=request_data)

    assert response.status_code == 200
    assert "predictions" in response.json()
    assert isinstance(response.json()["predictions"], list)


def test_fine_tune_endpoint():
    request_data = {
        "texts": [["Text", "1"], ["Text", "2"]],
        "labels": [[1, 2], [0, 0]],
        "epochs": 3,
        "ml_model_path": MODEL_PATH
    }
    response = client.post("/v1/pico/fine_tune", json=request_data)

    assert response.status_code == 200
    assert response.json() == {"message": "Fine-tuning completed successfully"}


def teardown_module(module):
    if os.path.exists(MODEL_PATH):
        shutil.rmtree(MODEL_PATH)
