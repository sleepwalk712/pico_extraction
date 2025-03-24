import os
import shutil

from fastapi.testclient import TestClient

from main import app


client = TestClient(app)


def test_predict_endpoint():
    request_data = {
        "text": "Test text for prediction"
    }
    response = client.post("/v1/pico/predict", json=request_data)

    assert response.status_code == 200
    assert "predictions" in response.json()


def test_fine_tune_endpoint():
    request_data = {
        "texts": [["Text", "1"], ["Text", "2"]],
        "labels": [[1, 2], [0, 0]],
        "epochs": 3,
        "ml_model_path": "test_ner_service"
    }
    response = client.post("/v1/pico/fine_tune", json=request_data)

    assert response.status_code == 200
    assert response.json() == {"message": "Fine-tuning completed successfully"}

    try:
        if os.path.exists('test_ner_service'):
            shutil.rmtree('test_ner_service')
    except FileNotFoundError:
        pass
