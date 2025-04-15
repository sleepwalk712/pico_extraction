from app.models.schemas import NerPredictRequest


def test_ner_predict_request_schema():
    test_data = {
        "text": "Test text for prediction",
        "ml_model_path": "ner_model",
    }

    request = NerPredictRequest(**test_data)

    assert request.text == "Test text for prediction"
    assert request.ml_model_path == "ner_model"
