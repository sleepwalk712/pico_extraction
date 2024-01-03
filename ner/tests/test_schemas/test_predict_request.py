from app.models.schemas.predict_request import PredictRequest


def test_predict_request_schema():
    test_data = {
        "text": "Test text for prediction"
    }

    request = PredictRequest(**test_data)

    assert request.text == "Test text for prediction"
