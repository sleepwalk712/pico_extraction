import os
import shutil

from app.services.ner_service import NerService


def test_ner_service_predict():
    model_path = "/tmp/test_ner_service"
    ner_service = NerService(model_path=model_path)

    text = "Sunitinib is a tyrosine kinase inhibitor"
    result = ner_service.predict(text)

    assert isinstance(result, list), "Result should be a list"
    assert all(isinstance(label, int) for label in result)


def test_ner_service_fine_tune():
    model_path = "/tmp/test_ner_service"
    ner_service = NerService(model_path=model_path)

    texts = [
        ["Sunitinib", "is", "a", "tyrosine", "kinase", "inhibitor"],
        ["This", "is", "another", "example"],
    ]
    labels = [[1, 2, 2, 0, 0, 0], [0, 0, 0, 0]]

    ner_service.fine_tune(
        texts,
        labels,
        epochs=1,
    )

    assert os.path.exists('/tmp/test_ner_service')

    try:
        if os.path.exists('/tmp/test_ner_service'):
            shutil.rmtree('/tmp/test_ner_service')
    except FileNotFoundError:
        pass
