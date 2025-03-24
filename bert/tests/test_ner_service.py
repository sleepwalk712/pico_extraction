import os
import shutil

from app.services.ner_service import NERService


def test_ner_service_predict():
    ner_service = NERService()
    text = "Sunitinib is a tyrosine kinase inhibitor"
    result = ner_service.predict(text)

    assert isinstance(result, list), "Result should be a list"
    assert all(isinstance(label, int)
               for label in result), "All items in result should be integers"


def test_ner_service_fine_tune():
    ner_service = NERService()

    texts = [
        ["Sunitinib", "is", "a", "tyrosine", "kinase", "inhibitor"],
        ["This", "is", "another", "example"],
    ]
    labels = [[1, 2, 2, 0, 0, 0], [0, 0, 0, 0]]

    ner_service.fine_tune(
        texts,
        labels,
        epochs=1,
        ml_model_path='/tmp/test_ner_service',
    )

    assert os.path.exists(
        '/tmp/test_ner_service'), "Model directory was not saved"

    try:
        if os.path.exists('/tmp/test_ner_service'):
            shutil.rmtree('/tmp/test_ner_service')
    except FileNotFoundError:
        pass
