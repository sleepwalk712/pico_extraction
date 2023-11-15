import os
import shutil

from app.services.ner_service import NERService


def test_ner_service_predict():
    ner_service = NERService()
    text = "Sunitinib is a tyrosine kinase inhibitor"
    result = ner_service.predict(text)

    assert isinstance(result, list), "Result should be a list"
    assert all(isinstance(label, int)
               for label in result), "All items in result should be strings"


def test_ner_service_fine_tune():
    ner_service = NERService()

    texts = ["Sunitinib is a tyrosine kinase inhibitor",
             "This is another example"]
    texts = [text.split() for text in texts]
    labels = [[1, 2, 2, 0, 0, 0], [0, 0, 0, 0]]

    ner_service.fine_tune(
        texts,
        labels,
        epochs=1,
        model_path='test_ner_service',
    )

    assert os.path.exists('test_ner_service'), "Model file was not saved"

    if os.path.exists('test_ner_service'):
        shutil.rmtree('test_ner_service')
