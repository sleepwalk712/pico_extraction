import os
import shutil

from app.services.cls_service import ClassificationService


def test_cls_service_predict():
    model_path = "/tmp/test_cls_service"
    service = ClassificationService(model_path=model_path, num_labels=3)

    text = "This is a prediction test"
    result = service.predict(text)

    assert isinstance(result, int)


def test_cls_service_predict_batch():
    model_path = "/tmp/test_cls_service"
    service = ClassificationService(model_path=model_path, num_labels=3)

    texts = ["This is a prediction test", "Another prediction example"]
    results = service.predict_batch(texts)

    assert isinstance(results, list)
    assert all(isinstance(label, int) for label in results)
    assert len(results) == len(texts)


def test_cls_service_fine_tune():
    model_path = "/tmp/test_cls_service"
    service = ClassificationService(model_path=model_path, num_labels=3)

    texts = ["This is a training example", "Another training example"]
    labels = [1, 2]

    service.fine_tune(
        texts,
        labels,
        epochs=1,
        validation_split=0.5,
    )

    assert os.path.exists(model_path)

    try:
        if os.path.exists(model_path):
            shutil.rmtree(model_path)
    except FileNotFoundError:
        pass
