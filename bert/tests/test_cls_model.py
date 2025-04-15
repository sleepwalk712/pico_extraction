import torch

from app.core.cls_model import ClassificationModel


def test_cls_model_inference():
    model_path = "/tmp/test_cls_model"
    cls_model = ClassificationModel(model_path=model_path, num_labels=3)

    text = "This is a classification test"
    inputs = cls_model.encode_for_inference(text)
    assert isinstance(inputs, dict)
    assert isinstance(inputs['input_ids'], torch.Tensor)
    assert isinstance(inputs['attention_mask'], torch.Tensor)

    logits = cls_model.predict(inputs)
    assert isinstance(logits, torch.Tensor)
    assert logits.shape[0] == 1
    assert logits.shape[-1] == cls_model.model.config.num_labels


def test_cls_model_batch_inference():
    model_path = "/tmp/test_cls_model"
    cls_model = ClassificationModel(model_path=model_path, num_labels=3)

    texts = ["This is a test", "Another test text"]
    inputs = cls_model.encode_for_batch_inference(texts)
    assert isinstance(inputs, dict)
    assert isinstance(inputs['input_ids'], torch.Tensor)
    assert inputs['input_ids'].shape[0] == len(texts)

    logits = cls_model.predict(inputs)
    assert logits.shape[0] == len(texts)
