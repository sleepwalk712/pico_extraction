import torch

from app.core.ner_model import NERModel


def test_ner_model():
    ner_model = NERModel()
    texts = [["Sunitinib", "is", "a", "tyrosine", "kinase", "inhibitor"]]
    labels = [[1, 2, 2, 0, 0, 0]]

    inputs = ner_model.encode(texts, labels)

    assert isinstance(inputs, dict)
    assert isinstance(inputs['input_ids'], torch.Tensor)
    assert isinstance(inputs['labels'], torch.Tensor)

    logits = ner_model.predict(inputs)

    assert isinstance(logits, torch.Tensor)
    assert logits.size(0) == 1
    assert logits.size(-1) == ner_model.model.num_labels
