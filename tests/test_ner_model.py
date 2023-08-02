import torch

from app.core.ner_model import NERModel


def test_ner_model():
    ner_model = NERModel()
    text = "Sunitinib is a tyrosine kinase inhibitor"
    inputs = ner_model.encode(text)

    assert isinstance(inputs, dict)
    assert isinstance(inputs['input_ids'], torch.Tensor)

    outputs = ner_model.predict(inputs)

    assert isinstance(outputs, torch.Tensor)
