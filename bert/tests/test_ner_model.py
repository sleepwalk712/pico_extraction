import torch

from app.core.ner_model import NERModel


def test_ner_model():
    ner_model = NERModel(model_path="/tmp/test_ner_model")
    texts = ["Sunitinib", "is", "a", "tyrosine", "kinase", "inhibitor"]

    inputs = ner_model.encode_for_inference(texts)

    assert isinstance(inputs, dict)
    assert isinstance(inputs['input_ids'], torch.Tensor)
    assert isinstance(inputs['attention_mask'], torch.Tensor)

    logits = ner_model.predict(inputs)

    assert isinstance(logits, torch.Tensor)
    assert logits.size(0) == 1
    assert logits.size(-1) == ner_model.model.num_labels
