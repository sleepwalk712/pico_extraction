import pytest
import torch

from app.core.cls_dataset import ClassificationDataset


class MockEncoding:
    def __init__(self, max_length: int):
        self.data = {
            'input_ids': torch.tensor([[0] * max_length]),
            'attention_mask': torch.tensor([[1] * max_length])
        }

    def __getitem__(self, key):
        return self.data[key]


class MockTokenizer:
    def __call__(self, text, truncation, padding, max_length, return_tensors):
        return MockEncoding(max_length)


@pytest.fixture
def mock_tokenizer():
    return MockTokenizer()


@pytest.fixture
def cls_dataset(mock_tokenizer):
    texts = ["This is a test text", "Another text"]
    labels = [1, 0]
    return ClassificationDataset(texts=texts, labels=labels, tokenizer=mock_tokenizer, max_length=10)


def test_cls_dataset_getitem(cls_dataset):
    encoding_dict = cls_dataset[0]

    assert isinstance(encoding_dict, dict)
    assert 'input_ids' in encoding_dict
    assert 'attention_mask' in encoding_dict
    assert 'labels' in encoding_dict

    assert encoding_dict['input_ids'].shape[0] == 10
    assert encoding_dict['attention_mask'].shape[0] == 10

    assert encoding_dict['labels'].item() == 1
