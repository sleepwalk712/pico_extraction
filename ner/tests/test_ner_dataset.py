import pytest
import torch
from app.core.ner_dataset import NERDataset


class MockEncoding:
    def __init__(self, text_length):
        self.data = {
            'input_ids': torch.tensor([0] * text_length),
            'attention_mask': torch.tensor([1] * text_length)
        }

    def __getitem__(self, key):
        return self.data[key]

    def word_ids(self, batch_index=0):
        return list(range(len(self.data['input_ids'])))


class MockTokenizer:
    def __call__(self, text, is_split_into_words, truncation, padding, max_length, return_tensors):
        text_length = len(text) if not truncation else min(
            len(text), max_length)
        return MockEncoding(text_length)


@pytest.fixture
def mock_tokenizer():
    return MockTokenizer()


@pytest.fixture
def ner_dataset(mock_tokenizer):
    texts = [['Hello', 'world', '!'], ['Another', 'sentence']]
    labels = [[1, 0, 1], [0, 1]]
    return NERDataset(texts=texts, labels=labels, tokenizer=mock_tokenizer, max_length=5)


def test_ner_dataset_getitem(ner_dataset):
    encoding_dict = ner_dataset.__getitem__(0)
    assert isinstance(encoding_dict, dict)
    assert 'input_ids' in encoding_dict
    assert 'attention_mask' in encoding_dict
    assert 'labels' in encoding_dict
    assert isinstance(encoding_dict['input_ids'], torch.Tensor)
    assert isinstance(encoding_dict['attention_mask'], torch.Tensor)
    assert isinstance(encoding_dict['labels'], torch.Tensor)
