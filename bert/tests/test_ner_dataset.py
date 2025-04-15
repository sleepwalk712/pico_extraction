import pytest
import torch

from app.core.ner_dataset import NerDataset


class MockEncoding:
    def __init__(self, text_length, max_length):
        self.data = {
            'input_ids': torch.tensor([[0] * max_length]),
            'attention_mask': torch.tensor([[1] * max_length])
        }
        self._word_ids = list(range(text_length)) + \
            [None] * (max_length - text_length)

    def __getitem__(self, key):
        return self.data[key]

    def word_ids(self, batch_index=0):
        return self._word_ids


class MockTokenizer:
    def __call__(self, text, is_split_into_words, truncation, padding, max_length, return_tensors):
        text_length = min(len(text), max_length)
        return MockEncoding(text_length, max_length)


@pytest.fixture
def mock_tokenizer():
    return MockTokenizer()


@pytest.fixture
def ner_dataset(mock_tokenizer):
    texts = [['Hello', 'world', '!'], ['Another', 'sentence']]
    labels = [[1, 0, 1], [0, 1]]
    return NerDataset(texts=texts, labels=labels, tokenizer=mock_tokenizer, max_length=5)


def test_ner_dataset_getitem(ner_dataset):
    encoding_dict = ner_dataset[0]

    assert isinstance(encoding_dict, dict)
    assert 'input_ids' in encoding_dict
    assert 'attention_mask' in encoding_dict
    assert 'labels' in encoding_dict

    assert encoding_dict['input_ids'].shape[0] == 5
    assert encoding_dict['attention_mask'].shape[0] == 5
    assert encoding_dict['labels'].shape[0] == 5

    assert encoding_dict['labels'].tolist() == [1, 0, 1, -100, -100]
