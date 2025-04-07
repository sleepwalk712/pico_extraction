from typing import Optional

from torch.utils.data import Dataset
import torch
from transformers import PreTrainedTokenizer  # type: ignore

from app.core.types import EncodingDict


class NERDataset(Dataset):
    def __init__(
        self,
        texts: list[list[str]],
        labels: list[list[int]],
        tokenizer: PreTrainedTokenizer,
        max_length: int = 512,
    ) -> None:
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> EncodingDict:
        text: list[str] = self.texts[idx]
        labels: list[int] = self.labels[idx]

        encoding = self.tokenizer(
            text,
            is_split_into_words=True,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors="pt"
        )

        word_ids: list[int] = encoding.word_ids(batch_index=0)

        previous_word_idx: Optional[int] = None
        label_ids: list[int] = []
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != previous_word_idx:
                label_ids.append(labels[word_idx])
            else:
                label_ids.append(-100)
            previous_word_idx = word_idx

        label_ids += [-100] * (self.max_length - len(label_ids))

        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels': torch.tensor(label_ids)
        }
