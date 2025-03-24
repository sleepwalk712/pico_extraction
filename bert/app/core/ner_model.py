from typing import Optional
from transformers import AutoTokenizer, AutoModelForTokenClassification  # type: ignore
import torch


class NERModel:
    def __init__(self, ml_model_path: Optional[str] = None) -> None:
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')
        model_path = ml_model_path if ml_model_path else 'michiyasunaga/BioLinkBERT-base'
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForTokenClassification.from_pretrained(
            model_path, num_labels=5).to(self.device)

    def encode(self, texts: list[list[str]], labels: Optional[list[list[int]]] = None) -> dict[str, torch.Tensor]:
        encoding = self.tokenizer(
            texts,
            is_split_into_words=True,
            truncation=True,
            padding='max_length',
            max_length=512,
            return_tensors="pt",
        ).to(self.device)

        inputs = {
            'input_ids': encoding['input_ids'],
            'attention_mask': encoding['attention_mask'],
        }

        if labels is not None:
            label_ids_batch = []
            for i, label in enumerate(labels):
                word_ids = encoding.word_ids(batch_index=i)
                label_ids = []
                previous_word_idx = None
                for word_idx in word_ids:
                    if word_idx is None:
                        label_ids.append(-100)
                    elif word_idx != previous_word_idx:
                        label_ids.append(label[word_idx])
                    else:
                        label_ids.append(-100)
                    previous_word_idx = word_idx
                label_ids_batch.append(label_ids)

            inputs['labels'] = torch.tensor(label_ids_batch).to(self.device)

        return inputs

    def predict(self, inputs: dict[str, torch.Tensor]) -> torch.Tensor:
        with torch.no_grad():
            outputs = self.model(**inputs)
        return outputs.logits

    def save_model(self, path: str) -> None:
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
