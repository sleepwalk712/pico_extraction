from typing import Optional
from transformers import AutoTokenizer, AutoModelForTokenClassification  # type: ignore
import torch


class NERModel:
    def __init__(self, ml_model_path: Optional[str] = None) -> None:
        model_path: str = ml_model_path if ml_model_path else 'michiyasunaga/BioLinkBERT-base'
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForTokenClassification.from_pretrained(
            model_path, num_labels=5)

    def encode(self, texts: list[str], labels: Optional[list[int]] = None) -> dict[str, torch.Tensor]:
        encoding = self.tokenizer(
            texts,
            truncation=True,
            padding=True,
            return_tensors="pt",
        )
        inputs: dict[str, torch.Tensor] = {
            'input_ids': encoding['input_ids'],
            'attention_mask': encoding['attention_mask'],
        }
        if labels is not None:
            inputs['labels'] = torch.tensor(labels)
        return inputs

    def predict(self, inputs: dict[str, torch.Tensor]) -> list[int]:
        with torch.no_grad():
            outputs = self.model(**inputs)
        logits = outputs.logits
        predictions: list[int] = torch.argmax(
            logits, dim=-1).squeeze().tolist()
        return predictions

    def save_model(self, path: str) -> None:
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
