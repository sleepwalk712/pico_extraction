import os

from transformers import AutoTokenizer, AutoModelForSequenceClassification  # type: ignore
import torch


class ClassificationModel:
    def __init__(self, model_path: str, num_labels: int = 2) -> None:
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu'
        )

        if os.path.isdir(model_path) and os.path.exists(os.path.join(model_path, 'config.json')):
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.model = AutoModelForSequenceClassification.from_pretrained(
                model_path).to(self.device)
        else:
            print(
                f"Model not found at {model_path}, loading from Hugging Face..."
            )
            base_model = 'michiyasunaga/BioLinkBERT-base'
            self.tokenizer = AutoTokenizer.from_pretrained(base_model)
            self.model = AutoModelForSequenceClassification.from_pretrained(
                base_model,
                num_labels=num_labels,
            ).to(self.device)
            self.save_model(model_path)

    def encode_for_inference(self, text: str) -> dict[str, torch.Tensor]:
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=512,
            return_tensors="pt"
        ).to(self.device)

        return {
            'input_ids': encoding['input_ids'],
            'attention_mask': encoding['attention_mask']
        }

    def encode_for_batch_inference(self, texts: list[str]) -> dict[str, torch.Tensor]:
        encoding = self.tokenizer(
            texts,
            truncation=True,
            padding='max_length',
            max_length=512,
            return_tensors="pt"
        ).to(self.device)

        return {
            'input_ids': encoding['input_ids'],
            'attention_mask': encoding['attention_mask']
        }

    def predict(self, inputs: dict[str, torch.Tensor]) -> torch.Tensor:
        with torch.no_grad():
            outputs = self.model(**inputs)
        return outputs.logits

    def save_model(self, path: str) -> None:
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
