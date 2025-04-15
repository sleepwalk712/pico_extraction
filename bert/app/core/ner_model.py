import os

from transformers import AutoTokenizer, AutoModelForTokenClassification  # type: ignore
import torch


class NerModel:
    def __init__(self, model_path: str, num_labels: int = 5) -> None:
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu'
        )

        if os.path.isdir(model_path) and os.path.exists(os.path.join(model_path, 'config.json')):
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.model = AutoModelForTokenClassification.from_pretrained(
                model_path).to(self.device)
        else:
            print(
                f"Model not found at {model_path}, loading from Hugging Face..."
            )
            base_model = 'michiyasunaga/BioLinkBERT-base'
            self.tokenizer = AutoTokenizer.from_pretrained(base_model)
            self.model = AutoModelForTokenClassification.from_pretrained(
                base_model,
                num_labels=num_labels,
            ).to(self.device)
            self.save_model(model_path)

    def encode_for_inference(self, texts: list[str]) -> dict[str, torch.Tensor]:
        encoding = self.tokenizer(
            texts,
            is_split_into_words=True,
            truncation=True,
            padding='max_length',
            max_length=512,
            return_tensors="pt",
        ).to(self.device)

        return {
            'input_ids': encoding['input_ids'],
            'attention_mask': encoding['attention_mask'],
        }

    def predict(self, inputs: dict[str, torch.Tensor]) -> torch.Tensor:
        with torch.no_grad():
            outputs = self.model(**inputs)
        return outputs.logits

    def save_model(self, path: str) -> None:
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
