from transformers import AutoTokenizer, AutoModelForTokenClassification
import torch


class NERModel:
    def __init__(self, model_path=None):
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path if model_path else 'michiyasunaga/BioLinkBERT-base')
        self.model = AutoModelForTokenClassification.from_pretrained(
            model_path if model_path else 'michiyasunaga/BioLinkBERT-base', num_labels=5)

    def encode(self, texts, labels=None):
        encoding = self.tokenizer(
            texts,
            truncation=True,
            padding=True,
            return_tensors="pt",
        )
        inputs = {
            'input_ids': encoding['input_ids'],
            'attention_mask': encoding['attention_mask'],
        }
        if labels:
            inputs['labels'] = torch.tensor(labels)
        return inputs

    def predict(self, inputs):
        with torch.no_grad():
            outputs = self.model(**inputs)
        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1).squeeze().tolist()
        return predictions

    def save_model(self, path):
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
