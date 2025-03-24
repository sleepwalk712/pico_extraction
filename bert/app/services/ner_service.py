from typing import Optional
from transformers import Trainer, TrainingArguments  # type: ignore
from torch.optim import AdamW
import torch

from app.core.ner_dataset import NERDataset
from app.core.ner_model import NERModel


class NERService:
    def __init__(self) -> None:
        self.ner_model = NERModel()

    def align_predictions(
        self,
        predictions: torch.Tensor,
        word_ids: list[Optional[int]]
    ) -> list[int]:
        prediction_labels = predictions.argmax(dim=-1).tolist()[0]

        aligned_labels = []
        prev_word_id = None
        for idx, word_id in enumerate(word_ids):
            if word_id is None or word_id == prev_word_id:
                continue
            aligned_labels.append(prediction_labels[idx])
            prev_word_id = word_id

        return aligned_labels

    def predict(self, text: str) -> list[int]:
        tokenized_input = self.ner_model.tokenizer(
            text.split(),
            is_split_into_words=True,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=512,
        ).to(self.ner_model.device)

        logits = self.ner_model.predict(tokenized_input)

        word_ids = tokenized_input.word_ids(batch_index=0)
        aligned_predictions = self.align_predictions(logits, word_ids)

        return aligned_predictions

    def fine_tune(
        self,
        texts: list[list[str]],
        labels: list[list[int]],
        epochs: int = 3,
        ml_model_path: str = "ner_model"
    ) -> None:
        optimizer = AdamW(self.ner_model.model.parameters(), lr=5e-5)

        training_args = TrainingArguments(
            output_dir=ml_model_path,
            per_device_train_batch_size=32,
            num_train_epochs=epochs,
            logging_dir='./logs',
        )

        train_dataset = NERDataset(
            texts, labels, self.ner_model.tokenizer)

        trainer = Trainer(
            model=self.ner_model.model,
            args=training_args,
            train_dataset=train_dataset,
            optimizers=(optimizer, None),
        )

        trainer.train()
        self.ner_model.save_model(ml_model_path)
