from typing import Optional

from sklearn.model_selection import train_test_split  # type: ignore
from transformers import Trainer, TrainingArguments  # type: ignore
from torch.optim import AdamW
import torch

from app.core.ner_dataset import NERDataset
from app.core.ner_model import NERModel


class NERService:
    def __init__(self, model_path: str) -> None:
        self.model_path = model_path
        self.ner_model = NERModel(model_path=model_path)

    def align_predictions(
        self,
        predictions: torch.Tensor,
        word_ids: list[Optional[int]],
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
        tokenized_input = self.ner_model.encode_for_inference(text.split())
        logits = self.ner_model.predict(tokenized_input)
        word_ids = self.ner_model.tokenizer(
            text.split(),
            is_split_into_words=True,
        ).word_ids()

        return self.align_predictions(logits, word_ids)

    def fine_tune(
        self,
        texts: list[list[str]],
        labels: list[list[int]],
        epochs: int = 3,
        validation_split: float = 0.2,
    ) -> None:
        train_texts, val_texts, train_labels, val_labels = train_test_split(
            texts, labels, test_size=validation_split, random_state=42
        )

        train_dataset = NERDataset(
            train_texts,
            train_labels,
            self.ner_model.tokenizer,
        )
        val_dataset = NERDataset(
            val_texts,
            val_labels,
            self.ner_model.tokenizer,
        )

        optimizer = AdamW(self.ner_model.model.parameters(), lr=5e-5)

        training_args = TrainingArguments(
            output_dir=self.model_path,
            per_device_train_batch_size=32,
            num_train_epochs=epochs,
            logging_dir='./logs',
            eval_strategy='epoch',
            save_strategy='epoch',
            save_total_limit=1,
            load_best_model_at_end=True,
        )

        trainer = Trainer(
            model=self.ner_model.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            optimizers=(optimizer, None),
        )

        trainer.train()
        self.ner_model.save_model(self.model_path)
