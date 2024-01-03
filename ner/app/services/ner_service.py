from typing import Optional
from transformers import Trainer, TrainingArguments  # type: ignore
from torch.optim import AdamW

from app.core.ner_dataset import NERDataset
from app.core.ner_model import NERModel


class NERService:
    def __init__(self) -> None:
        self.ner_model: NERModel = NERModel()

    def align_predictions(
        self,
        predictions: list[int],
        word_ids: list[Optional[int]]
    ) -> list[int]:
        aligned_labels: list[int] = []
        for idx, word_id in enumerate(word_ids):
            if word_id is None:
                continue
            elif idx == 0 or word_id != word_ids[idx - 1]:
                aligned_labels.append(predictions[word_id])
            else:
                continue
        return aligned_labels

    def predict(self, text: str) -> list[int]:
        inputs = self.ner_model.encode([text])
        predictions: list[int] = self.ner_model.predict(inputs)
        encoding = self.ner_model.tokenizer(text, return_tensors="pt")
        word_ids: list[Optional[int]] = encoding.word_ids(batch_index=0)
        aligned_predictions: list[int] = self.align_predictions(
            predictions, word_ids)
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

        train_dataset: NERDataset = NERDataset(
            texts, labels, self.ner_model.tokenizer)
        trainer = Trainer(
            model=self.ner_model.model,
            args=training_args,
            train_dataset=train_dataset,
            optimizers=(optimizer, None),
        )

        trainer.train()
        self.ner_model.save_model(ml_model_path)
