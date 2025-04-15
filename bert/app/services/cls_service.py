from sklearn.model_selection import train_test_split  # type: ignore
from transformers import Trainer, TrainingArguments  # type: ignore
from torch.optim import AdamW

from app.core.cls_dataset import ClassificationDataset
from app.core.cls_model import ClassificationModel


class ClassificationService:
    def __init__(self, model_path: str, num_labels: int = 2) -> None:
        self.model_path = model_path
        self.cls_model = ClassificationModel(
            model_path=model_path, num_labels=num_labels)

    def predict(self, text: str) -> int:
        tokenized_input = self.cls_model.encode_for_inference(text)
        logits = self.cls_model.predict(tokenized_input)
        prediction = int(logits.argmax(dim=-1).item())
        return prediction

    def predict_batch(self, texts: list[str]) -> list[int]:
        tokenized_input = self.cls_model.encode_for_batch_inference(texts)
        logits = self.cls_model.predict(tokenized_input)
        predictions = logits.argmax(dim=-1).tolist()
        return predictions

    def fine_tune(
        self,
        texts: list[str],
        labels: list[int],
        epochs: int = 3,
        validation_split: float = 0.2,
    ) -> None:
        train_texts, val_texts, train_labels, val_labels = train_test_split(
            texts, labels, test_size=validation_split, random_state=42
        )

        train_dataset = ClassificationDataset(
            train_texts,
            train_labels,
            self.cls_model.tokenizer
        )
        val_dataset = ClassificationDataset(
            val_texts,
            val_labels,
            self.cls_model.tokenizer
        )

        optimizer = AdamW(self.cls_model.model.parameters(), lr=5e-5)
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
            model=self.cls_model.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            optimizers=(optimizer, None),
        )

        trainer.train()
        self.cls_model.save_model(self.model_path)
