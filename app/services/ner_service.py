from transformers import Trainer, TrainingArguments
from torch.optim import AdamW

from app.core.ner_dataset import NERDataset
from app.core.ner_model import NERModel


class NERService:
    def __init__(self):
        self.ner_model = NERModel()

    def predict(self, text):
        inputs = self.ner_model.encode(text)
        prediction = self.ner_model.predict(inputs)
        return prediction

    def fine_tune(self, texts, labels, epochs=3, model_path="ner_model"):
        optimizer = AdamW(self.ner_model.model.parameters(), lr=5e-5)

        training_args = TrainingArguments(
            output_dir=model_path,
            per_device_train_batch_size=32,
            num_train_epochs=epochs,
            logging_dir='./logs',
        )

        train_dataset = NERDataset(texts, labels, self.ner_model.tokenizer)
        trainer = Trainer(
            model=self.ner_model.model,
            args=training_args,
            train_dataset=train_dataset,
            optimizers=(optimizer, None),
        )

        trainer.train()
        self.ner_model.save_model(model_path)
