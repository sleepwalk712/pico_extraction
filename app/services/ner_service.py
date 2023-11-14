from transformers import Trainer, TrainingArguments
from torch.optim import AdamW

from app.core.ner_dataset import NERDataset
from app.core.ner_model import NERModel


class NERService:
    def __init__(self, label_map=None):
        self.ner_model = NERModel()
        self.label_map = label_map

    def align_predictions(self, predictions, word_ids):
        aligned_labels = []
        for idx, word_id in enumerate(word_ids):
            if word_id is None:
                aligned_labels.append("O")
            elif idx == 0 or word_id != word_ids[idx - 1]:
                aligned_labels.append(self.label_map[predictions[word_id]])
            else:
                aligned_labels.append("O")
        return aligned_labels

    def predict(self, text):
        inputs = self.ner_model.encode([text])
        predictions = self.ner_model.predict(inputs)
        encoding = self.ner_model.tokenizer(text, return_tensors="pt")
        word_ids = encoding.word_ids(batch_index=0)
        aligned_predictions = self.align_predictions(predictions, word_ids)
        return aligned_predictions

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
