import json
import os
import torch
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, f1_score, confusion_matrix
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments

DATA_DIR = os.path.join(os.path.dirname(__file__), "../data/datasets")
MODEL_DIR = os.path.join(os.path.dirname(__file__), "../models/intent_classifier_v2")

TRAIN_FILE = os.path.join(DATA_DIR, "train.jsonl")
VAL_FILE = os.path.join(DATA_DIR, "val.jsonl")
TEST_FILE = os.path.join(DATA_DIR, "test.jsonl")

def load_data(filepath):
    texts, labels = [], []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip(): continue
            data = json.loads(line)
            texts.append(data["text"])
            labels.append(data["label"])
    return texts, labels

def main():
    os.makedirs(MODEL_DIR, exist_ok=True)

    # 1. Load Data
    X_train, y_train_text = load_data(TRAIN_FILE)
    X_val, y_val_text = load_data(VAL_FILE)
    X_test, y_test_text = load_data(TEST_FILE)

    # 2. Setup Label Mapping (exactly 10 classes)
    unique_labels = sorted(list(set(y_train_text + y_val_text + y_test_text)))
    label2id = {label: i for i, label in enumerate(unique_labels)}
    id2label = {i: label for i, label in enumerate(unique_labels)}

    # Encode labels
    y_train = [label2id[l] for l in y_train_text]
    y_val = [label2id[l] for l in y_val_text]
    y_test = [label2id[l] for l in y_test_text]

    # Save mapping
    with open(os.path.join(MODEL_DIR, "label_mapping.json"), "w", encoding="utf-8") as f:
        json.dump(id2label, f, indent=2)

    # 3. Base Model
    model_name = "law-ai/InLegalBERT"
    print(f"\nLoading tokenizer and model: {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, 
        num_labels=len(unique_labels),
        label2id=label2id,
        id2label=id2label
    )

    def tokenize(texts_list):
        return tokenizer(texts_list, padding=True, truncation=True, max_length=128)

    class CustomDataset(torch.utils.data.Dataset):
        def __init__(self, encodings, labels):
            self.encodings = encodings
            self.labels = labels

        def __getitem__(self, idx):
            item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
            item['labels'] = torch.tensor(self.labels[idx])
            return item

        def __len__(self):
            return len(self.labels)

    train_dataset = CustomDataset(tokenize(X_train), y_train)
    val_dataset = CustomDataset(tokenize(X_val), y_val)
    test_dataset = CustomDataset(tokenize(X_test), y_test)

    def compute_metrics(pred):
        labels = pred.label_ids
        preds = pred.predictions.argmax(-1)
        acc = accuracy_score(labels, preds)
        macro_f1 = f1_score(labels, preds, average="macro", zero_division=0)
        return {"accuracy": acc, "f1": macro_f1}

    # 4. Training Config
    CHECKPOINT_DIR = os.path.join(MODEL_DIR, "checkpoints")
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=CHECKPOINT_DIR,
        num_train_epochs=3,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=16,
        warmup_steps=47,
        weight_decay=0.01,
        learning_rate=5e-5,
        eval_strategy="steps",
        eval_steps=50,
        save_strategy="steps",
        save_steps=50,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        logging_steps=25,
        report_to="none",
        use_cpu=True
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics
    )

    import glob
    checkpoints = glob.glob(os.path.join(CHECKPOINT_DIR, "checkpoint-*"))
    latest_checkpoint = max(checkpoints, key=os.path.getctime) if checkpoints else None

    # 5. Train
    print("\nStarting fine-tuning...")
    if latest_checkpoint:
        print(f"Resuming from checkpoint: {latest_checkpoint}")
        trainer.train(resume_from_checkpoint=latest_checkpoint)
    else:
        trainer.train()

    # 6. Evaluate on Test Set
    print("\nEvaluating on held-out TEST set...")
    predictions = trainer.predict(test_dataset)
    preds = predictions.predictions.argmax(-1)
    
    acc = accuracy_score(y_test, preds)
    macro_f1 = f1_score(y_test, preds, average="macro", zero_division=0)
    
    precision, recall, fscore, support = precision_recall_fscore_support(y_test, preds, average=None, zero_division=0)
    
    print(f"\nTest Accuracy: {acc:.4f}")
    print(f"Test Macro F1: {macro_f1:.4f}")
    print("\n--- Per-Category F1 Scores ---")
    for i, cls in enumerate(unique_labels):
        print(f"{cls}: {fscore[i]:.4f}")
        
    print("\n--- Confusion Matrix ---")
    cm = confusion_matrix(y_test, preds)
    print(cm)
    
    # Save best model
    print(f"\nSaving final model to {MODEL_DIR}...")
    trainer.save_model(MODEL_DIR)
    tokenizer.save_pretrained(MODEL_DIR)

if __name__ == "__main__":
    main()
