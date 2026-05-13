"""Fine-tune BERT for multi-label sentiment + intent classification."""
  import torch
  import numpy as np
  from transformers import (BertTokenizerFast, BertForSequenceClassification,
                             TrainingArguments, Trainer)
  from datasets import Dataset
  from sklearn.model_selection import train_test_split
  from sklearn.metrics import accuracy_score, f1_score

  INTENTS = ["refund", "technical_support", "billing_info", "complaint", "general_info"]
  SENTIMENTS = ["positive", "negative", "neutral"]
  LABELS = INTENTS + SENTIMENTS
  NUM_LABELS = len(LABELS)

  # ── Synthetic training data (replace with real tickets) ───────────────────
  SAMPLES = [
      ("I want a refund for my order, it was broken", [1,0,0,0,0, 0,1,0]),
      ("Your app keeps crashing, please fix it!", [0,1,0,0,0, 0,1,0]),
      ("How much does the premium plan cost?", [0,0,1,0,0, 0,0,1]),
      ("This is the best product I have ever used!", [0,0,0,0,1, 1,0,0]),
      ("I am very disappointed with the service", [0,0,0,1,0, 0,1,0]),
      ("Can you tell me about the refund policy?", [0,0,1,0,0, 0,0,1]),
      ("My payment was charged twice", [0,0,0,1,0, 0,1,0]),
      ("Great support team, very helpful!", [0,0,0,0,1, 1,0,0]),
  ] * 125  # Simulate 1000 samples

  texts  = [s[0] for s in SAMPLES]
  labels = [s[1] for s in SAMPLES]

  train_texts, val_texts, train_labels, val_labels = train_test_split(
      texts, labels, test_size=0.2, random_state=42)

  tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")

  def tokenize(batch):
      return tokenizer(batch["text"], truncation=True, padding="max_length", max_length=128)

  train_ds = Dataset.from_dict({"text": train_texts, "labels": train_labels}).map(tokenize, batched=True)
  val_ds   = Dataset.from_dict({"text": val_texts,   "labels": val_labels}).map(tokenize, batched=True)
  train_ds.set_format("torch", columns=["input_ids","attention_mask","labels"])
  val_ds.set_format("torch", columns=["input_ids","attention_mask","labels"])

  model = BertForSequenceClassification.from_pretrained(
      "bert-base-uncased", num_labels=NUM_LABELS, problem_type="multi_label_classification")

  def compute_metrics(eval_pred):
      logits, labels = eval_pred
      preds = (torch.sigmoid(torch.tensor(logits)) > 0.5).int().numpy()
      return {"f1": f1_score(labels, preds, average="macro", zero_division=0),
              "accuracy": accuracy_score(labels.argmax(1), preds.argmax(1))}

  args = TrainingArguments(output_dir="./bert-classifier", num_train_epochs=3,
      per_device_train_batch_size=16, per_device_eval_batch_size=16,
      warmup_steps=50, weight_decay=0.01, logging_dir="./logs",
      evaluation_strategy="epoch", save_strategy="epoch", load_best_model_at_end=True)

  trainer = Trainer(model=model, args=args, train_dataset=train_ds,
                    eval_dataset=val_ds, compute_metrics=compute_metrics)

  print("Fine-tuning BERT...")
  trainer.train()
  trainer.save_model("./bert-classifier/final")
  tokenizer.save_pretrained("./bert-classifier/final")
  print("Saved to ./bert-classifier/final")
  