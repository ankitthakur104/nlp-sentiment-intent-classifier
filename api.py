"""BERT Inference API - batch sentiment + intent classification."""
  import torch
  from fastapi import FastAPI
  from pydantic import BaseModel
  from transformers import BertTokenizerFast, BertForSequenceClassification

  app = FastAPI(title="NLP Classifier API", version="1.0.0")

  LABELS = ["refund","technical_support","billing_info","complaint","general_info",
            "positive","negative","neutral"]

  MODEL_PATH = "./bert-classifier/final"
  tokenizer = BertTokenizerFast.from_pretrained(MODEL_PATH)
  model = BertForSequenceClassification.from_pretrained(MODEL_PATH)
  model.eval()

  class TextRequest(BaseModel):
      texts: list[str]
      threshold: float = 0.5

  @app.post("/classify")
  def classify(req: TextRequest):
      enc = tokenizer(req.texts, truncation=True, padding=True, max_length=128, return_tensors="pt")
      with torch.no_grad():
          logits = model(**enc).logits
      probs = torch.sigmoid(logits).numpy()
      results = []
      for i, text in enumerate(req.texts):
          active = [LABELS[j] for j, p in enumerate(probs[i]) if p > req.threshold]
          results.append({"text": text, "labels": active, "scores": {LABELS[j]: round(float(probs[i][j]), 3) for j in range(len(LABELS))}})
      return {"predictions": results}

  @app.get("/health")
  def health(): return {"status": "online", "model": MODEL_PATH}
  