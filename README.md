# NLP Sentiment & Intent Classifier

Fine-tuned BERT for multi-label sentiment analysis and intent detection over customer support tickets.

## Features
- Fine-tuned bert-base-uncased on 200K+ labelled samples
- Multi-label: sentiment (positive/negative/neutral) + intent (refund/support/info/complaint)
- 91% intent accuracy on holdout set
- FastAPI inference server with batch prediction
- Dockerized for scalable deployment

## Stack
Python · PyTorch · Transformers (HuggingFace) · FastAPI · Docker

## Setup
```bash
pip install -r requirements.txt
python train.py    # Fine-tune BERT
uvicorn api:app --reload
```
