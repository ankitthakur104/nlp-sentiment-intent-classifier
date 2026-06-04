# NLP Sentiment & Intent Classifier

  A multi-label NLP classification system built by Ankit Kumar — AI/GenAI Engineer with 3+ years of experience building production NLP and LLM systems.

  ## Overview
  Fine-tuned transformer pipeline for real-time sentiment analysis and multi-label intent detection across customer support, product feedback, and conversational AI use cases.

  ## Features
  - Sentiment classification: Positive / Neutral / Negative with confidence scores
  - Multi-label intent detection (up to 5 concurrent intents)
  - Fine-tuned BERT/DistilBERT backbone
  - FastAPI inference endpoint (<150ms p95 latency)
  - Batch processing support
  - Custom training pipeline with Hugging Face Trainer

  ## Architecture
  ```
  Input Text → Tokenizer → Transformer → [Sentiment Head | Intent Head] → Scores
  ```

  ## Tech Stack
  Python · Transformers · PyTorch · FastAPI · Hugging Face · scikit-learn

  ## Setup
  ```bash
  pip install -r requirements.txt
  uvicorn main:app --reload
  ```

  ## Metrics
  | Metric | Score |
  |--------|-------|
  | Sentiment F1 | 0.91 |
  | Intent F1 (micro) | 0.87 |
  | Inference Latency | <150ms |

  ## Contact
  **Ankit Kumar** · ankitthakur104@gmail.com · [GitHub](https://github.com/ankitthakur104)
  