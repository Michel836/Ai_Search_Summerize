# Ai Search Summarize

This project demonstrates a pipeline for fine-tuning a language model for sentiment
classification and generating context-aware responses.

The code includes:
- Data loading utilities using the IMDB dataset.
- A parameter-efficient LoRA fine-tuning setup for a DistilBERT classifier.
- Training script with Hugging Face Transformers `Trainer`.
- A FAISS-based retrieval component built with sentence-transformers.
- A text generation module that augments prompts with retrieved context.
- A simple Streamlit application that ties everything together.

Install dependencies with:

```bash
pip install -r requirements.txt
```

Then run the Streamlit app:

```bash
streamlit run src/app.py
```
