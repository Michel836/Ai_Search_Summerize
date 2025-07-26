from datasets import load_dataset
from transformers import AutoTokenizer


def load_data(split: str = "train", sample_size: int = 10000):
    """Load and tokenize a subset of the IMDB dataset."""
    dataset = load_dataset("imdb", split=split)
    if sample_size:
        dataset = dataset.shuffle(seed=42).select(range(sample_size))
    return dataset


def get_tokenizer(model_name: str = "distilbert-base-uncased"):
    return AutoTokenizer.from_pretrained(model_name)
