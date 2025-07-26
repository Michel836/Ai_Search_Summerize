from transformers import AutoModelForSequenceClassification
from peft import get_peft_model, LoraConfig, TaskType


MODEL_NAME = "distilbert-base-uncased"


def load_model(num_labels: int = 2):
    base_model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=num_labels
    )
    peft_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        inference_mode=False,
        r=8,
        lora_alpha=16,
        lora_dropout=0.1,
    )
    model = get_peft_model(base_model, peft_config)
    return model
