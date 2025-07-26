from transformers import pipeline

from retrieval import Retriever


class ContextAwareGenerator:
    def __init__(self, model_name: str = "gpt2"):
        self.generator = pipeline("text-generation", model=model_name)
        self.retriever = Retriever()

    def build_retrieval(self, documents):
        self.retriever.build_index(documents)

    def generate(self, query: str, top_k: int = 3, max_new_tokens: int = 50):
        context_docs = self.retriever.query(query, top_k=top_k)
        prompt = query + "\n" + "\n".join(context_docs)
        outputs = self.generator(prompt, max_new_tokens=max_new_tokens)
        return outputs[0]["generated_text"]
