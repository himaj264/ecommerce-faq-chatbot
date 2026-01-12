import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from sentence_transformers import SentenceTransformer
from datasets import load_dataset
import faiss
import numpy as np

class RAGChatbot:
    def __init__(self, model_name="tiiuae/falcon-7b"):
        self.model_name = model_name
        self.model = None
        self.tokenizer = None
        self.embedding_model = None
        self.index = None
        self.instructions = []
        self.responses = []

    def load_model(self):
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True
        )

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        print("LLM loaded successfully")

    def build_index(self, num_samples=5000):
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

        dataset = load_dataset("bitext/Bitext-customer-support-llm-chatbot-training-dataset")
        data = dataset['train'].select(range(min(num_samples, len(dataset['train']))))

        self.instructions = [item['instruction'] for item in data]
        self.responses = [item['response'] for item in data]

        print("Creating embeddings...")
        embeddings = self.embedding_model.encode(self.instructions, show_progress_bar=True)
        embeddings = np.array(embeddings).astype('float32')

        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(embeddings)

        print(f"Index created with {self.index.ntotal} vectors")

    def retrieve(self, query, k=3):
        query_embedding = self.embedding_model.encode([query])
        distances, indices = self.index.search(np.array(query_embedding).astype('float32'), k)

        retrieved = []
        for idx in indices[0]:
            retrieved.append({
                'instruction': self.instructions[idx],
                'response': self.responses[idx]
            })
        return retrieved

    def generate_response(self, query, max_tokens=200, temperature=0.7):
        retrieved = self.retrieve(query, k=3)

        context = ""
        for i, item in enumerate(retrieved, 1):
            context += f"Example {i}:\nQ: {item['instruction']}\nA: {item['response']}\n\n"

        prompt = f"""### Instruction:
You are a helpful e-commerce customer support assistant. Use the following examples to help answer the customer's question.

### Context:
{context}

### Customer Query:
{query}

### Response:"""

        inputs = self.tokenizer(prompt, return_tensors="pt", max_length=1024, truncation=True).to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature,
                top_p=0.9,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )

        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = response.split("### Response:")[-1].strip()
        return response

if __name__ == "__main__":
    chatbot = RAGChatbot()
    chatbot.load_model()
    chatbot.build_index()

    test_queries = [
        "Where is my order?",
        "How can I return a product?",
        "I want to cancel my subscription"
    ]

    for query in test_queries:
        print(f"Query: {query}")
        print(f"Response: {chatbot.generate_response(query)}")
        print("-" * 50)
