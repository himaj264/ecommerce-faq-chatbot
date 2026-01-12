import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

class EcommerceChatbot:
    def __init__(self, model_name="tiiuae/falcon-7b", adapter_path=None):
        self.model_name = model_name
        self.adapter_path = adapter_path
        self.model = None
        self.tokenizer = None

    def load_model(self):
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True
        )

        base_model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True
        )

        if self.adapter_path:
            self.model = PeftModel.from_pretrained(base_model, self.adapter_path)
        else:
            self.model = base_model

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        print("Model loaded successfully")

    def generate_response(self, query, max_tokens=200, temperature=0.7):
        prompt = f"""### Instruction:
You are a helpful e-commerce customer support assistant. Answer the customer's question professionally and helpfully.

### Customer Query:
{query}

### Response:"""

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

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
    chatbot = EcommerceChatbot(
        model_name="tiiuae/falcon-7b",
        adapter_path="../models/falcon-7b-ecommerce-lora"
    )
    chatbot.load_model()

    test_queries = [
        "Where is my order?",
        "How can I return a product?",
        "I want to cancel my subscription"
    ]

    for query in test_queries:
        print(f"Query: {query}")
        print(f"Response: {chatbot.generate_response(query)}")
        print("-" * 50)
