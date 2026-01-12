import streamlit as st
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

st.set_page_config(
    page_title="E-Commerce FAQ Chatbot",
    page_icon="🛒",
    layout="wide"
)

@st.cache_resource
def load_model():
    MODEL_NAME = "tiiuae/falcon-7b"
    ADAPTER_PATH = "../models/falcon-7b-ecommerce-lora"

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True
    )

    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True
    )

    model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer

def generate_response(model, tokenizer, query, max_tokens, temperature):
    prompt = f"""### Instruction:
You are a helpful e-commerce customer support assistant. Answer the customer's question professionally and helpfully.

### Customer Query:
{query}

### Response:"""

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    response = response.split("### Response:")[-1].strip()
    return response

st.title("E-Commerce FAQ Chatbot")
st.markdown("Fine-tuned Falcon-7B with LoRA for customer support")

st.sidebar.header("Settings")
max_tokens = st.sidebar.slider("Max Response Length", 50, 300, 200)
temperature = st.sidebar.slider("Temperature", 0.1, 1.0, 0.7)

st.sidebar.markdown("---")
st.sidebar.markdown("### About")
st.sidebar.markdown("""
This chatbot uses Falcon-7B fine-tuned with LoRA adapters
on e-commerce customer support data.
""")

if "messages" not in st.session_state:
    st.session_state.messages = []

try:
    with st.spinner("Loading model..."):
        model, tokenizer = load_model()
    st.success("Model loaded successfully!")
except Exception as e:
    st.error(f"Error loading model: {str(e)}")
    st.info("Make sure the model adapter is saved in ../models/falcon-7b-ecommerce-lora")
    st.stop()

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

example_queries = [
    "Where is my order?",
    "How can I return a product?",
    "I want to cancel my subscription",
    "The product I received is damaged"
]

st.markdown("### Try these examples:")
cols = st.columns(4)
for i, query in enumerate(example_queries):
    if cols[i].button(query, key=f"example_{i}"):
        st.session_state.example_query = query

if "example_query" in st.session_state:
    user_input = st.session_state.example_query
    del st.session_state.example_query
else:
    user_input = st.chat_input("Ask a question about your order, returns, or any e-commerce query...")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})

    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Generating response..."):
            response = generate_response(model, tokenizer, user_input, max_tokens, temperature)
        st.markdown(response)

    st.session_state.messages.append({"role": "assistant", "content": response})

if st.sidebar.button("Clear Chat"):
    st.session_state.messages = []
    st.rerun()
