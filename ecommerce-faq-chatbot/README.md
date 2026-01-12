# E-Commerce FAQ Chatbot with Parameter-Efficient Fine-Tuning

A Masters-level project demonstrating fine-tuning of Falcon-7B using LoRA (Low-Rank Adaptation) for domain-specific e-commerce customer support responses.

## Project Overview

This project implements an end-to-end LLM fine-tuning pipeline for an e-commerce FAQ chatbot. We compare three approaches:
1. **Zero-shot inference** - Using the base Falcon-7B model without any fine-tuning
2. **Fine-tuned model** - Falcon-7B fine-tuned with LoRA adapters on e-commerce FAQ data
3. **RAG-based approach** - Retrieval-Augmented Generation using vector similarity search

## Key Features

- Parameter-efficient fine-tuning using LoRA/PEFT
- 4-bit quantization for memory efficiency
- BLEU score evaluation for response quality
- Comparative analysis of different approaches
- Interactive Streamlit deployment

## Architecture Diagram

### System Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         E-COMMERCE FAQ CHATBOT                              │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│                              DATA PIPELINE                                   │
│  ┌──────────────┐    ┌──────────────────┐    ┌────────────────────┐        │
│  │  Bitext      │    │  Preprocessing   │    │  Train/Val Split   │        │
│  │  Dataset     │───▶│  & Formatting    │───▶│  (90%/10%)         │        │
│  │  (27K pairs) │    │                  │    │                    │        │
│  └──────────────┘    └──────────────────┘    └────────────────────┘        │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           TRAINING PIPELINE                                  │
│                                                                             │
│  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐         │
│  │   Falcon-7B     │    │   4-bit         │    │   LoRA          │         │
│  │   Base Model    │───▶│   Quantization  │───▶│   Adapters      │         │
│  │                 │    │   (NF4)         │    │   (r=16, α=32)  │         │
│  └─────────────────┘    └─────────────────┘    └─────────────────┘         │
│                                                        │                    │
│                                                        ▼                    │
│                                               ┌─────────────────┐           │
│                                               │   SFT Trainer   │           │
│                                               │   (3 epochs)    │           │
│                                               └─────────────────┘           │
│                                                        │                    │
│                                                        ▼                    │
│                                               ┌─────────────────┐           │
│                                               │  Saved LoRA     │           │
│                                               │  Adapters       │           │
│                                               └─────────────────┘           │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Three Approaches Comparison

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                            INFERENCE APPROACHES                              │
└─────────────────────────────────────────────────────────────────────────────┘

                            ┌─────────────────┐
                            │  Customer Query │
                            │  "Where is my   │
                            │   order?"       │
                            └────────┬────────┘
                                     │
           ┌─────────────────────────┼─────────────────────────┐
           │                         │                         │
           ▼                         ▼                         ▼
┌──────────────────┐    ┌──────────────────┐    ┌──────────────────┐
│   ZERO-SHOT      │    │   FINE-TUNED     │    │      RAG         │
│                  │    │                  │    │                  │
│ ┌──────────────┐ │    │ ┌──────────────┐ │    │ ┌──────────────┐ │
│ │  Falcon-7B   │ │    │ │  Falcon-7B   │ │    │ │  Embedding   │ │
│ │  (Base)      │ │    │ │  + LoRA      │ │    │ │  Model       │ │
│ └──────────────┘ │    │ └──────────────┘ │    │ └──────┬───────┘ │
│        │         │    │        │         │    │        │         │
│        ▼         │    │        ▼         │    │        ▼         │
│ ┌──────────────┐ │    │ ┌──────────────┐ │    │ ┌──────────────┐ │
│ │   Generic    │ │    │ │   Domain     │ │    │ │    FAISS     │ │
│ │   Response   │ │    │ │   Specific   │ │    │ │    Index     │ │
│ └──────────────┘ │    │ └──────────────┘ │    │ └──────┬───────┘ │
│                  │    │                  │    │        │         │
│                  │    │                  │    │        ▼         │
│                  │    │                  │    │ ┌──────────────┐ │
│                  │    │                  │    │ │  Retrieved   │ │
│                  │    │                  │    │ │  Context     │ │
│                  │    │                  │    │ └──────┬───────┘ │
│                  │    │                  │    │        │         │
│                  │    │                  │    │        ▼         │
│                  │    │                  │    │ ┌──────────────┐ │
│                  │    │                  │    │ │  Falcon-7B   │ │
│                  │    │                  │    │ │  + Context   │ │
│                  │    │                  │    │ └──────────────┘ │
└────────┬─────────┘    └────────┬─────────┘    └────────┬─────────┘
         │                       │                       │
         ▼                       ▼                       ▼
   ┌───────────┐           ┌───────────┐           ┌───────────┐
   │ BLEU: Low │           │BLEU: High │           │BLEU: Medium│
   └───────────┘           └───────────┘           └───────────┘
```

### LoRA Architecture Detail

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         LoRA ADAPTER ARCHITECTURE                            │
└─────────────────────────────────────────────────────────────────────────────┘

                    Original Transformer Layer
                    ┌─────────────────────────┐
                    │                         │
     Input ────────▶│   Frozen Weights (W)    │────────▶ Output
       │            │      [d × d]            │            ▲
       │            └─────────────────────────┘            │
       │                                                   │
       │            LoRA Adapter (Trainable)               │
       │            ┌─────────────────────────┐            │
       │            │                         │            │
       └───────────▶│  Down: W_A [d × r]      │            │
                    │         ↓               │            │
                    │  Up:   W_B [r × d]      │───────────▶⊕
                    │                         │
                    │  r = 16 (rank)          │
                    │  α = 32 (scaling)       │
                    └─────────────────────────┘

    Parameters:
    - Original W: d × d = 4096 × 4096 = 16.7M params (frozen)
    - LoRA W_A:   d × r = 4096 × 16   = 65K params  (trainable)
    - LoRA W_B:   r × d = 16 × 4096   = 65K params  (trainable)
    - Total trainable: ~0.8% of original parameters
```

### Deployment Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         DEPLOYMENT (STREAMLIT)                               │
└─────────────────────────────────────────────────────────────────────────────┘

┌──────────────┐     ┌──────────────────────────────────────────────────────┐
│              │     │                  GPU Server (16GB+)                  │
│    User      │     │  ┌────────────┐  ┌─────────────┐  ┌──────────────┐  │
│   Browser    │────▶│  │ Streamlit  │─▶│  Falcon-7B  │─▶│   Response   │  │
│              │◀────│  │    UI      │◀─│  + LoRA     │◀─│  Generation  │  │
│              │     │  └────────────┘  └─────────────┘  └──────────────┘  │
└──────────────┘     └──────────────────────────────────────────────────────┘
```

## Project Structure

```
ecommerce-faq-chatbot/
├── notebooks/
│   ├── 01_data_preprocessing.ipynb    # Dataset preparation
│   ├── 02_fine_tuning.ipynb           # LoRA fine-tuning (Colab)
│   ├── 03_evaluation.ipynb            # Model evaluation
│   └── 04_comparison.ipynb            # Zero-shot vs Fine-tuned vs RAG
├── src/
│   ├── data_utils.py                  # Data preprocessing utilities
│   ├── train.py                       # Training script
│   ├── inference.py                   # Inference utilities
│   └── rag_baseline.py                # RAG implementation
├── app/
│   └── streamlit_app.py               # Streamlit deployment
├── data/                              # Dataset directory
├── models/                            # Saved model checkpoints
├── requirements.txt
└── README.md
```

## Hardware Requirements

- **Recommended**: Google Colab Pro (A100/V100 GPU with 16GB+ VRAM)
- **Alternative**: Any machine with 16GB+ GPU memory
- **Not suitable**: MacBook Air M2 8GB (insufficient for Falcon-7B)

## Installation

### For Google Colab (Recommended)

1. Upload the notebooks to Google Colab
2. Enable GPU runtime: `Runtime > Change runtime type > T4/A100 GPU`
3. Run the installation cell in each notebook

### For Local Setup

```bash
git clone https://github.com/yourusername/ecommerce-faq-chatbot.git
cd ecommerce-faq-chatbot
pip install -r requirements.txt
```

## Quick Start Guide

### Step 1: Data Preprocessing

Open `notebooks/01_data_preprocessing.ipynb` in Colab and run all cells. This will:
- Download the e-commerce FAQ dataset from Hugging Face
- Clean and format the data for instruction tuning
- Save the processed dataset

### Step 2: Fine-Tuning (Colab Pro Required)

Open `notebooks/02_fine_tuning.ipynb` in Colab Pro and run all cells. This will:
- Load Falcon-7B with 4-bit quantization
- Configure LoRA adapters (rank=16, alpha=32)
- Train the model for 3 epochs
- Save the LoRA adapters to Google Drive

**Expected Training Time**: ~2-3 hours on A100 GPU

### Step 3: Evaluation

Open `notebooks/03_evaluation.ipynb` to:
- Load the fine-tuned model
- Calculate BLEU scores
- Generate sample responses
- Analyze model performance

### Step 4: Comparison Analysis

Open `notebooks/04_comparison.ipynb` to:
- Compare zero-shot, fine-tuned, and RAG approaches
- Visualize performance metrics
- Generate comparison tables

### Step 5: Deployment (Optional)

For Streamlit deployment on a GPU machine:

```bash
cd app
streamlit run streamlit_app.py
```

## Dataset

We use the **Bitext Customer Support LLM Chatbot Training Dataset** from Hugging Face:
- Dataset: `bitext/Bitext-customer-support-llm-chatbot-training-dataset`
- Size: ~27,000 instruction-response pairs
- Categories: Order tracking, refunds, account issues, product inquiries, etc.

## Model Architecture

### Base Model
- **Falcon-7B**: A 7 billion parameter causal language model by TII

### LoRA Configuration
```python
LoraConfig(
    r=16,                    # Rank of update matrices
    lora_alpha=32,           # Scaling factor
    target_modules=["query_key_value"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)
```

### Quantization
- 4-bit quantization using BitsAndBytes
- NF4 quantization type
- Double quantization enabled

## Training Configuration

| Parameter | Value |
|-----------|-------|
| Learning Rate | 2e-4 |
| Batch Size | 4 |
| Gradient Accumulation | 4 |
| Epochs | 3 |
| Max Sequence Length | 512 |
| Optimizer | paged_adamw_8bit |
| LR Scheduler | cosine |

## Results

### BLEU Score Comparison

| Approach | BLEU-1 | BLEU-4 |
|----------|--------|--------|
| Zero-shot | ~0.15 | ~0.05 |
| Fine-tuned | ~0.45 | ~0.25 |
| RAG | ~0.35 | ~0.18 |

### Key Observations

1. **Fine-tuning significantly improves domain-specific responses**
   - The model learns e-commerce terminology and response patterns
   - More coherent and contextually appropriate answers

2. **Zero-shot struggles with domain-specific queries**
   - Generic responses lacking e-commerce context
   - Often fails to understand customer intent

3. **RAG provides a middle ground**
   - Good for factual queries with available context
   - Limited by the quality of retrieved documents

## Sample Outputs

### Query: "Where is my order?"

**Zero-shot Response:**
> "I'm not sure what order you're referring to. Could you provide more details?"

**Fine-tuned Response:**
> "I understand you'd like to track your order. To help you locate your package, could you please provide your order number? You can find this in your confirmation email. Once I have that, I can check the current status and estimated delivery date for you."

**RAG Response:**
> "To track your order, please visit our order tracking page and enter your order number from the confirmation email."

## Limitations

- Model requires significant GPU memory (16GB+)
- Fine-tuning time can be lengthy without powerful hardware
- Limited to English language support
- Response quality depends on training data coverage

## Future Work

- Experiment with different LoRA ranks and configurations
- Add multi-turn conversation support
- Implement caching for faster inference
- Add more evaluation metrics (ROUGE, BERTScore)

## References

1. Hu et al., "LoRA: Low-Rank Adaptation of Large Language Models" (2021)
2. Falcon LLM by Technology Innovation Institute
3. Hugging Face PEFT Library Documentation
4. Bitext Customer Support Dataset

## License

This project is for educational purposes as part of a Master's degree program.

## Author

Masters Student Project - [Your Name]
