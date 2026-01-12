# E-Commerce FAQ Chatbot with Parameter-Efficient Fine-Tuning

A  project demonstrating fine-tuning of Falcon-7B using LoRA (Low-Rank Adaptation) for domain-specific e-commerce customer support responses.

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


## Installation

### For Google Colab 

1. Upload the notebooks to Google Colab
2. Enable GPU runtime: `Runtime > Change runtime type > T4/A100 GPU`
3. Run the installation cell in each notebook

### For Local Setup

```bash
git clone https://github.com/yourusername/ecommerce-faq-chatbot.git
cd ecommerce-faq-chatbot
pip install -r requirements.txt
```



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

### Step 5: Deployment 

For Streamlit deployment on a GPU machine


## Dataset

We use the **Bitext Customer Support LLM Chatbot Training Dataset** from Hugging Face:
- Dataset: `bitext/Bitext-customer-support-llm-chatbot-training-dataset`
- Size: ~27,000 instruction-response pairs
- Categories: Order tracking, refunds, account issues, product inquiries, etc.

## Model Architecture

### Base Model
- **Falcon-7B**: A 7 billion parameter causal language model by TII





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



