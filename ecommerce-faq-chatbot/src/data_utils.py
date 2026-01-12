from datasets import load_dataset, Dataset, DatasetDict
from sklearn.model_selection import train_test_split
import pandas as pd

def load_ecommerce_dataset():
    dataset = load_dataset("bitext/Bitext-customer-support-llm-chatbot-training-dataset")
    return dataset

def format_instruction(example):
    text = f"""### Instruction:
You are a helpful e-commerce customer support assistant. Answer the customer's question professionally and helpfully.

### Customer Query:
{example['instruction']}

### Response:
{example['response']}"""
    return {"text": text}

def prepare_dataset(dataset, test_size=0.1, seed=42):
    dataset = dataset.map(format_instruction)
    split_dataset = dataset['train'].train_test_split(test_size=test_size, seed=seed)
    return split_dataset['train'], split_dataset['test']

def save_processed_data(train_dataset, val_dataset, output_dir):
    dataset_dict = DatasetDict({
        'train': train_dataset,
        'validation': val_dataset
    })
    dataset_dict.save_to_disk(output_dir)
    print(f"Dataset saved to {output_dir}")

if __name__ == "__main__":
    dataset = load_ecommerce_dataset()
    train_data, val_data = prepare_dataset(dataset)
    print(f"Training samples: {len(train_data)}")
    print(f"Validation samples: {len(val_data)}")
    save_processed_data(train_data, val_data, "../data/processed")
