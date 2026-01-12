import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import load_dataset
from trl import SFTTrainer
import argparse

def get_model_and_tokenizer(model_name):
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    return model, tokenizer

def get_lora_config(r=16, alpha=32, dropout=0.05):
    return LoraConfig(
        r=r,
        lora_alpha=alpha,
        target_modules=["query_key_value"],
        lora_dropout=dropout,
        bias="none",
        task_type="CAUSAL_LM"
    )

def format_instruction(example):
    text = f"""### Instruction:
You are a helpful e-commerce customer support assistant. Answer the customer's question professionally and helpfully.

### Customer Query:
{example['instruction']}

### Response:
{example['response']}"""
    return {"text": text}

def train(args):
    model, tokenizer = get_model_and_tokenizer(args.model_name)
    model = prepare_model_for_kbit_training(model)

    lora_config = get_lora_config(args.lora_r, args.lora_alpha, args.lora_dropout)
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    dataset = load_dataset("bitext/Bitext-customer-support-llm-chatbot-training-dataset")
    dataset = dataset.map(format_instruction)
    dataset = dataset['train'].train_test_split(test_size=0.1, seed=42)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation,
        learning_rate=args.learning_rate,
        weight_decay=0.01,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        logging_steps=25,
        save_strategy="epoch",
        evaluation_strategy="epoch",
        fp16=True,
        optim="paged_adamw_8bit",
        report_to="none",
        save_total_limit=2
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset['train'],
        eval_dataset=dataset['test'],
        dataset_text_field="text",
        max_seq_length=args.max_length,
        tokenizer=tokenizer,
        args=training_args
    )

    trainer.train()
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"Model saved to {args.output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="tiiuae/falcon-7b")
    parser.add_argument("--output_dir", type=str, default="../models/falcon-7b-ecommerce-lora")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--gradient_accumulation", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    args = parser.parse_args()
    train(args)
