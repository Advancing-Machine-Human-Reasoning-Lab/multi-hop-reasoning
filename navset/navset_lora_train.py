import os 
import torch
import argparse
import pandas as pd
import random

from datasets import Dataset
from huggingface_hub import login
from trl import SFTTrainer, setup_chat_format
from dotenv import load_dotenv

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    EarlyStoppingCallback,
)

from peft import (
    LoraConfig,
    get_peft_model,
)

random.seed(42)

load_dotenv(".env")
hf_token = os.getenv("HF_TOKEN")
login(token = hf_token)

argparser = argparse.ArgumentParser()

argparser.add_argument(
    "--model_name",
    type=str,
    default="meta-llama/Meta-Llama-3-8B-Instruct",
    help="Model name. Can be any of the huggingface models.",
)

args = argparser.parse_args()
print(args)


model_filename = args.model_name
if "/" or ":" in args.model_name:
    model_filename = args.model_name.replace("/","_")
    model_filename = model_filename.replace(":","_")


base_model = args.model_name
new_model = f"{model_filename}_navset_lora"


# QLORA:
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

model = AutoModelForCausalLM.from_pretrained(
    base_model,
    quantization_config=bnb_config,
    device_map="auto",
    attn_implementation="eager",
)


tokenizer = AutoTokenizer.from_pretrained(base_model)
model, tokenizer = setup_chat_format(model, tokenizer)

if args.model_name == "microsoft/Phi-3.5-mini-instruct":
    lora_layers = ['qkv_proj']
else:
    lora_layers = ['k_proj', 'q_proj', 'v_proj']


# LORA:
peft_config = LoraConfig(
    r=4, 
    lora_alpha=8,
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=lora_layers
)

model = get_peft_model(model, peft_config)

dataset = pd.read_csv('navset.tsv', delimiter='\t')

train_dataset = dataset[dataset["split"] == "train"]

dataset = Dataset.from_dict({"input_text": train_dataset['text'], "labels": train_dataset['facts']})


def format_chat_template(row):

    query = "Convert the scenario into the following facts: starboard, port, head-on, engine, sail, restricted."

    scenario = row["input_text"]
    user_prompt = f"Scenario: {scenario}\n{query}"

    facts = row["labels"]
    out1 = f"Facts:\n{facts}"

    row_json = [{"role": "user", "content": user_prompt},
                {"role": "assistant", "content": out1}]
    
    row["text"] = tokenizer.apply_chat_template(row_json, tokenize=False)

    return row


dataset = dataset.map(format_chat_template, num_proc=4)
dataset = dataset.shuffle(seed=42)
dataset = dataset.train_test_split(test_size=0.1, seed=42)


training_arguments = TrainingArguments(
    output_dir=new_model,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=2,
    optim="paged_adamw_32bit",
    num_train_epochs=200,
    evaluation_strategy="steps",
    eval_steps=2,
    logging_steps=1,
    warmup_steps=10,
    logging_strategy="steps",
    learning_rate=1e-06,
    fp16=False,
    bf16=False,
    group_by_length=True,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    save_total_limit=2,
)

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    peft_config=peft_config,
    max_seq_length=512,
    dataset_text_field="text",
    tokenizer=tokenizer,
    args=training_arguments,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=20)],
)

trainer.train()

model.config.use_cache = True
model.save_pretrained(new_model)
tokenizer.save_pretrained(new_model)

