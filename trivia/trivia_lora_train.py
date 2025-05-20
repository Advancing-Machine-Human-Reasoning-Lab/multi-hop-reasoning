import os 
import torch
import argparse
import random

from datasets import Dataset
from huggingface_hub import login
from trl import SFTTrainer, setup_chat_format
from dotenv import load_dotenv
from datasets import load_dataset

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
new_model = f"{model_filename}_trivia_lora"


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



train_size = 200
train_dataset = load_dataset("mandarjoshi/trivia_qa", "rc", split="train")
random_indices = random.sample(range(len(train_dataset)), train_size)

train_dataset = train_dataset.select(random_indices)

dataset = Dataset.from_dict({"input_text": train_dataset['question'], "labels": train_dataset['answer']})


def format_chat_template(row):

    question = row["input_text"]

    answer = row["labels"]["aliases"][0]

    row_json = [{"role": "user", "content": question},
                {"role": "assistant", "content": answer}]
    
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

