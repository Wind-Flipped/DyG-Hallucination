from datasets import Dataset
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM, DataCollatorForSeq2Seq, TrainingArguments, Trainer, \
    GenerationConfig
import torch
from peft import LoraConfig, TaskType, get_peft_model
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["NCCL_P2P_DISABLE"] = "1"
os.environ["NCCL_IB_DISABLE"] = "1"

df = pd.read_json("../new_lora_data/42/graphs_n5_t10_p0.5/ER/No/when degree natural.json")

ds = Dataset.from_pandas(df)
model_path = ""
tokenizer = AutoTokenizer.from_pretrained(model_path,
                                          use_fast=False, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token


def process_func(example):
    MAX_LENGTH = 2048  
    input_ids, attention_mask, labels = [], [], []
    instruction = tokenizer(
        f"<|start_header_id|>user<|end_header_id|>\n\n{example['instruction'] + example['input']}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n",
        add_special_tokens=False)  
    response = tokenizer(f"{example['output']}<|eot_id|>", add_special_tokens=False)
    input_ids = instruction["input_ids"] + response["input_ids"] + [tokenizer.pad_token_id]
    attention_mask = instruction["attention_mask"] + response["attention_mask"] + [1]
    labels = [-100] * len(instruction["input_ids"]) + response["input_ids"] + [tokenizer.pad_token_id]
    if len(input_ids) > MAX_LENGTH: 
        input_ids = input_ids[:MAX_LENGTH]
        attention_mask = attention_mask[:MAX_LENGTH]
        labels = labels[:MAX_LENGTH]
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels
    }


tokenized_id = ds.map(process_func, remove_columns=ds.column_names)

model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto", torch_dtype=torch.bfloat16)
model.enable_input_require_grads()
config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    target_modules=["gate_proj", "up_proj", "down_proj"],
    inference_mode=False,
    r=8,  
    lora_alpha=32, 
    lora_dropout=0.1 
)
model = get_peft_model(model, config)
model.print_trainable_parameters()

args = TrainingArguments(
    output_dir="./new_output/degree_natural/llama3_1_instruct_lora_gud",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    logging_steps=10,
    num_train_epochs=20,
    save_steps=100, 
    learning_rate=1e-5,
    save_on_each_node=True,
    gradient_checkpointing=True
)
trainer = Trainer(
    model=model,
    args=args,
    train_dataset=tokenized_id,
    data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True),
)
trainer.train()
