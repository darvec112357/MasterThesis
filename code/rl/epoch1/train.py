from datasets import load_dataset,Dataset
from unsloth import FastLanguageModel
import torch
import json,os,sys
import random

max_seq_length = 1024 # Choose any! We auto support RoPE Scaling internally!
dtype = None # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
load_in_4bit = False # Use 4bit quantization to reduce memory usage. Can be False.
model_map = {'llama':'llama-3-8b-Instruct',
             'gemma7':'gemma-7b-it'}
MODEL_NAME = "unsloth/" + model_map[sys.argv[1]]
cut_ratio = float(sys.argv[2])
OUTPUT_DIR = sys.argv[3]

import wandb 
## wandb variables
wandb.login(relogin=True, key='422f5b2b051bce299f9723bb50b4e677b8ffc814')
os.environ["WANDB_PROJECT"] = "gemma-2b-it-lora-rl"
# os.environ["NCCL_P2P_DISABLE"] = "1"
# os.environ["NCCL_IB_DISABLE"] = "1"

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = MODEL_NAME,
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
    # token = "hf_...", # use one if using gated models like meta-llama/Llama-2-7b-hf
)

model = FastLanguageModel.get_peft_model(
    model,
    r = 16, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj"],
    lora_alpha = 32,
    lora_dropout = 0, # Supports any, but = 0 is optimized
    bias = "none",    # Supports any, but = "none" is optimized
    # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
    use_gradient_checkpointing = "unsloth", # True or "unsloth" for very long context
    use_rslora = False,  # We support rank stabilized LoRA
    loftq_config = None, # And LoftQ
)

with open('data/train.json', 'r') as json_file:
    dataset_train = json.load(json_file)
    dataset_train['text'] = random.sample(dataset_train['text'],int(len(dataset_train['text']) * cut_ratio))
    dataset_train = Dataset.from_dict(dataset_train)
    
with open('data/val.json', 'r') as json_file:
    dataset_val = json.load(json_file)
    dataset_val['text'] = random.sample(dataset_val['text'],int(len(dataset_val['text']) * cut_ratio))
    dataset_val = Dataset.from_dict(dataset_val)

print(dataset_train)
print(dataset_val)
print(dataset_train['text'][0])
from trl import SFTTrainer
from transformers import TrainingArguments
from unsloth import is_bfloat16_supported

steps = 2000
trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset_train,
    eval_dataset = dataset_val,
    dataset_text_field = "text",
    max_seq_length = max_seq_length,
    # dataset_num_proc = 2,
    packing = False, # Can make training 5x faster for short sequences.
    args = TrainingArguments(
        eval_strategy = "steps",
        eval_steps = steps,
        per_device_eval_batch_size = 4,
        per_device_train_batch_size = 4,
        gradient_accumulation_steps = 4,
        warmup_steps = 5,
        learning_rate = 2e-4,
        logging_steps = steps,
        save_steps = steps,
        save_total_limit = 3,
        num_train_epochs = 1,
        fp16 = not is_bfloat16_supported(),
        bf16 = is_bfloat16_supported(),
        # optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        output_dir = OUTPUT_DIR,
        load_best_model_at_end = True
    )
)

trainer_stats = trainer.train()
model.save_pretrained(OUTPUT_DIR)
# wandb.finish()
