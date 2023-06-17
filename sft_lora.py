import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
from datasets import load_dataset
import transformers
from transformers import Trainer, TrainingArguments
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import BitsAndBytesConfig
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    set_peft_model_state_dict,
)
import torch


### 定义一些配置信息
CUTOFF_LEN = 1024  
VAL_SET_SIZE = 2000
DATA_PATH = "./dataset/Belle_open_source_0.5M.json" 
OUTPUT_DIR = "baichuansft"
resume_from_checkpoint = "baichuansft"


device_map = {"": 0}
tokenizer = AutoTokenizer.from_pretrained("./baichuan-7B",trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained("./baichuan-7B",
                                             trust_remote_code=True,
                                             quantization_config=BitsAndBytesConfig(
                                                 load_in_4bit=True,
                                                 bnb_4bit_compute_dtype=torch.bfloat16,
                                                 bnb_4bit_use_double_quant=True,
                                                 bnb_4bit_quant_type='nf4'
                                             ),
                                             device_map=device_map)

model = prepare_model_for_kbit_training(model)

### 所有的线性layer都装配上lora
import bitsandbytes as bnb
def find_all_linear_names(model):
    #cls = bnb.nn.Linear8bitLt 
    cls = bnb.nn.Linear4bit 
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])


    if 'lm_head' in lora_module_names: # needed for 16-bit
        lora_module_names.remove('lm_head')
    return list(lora_module_names)
modules = find_all_linear_names(model)


config = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    bias="none",
    target_modules=modules,
    task_type="CAUSAL_LM",
)


model = get_peft_model(model, config)
tokenizer.pad_token_id = 0

if resume_from_checkpoint:
    # Check the available weights and load them
    checkpoint_name = os.path.join(
        resume_from_checkpoint, "pytorch_model.bin"
    )  # Full checkpoint
    if not os.path.exists(checkpoint_name):
        checkpoint_name = os.path.join(
            resume_from_checkpoint, "adapter_model.bin"
        )  # only LoRA model - LoRA config above has to fit
        resume_from_checkpoint = (
            False  # So the trainer won't try loading its state
        )
    # The two files above have a different name depending on how they were saved, but are actually the same.
    if os.path.exists(checkpoint_name):
        print(f"Restarting from {checkpoint_name}")
        adapters_weights = torch.load(checkpoint_name)
        set_peft_model_state_dict(model, adapters_weights)
    else:
        print(f"Checkpoint {checkpoint_name} not found")


data = load_dataset("json", data_files=DATA_PATH)

def tokenize(prompt, add_eos_token=True):
    result = tokenizer(
        prompt,
        truncation=True,
        max_length=CUTOFF_LEN,
        padding=False,
        return_tensors=None,
    )
    if (
            result["input_ids"][-1] != tokenizer.eos_token_id
            and len(result["input_ids"]) < CUTOFF_LEN
            and add_eos_token
    ):
        result["input_ids"].append(tokenizer.eos_token_id)
        result["attention_mask"].append(1)

    if add_eos_token and len(result["input_ids"]) >= CUTOFF_LEN:
        result["input_ids"][CUTOFF_LEN - 1] = tokenizer.eos_token_id
        result["attention_mask"][CUTOFF_LEN - 1] = 1

    result["labels"] = result["input_ids"].copy()

    return result


def generate_and_tokenize_prompt(data_point):
    instruction = data_point['instruction']
    input_text = data_point["input"]
    input_text = "Human: " + instruction + input_text + "\n\nAssistant: "
    input_text = tokenizer.bos_token + input_text if tokenizer.bos_token != None else input_text
    target_text = data_point["output"] + tokenizer.eos_token
    full_prompt = input_text + target_text
    tokenized_full_prompt = tokenize(full_prompt)
    return tokenized_full_prompt


if VAL_SET_SIZE > 0:
    train_val = data["train"].train_test_split(
        test_size=VAL_SET_SIZE, shuffle=True, seed=42
    )
    train_data = train_val["train"].shuffle().map(generate_and_tokenize_prompt)
    val_data = train_val["test"].shuffle().map(generate_and_tokenize_prompt)
else:
    train_data = data['train'].shuffle().map(generate_and_tokenize_prompt)
    val_data = None

trainer = Trainer(
    model=model,
    train_dataset=train_data,
    eval_dataset=val_data,
    args=TrainingArguments(
        num_train_epochs=1,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        learning_rate=3e-4,
        gradient_accumulation_steps=4,
        evaluation_strategy="steps" if VAL_SET_SIZE > 0 else "no",
        save_strategy="steps",
        eval_steps=2000 if VAL_SET_SIZE > 0 else None,
        save_steps=2000,
        output_dir=OUTPUT_DIR,
        report_to = "tensorboard",
        save_total_limit=3,
        load_best_model_at_end=True if VAL_SET_SIZE > 0 else False,
        optim="adamw_torch"
    ),
    data_collator=transformers.DataCollatorForSeq2Seq(tokenizer,
                                                      pad_to_multiple_of=8,
                                                      return_tensors="pt",
                                                      padding=True),
)


trainer.train(resume_from_checkpoint=False)
model.save_pretrained(OUTPUT_DIR)
