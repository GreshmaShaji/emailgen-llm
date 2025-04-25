from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import get_peft_model, LoraConfig, TaskType
from transformers import TrainingArguments, Trainer, DataCollatorForLanguageModeling

dataset = load_dataset("json", data_files="emailgen-llm/data/prompt_completion_pairs.jsonl")
dataset = dataset["train"].train_test_split(test_size=0.1)

model_name = "mistralai/Mistral-7B-Instruct-v0.1"

bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_use_double_quant=True,
                                bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype="float16")

model = AutoModelForCausalLM.from_pretrained(model_name,quantization_config=bnb_config, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

peft_config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.1,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
)

model = get_peft_model(model, peft_config)
model.print_trainable_parameters()

training_args = TrainingArguments(
    output_dir="emailgen-llm/model/emailgen-qlora",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,
    num_train_epochs=3,
    learning_rate=2e-4,
    logging_dir="./logs",
    save_total_limit=2,
    fp16=True,
    report_to="none",
    remove_unused_columns=False,
)

data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

def tokenize(example):
    full_text = example["prompt"] + example["completion"]
    return tokenizer(full_text, truncation=True, padding="max_length", max_length=512)

tokenized_dataset = dataset.map(tokenize, batched=False)

tokenized_dataset = tokenized_dataset.remove_columns(["prompt", "completion"])


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["test"],
    tokenizer=tokenizer,
    data_collator=data_collator
)

trainer.train()
model.save_pretrained("emailgen-llm/model/emailgen-qlora")
tokenizer.save_pretrained("emailgen-llm/model/emailgen-qlora")