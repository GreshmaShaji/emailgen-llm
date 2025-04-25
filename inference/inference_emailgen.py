from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import torch
from transformers import BitsAndBytesConfig

# === PATHS ===
base_model = "mistralai/Mistral-7B-Instruct-v0.1"
adapter_model = "emailgen-llm/model/emailgen-qlora"

# === LOAD TOKENIZER & BASE MODEL ===
print("🔄 Loading base model...")
tokenizer = AutoTokenizer.from_pretrained(base_model)

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16
)


model = AutoModelForCausalLM.from_pretrained(
    base_model,
    device_map="auto",
    torch_dtype=torch.float16,
    quantization_config=bnb_config,

)


# === LOAD LoRA ADAPTER ===
print("🔗 Loading LoRA fine-tuned weights...")
model = PeftModel.from_pretrained(model, adapter_model)

# === INPUT PROMPT ===
prompt = """Category: job_application
Context: Greshma is applying for a Data Scientist role at BMW, focusing on machine learning and automotive.
Write the email:
"""

inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

# === GENERATE OUTPUT ===
print("🚀 Generating email...")
outputs = model.generate(
    **inputs,
    max_new_tokens=300,
    temperature=0.7,
    top_p=0.9,
    do_sample=True
)

# === PRINT RESULT ===
print("\n📬 Generated Email:\n")
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
