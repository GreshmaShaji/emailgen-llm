import gradio as gr
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import torch
from transformers import BitsAndBytesConfig

base_model = "mistralai/Mistral-7B-Instruct-v0.1"
adapter_model = "emailgen-llm/model/emailgen-qlora"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16
)

tokenizer = AutoTokenizer.from_pretrained(base_model)
model = AutoModelForCausalLM.from_pretrained(
    base_model,
    device_map="auto",
    quantization_config=bnb_config,
    torch_dtype=torch.float16
)
model = PeftModel.from_pretrained(model, adapter_model)



def generate_email(category, context):
    prompt = f"""Category: {category}\nContext: {context}\nWrite the email:\n"""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=300, temperature=0.7, top_p=0.9, do_sample=True)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

demo = gr.Interface(
    fn=generate_email,
    inputs=[
        gr.Textbox(label="Category", value="job_application"),
        gr.Textbox(label="Context", lines=4, value="Greshma is applying for a Data Scientist role at BMW."),
    ],
    outputs="text",
    title="📧 EmailGen - LLM Email Generator",
    description="Fine-tuned with ❤️ by Greshma Shaji"
)

demo.launch(share=True)
