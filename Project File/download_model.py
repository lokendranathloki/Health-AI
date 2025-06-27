from transformers import AutoTokenizer, AutoModelForCausalLM

model_name = "ibm-granite/granite-3.3-2b-instruct"
save_dir = "./granite-3.3-2b-instruct"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer.save_pretrained(save_dir)
model.save_pretrained(save_dir)
print("Model downloaded and saved locally.") 