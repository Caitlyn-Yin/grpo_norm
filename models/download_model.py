from transformers import AutoTokenizer, AutoModelForCausalLM

model_name = "Qwen/Qwen2.5-Math-1.5B"

# Download the tokenizer
print(f"Downloading tokenizer for {model_name}...")
tokenizer = AutoTokenizer.from_pretrained(model_name)
print("Tokenizer downloaded successfully.")

# Download the model
print(f"Downloading model: {model_name}...")
model = AutoModelForCausalLM.from_pretrained(model_name)
print("Model downloaded successfully.")

# The model and tokenizer are now downloaded and cached.
# By default, they are saved in ~/.cache/huggingface/hub/