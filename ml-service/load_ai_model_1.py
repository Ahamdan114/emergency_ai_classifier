from unsloth import FastLanguageModel

# Load model + adapters directly
model, tokenizer = FastLanguageModel.from_pretrained("PM234/DeepSeek-R1-MedExpert-LoRA-8B-bnb4bit")

# Prep for inference
FastLanguageModel.for_inference(model)

# Example:
test_input = "Below is an instruction...\n### Instruction: Answer the following medical question.\n### Input: What is the primary source of energy for the human body?\n### Response:"
inputs = tokenizer(test_input, return_tensors="pt").to("cuda")
outputs = model.generate(**inputs, max_new_tokens=20)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))  # "Glucose"