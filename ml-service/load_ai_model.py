import os
os.environ['HF_HOME'] = 'D:\\huggingface_cache'

from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from dotenv import load_dotenv

load_dotenv()

def main():
    model_name = os.getenv("GEMMA_MODEL_NAME")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    ml_model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", torch_dtype="auto")
    
    # print("Model is loading...")
    #To uncomment
    # chat = pipeline("text-generation", model=ml_model, tokenizer=tokenizer, device_map='auto', return_full_text=False)
    # print("Model loaded")

    #To uncomment
    # prompt = "Patient: I feel like my stomach hurts a lot. I have diarrhea\nMedical evaluation: Please asses and make clear the level of urgency:"

    #To uncomment
    # outputs = chat(prompt, max_new_tokens=100, do_sample=True, temperature= 0.3, top_k=50, top_p=0.9, repetition_penalty=1.2)

    # print(outputs[0]["generated_text"])

    return tokenizer, ml_model
