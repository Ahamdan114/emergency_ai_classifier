from load_ai_model import main
from transformers import pipeline
import sys
import json

if __name__ == "__main__":
    tokenizer, ml_model = main()

    # print("Model is loading...")
    chat = pipeline("text-generation", model=ml_model, tokenizer=tokenizer, device_map='auto', return_full_text=False)
    # print("Model loaded")

    prompt = prompt = f"""
        You are a medical AI assistant. Based on the patient's statement below, return a JSON object with the following keys:
        - "urgency": either 0 (not urgent), 1 (moderately urgent), or 2 (very urgent)
        - "explanation": a brief reason for your classification

        Rules:
        - Do not include any text outside the JSON
        - No markdown, no explanations, no greetings
        - No formatting or tags
        - Just pure JSON like this:

        Respond ONLY with valid JSON in this format:
        {{
        "urgency": <number>,
        "explanation": "<your explanation>"
        }}

        Patient statement: "{sys.argv[1]}"

        Respond in **valid JSON only**.
    """

    
    # f"Patient: {sys.argv[1]} \nMedical evaluation: Please asses and make clear the level of urgency:"

    outputs = chat(prompt, max_new_tokens=150, do_sample=True, temperature=0.5, top_k=50, top_p=0.9, repetition_penalty=1.5)
    text = outputs[0]["generated_text"]
    # print(text)

    # sys.stdout.flush(text)
    print(json.dumps({"generated_text": text}))

# if __name__ == "__main__":
#     tokenizer, ml_model = main()

#     print("Model is loading...")
#     chat = pipeline("text-generation", model=ml_model, tokenizer=tokenizer, device_map='auto', return_full_text=False)
#     print("Model loaded")

#     prompt = "Patient: I am feeling great \nMedical evaluation: Please asses and make clear the level of urgency:"

#     outputs = chat(prompt, max_new_tokens=100, do_sample=True, temperature=0.5, top_k=50, top_p=0.9, repetition_penalty=1.5)

#     print(outputs[0]["generated_text"])