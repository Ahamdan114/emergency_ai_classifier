from load_ai_model import main
from transformers import pipeline
import sys
import json

from test_bart_mnli_model import generate_urgency_level

if __name__ == "__main__":
    print("Loading the model ...load_ai_model...")
    tokenizer, ml_model = main()

    print("Loading the model ...test_bart_mnli_model...")
    urgency_level_message, confidence = generate_urgency_level(
        'hi hello I feel like my stomach hurts I ate a lot of Ruth before going to bed and I am feeling nauseous can you please help'
        ) # (prompt=sys.argv[1])
    chat = pipeline("text-generation", model=ml_model, tokenizer=tokenizer, device_map='auto', return_full_text=False)
    print("Both models loaded")

    # - "urgency": %{confidence} either 0 (not urgent), 1 (moderately urgent), or 2 (very urgent)
    prompt = f"""
        You are a medical AI assistant. Based on the patient's statement below, return a JSON object with the following keys:
        - "urgency": {confidence:.2f}%
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
        Your response should have as assumption this urgency level: "{urgency_level_message}" with confidence of {confidence:.2f}%.
        Patient statement: "{sys.argv[1]}"

        Respond in **valid JSON only**.
    """

    
    # f"Patient: {sys.argv[1]} \nMedical evaluation: Please asses and make clear the level of urgency:"
    print(f"AI got prompt: {prompt}")
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