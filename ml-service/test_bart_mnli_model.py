from transformers import pipeline
import os

def generate_urgency_level(prompt = 'Default prompt given to the model'):
    if prompt == 'Default prompt given to the model':
        print("No prompt provided, using default prompt.")
        return "No prompt provided", 0.0
    
    model_name = os.getenv("BART_MODEL_NAME")
    chat_classifier = pipeline("zero-shot-classification", model=model_name)
    # model_name = "facebook/bart-large-mnli"
    # cache_placeholder = f"C:\\Users\\Ahmed\\.cache\\huggingface\\hub\\models--{model_name.replace('/','-')}"
    # classifier = pipeline("zero-shot-classification", model=cache_placeholder)
    # sequence_to_classify = "Hi, I feel like my stomach hurts. I ate a lot of fruits before going to bed and now I'm feeling nauseous." # prompt
    sequence_to_classify = prompt
    candidate_labels = [
        "Urgent: You should go to the hospital immediately",
        "Somewhat Urgent: You should see a doctor within a week",
        "Non-Urgent: You can wait for a few days and see how you feel"
    ]

    result = chat_classifier(sequence_to_classify, candidate_labels)
    print("This is the result:")
    print(result)
    return result["labels"][0], result["scores"][0] * 100