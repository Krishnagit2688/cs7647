from transformers import RobertaTokenizer, RobertaForSequenceClassification, XLMRobertaTokenizer, XLMRobertaModel,RobertaConfig
import os
from transformers import pipeline

import pandas as pd

# Load pre-trained RoBERTa model and tokenizer
# model_name = "roberta-base"
print(os.path.exists("..\\fine_tuned"))
tokenizer = RobertaTokenizer.from_pretrained("..\\fine_tuned", local_files_only=True)
model = RobertaForSequenceClassification.from_pretrained("..\\fine_tuned", local_files_only=True)
# model_name = "xlm-roberta-base"
#tokenizer = XLMRobertaModel.from_pretrained(model_name)
#model = XLMRobertaTokenizer.from_pretrained(model_name)


def classify_sentiment(text):
    # Tokenize the text and convert it into model-compatible format
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)

    # Get the model predictions
    outputs = model(**inputs)

    # Convert logits to probabilities using softmax function
    probabilities = outputs.logits.softmax(dim=1).tolist()[0]

    # Define sentiment labels
    sentiment_labels = [0, 1, 2]
    # sentiment_labels = ["Negative", "Neutral", "Positive"]

    # Get the most probable sentiment label
    max_prob_idx = probabilities.index(max(probabilities))
    sentiment = sentiment_labels[max_prob_idx]

    # Return sentiment label and probabilities for all classes
    return sentiment, {label: prob for label, prob in zip(sentiment_labels, probabilities)}


if __name__ == "__main__":
    df = pd.read_excel("..\\transcripts_text_clean\\model_score_input.xlsx")

    for i, row in df.iterrows():
    # Example text (replace this with your quarterly earnings text)
        quarterly_earnings_text = """company has recorded a loss of 5 billion this quarter"""
        # Perform sentiment classification
        sentiment, probabilities = classify_sentiment(row.text)
        print("Sentiment:", sentiment)
        print("Sentiment Probabilities:", probabilities)
