import pandas as pd
from transformers import BertTokenizer, BertForSequenceClassification
import torch

# Load pre-trained BERT model and tokenizer
model_name = 'nlptown/bert-base-multilingual-uncased-sentiment'
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name)

# Function to get sentiment labels from the sentiment scores
def get_sentiment_label(score):
    if score >= 0.05:
        return 'Positive'
    elif score <= -0.05:
        return 'Negative'
    else:
        return 'Neutral'

def label(company):
    df = pd.read_excel("..\\transcripts_text_clean\\"+company+"_clean_final_unlabelled.xlsx")
    processed_rows = []

    # Perform sentiment analysis on the earnings call text
    # sentiments = []

    for index, row in df.iterrows():
        text = row['text']
        print(row["transcripts"], row["id"])
        inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True)
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            sentiment_score = logits[0, 0].item()
            sentiment_label = get_sentiment_label(sentiment_score)
            row['sentiments'] = sentiment_label
            row['company'] = company
            processed_rows.append(row)

    # Add the sentiment labels to the DataFrame
    df_processed = pd.DataFrame(processed_rows)

    df_processed.to_excel("..\\transcripts_text_clean\\"+company+"_clean_final_labelled.xlsx", index=False)

label("infosys")