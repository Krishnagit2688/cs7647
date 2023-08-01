from transformers import RobertaTokenizer
import torch
import pandas as pd

# Load pre-trained RoBERTa tokenizer
tokenizer = RobertaTokenizer.from_pretrained("roberta-base")

# Example data (replace these with your actual data)
# transcripts = [
#     "The company reported strong earnings and revenue growth.",
#     "The economic downturn negatively impacted our profits.",
#     "We experienced steady growth in the last quarter.",
#     # Add more transcripts here...
# ]
#
# # Corresponding sentiment labels (0 for negative, 1 for neutral, 2 for positive)
# sentiment_labels = [2, 0, 2, 1, 2, 1, 0, 2, 1, 0, 1, 2, 2, 0, 0, 1, 1, 2, 2, 2]

df = pd.read_excel("..\\transcripts_text_clean\\consolidated_transcripts_labelled_updated.xlsx")
transcripts = list(df.text)

# Tokenize and encode the transcripts
inputs = tokenizer(
    transcripts,
    truncation=True,
    padding=True,
    return_tensors="pt",  # Return PyTorch tensors
)

# Convert sentiment labels to PyTorch tensors
labels = torch.tensor(list(df.label), dtype=torch.long)

# Split the dataset into training and validation sets (80% for training, 20% for validation)
split_ratio = 0.8
train_size = int(split_ratio * len(transcripts))

train_inputs = {key: value[:train_size] for key, value in inputs.items()}
train_labels = labels[:train_size]

val_inputs = {key: value[train_size:] for key, value in inputs.items()}
val_labels = labels[train_size:]

# Print the shapes of the training and validation sets
print("Training set shape:", train_inputs["input_ids"].shape, train_labels.shape)
print("Validation set shape:", val_inputs["input_ids"].shape, val_labels.shape)
