import pandas as pd
import torch
from transformers import RobertaTokenizer, RobertaForSequenceClassification, AdamW
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from tqdm import tqdm_notebook
import numpy as np

import torch.nn as nn



df = pd.read_excel("..\\transcripts_text_clean\\consolidated_transcripts_labelled_updated.xlsx")

# Define the RoBERTa tokenizer and model
tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
num_labels = 3  # Number of output labels (Positive, Negative, Neutral)
roberta_model = RobertaForSequenceClassification.from_pretrained("roberta-base", num_labels=num_labels)

for param in roberta_model.parameters():
    param.requires_grad_(True)

# Define the classifier model
input_size = roberta_model.config.hidden_size
hidden_size=roberta_model.config.hidden_size

classifier = nn.Linear(input_size, hidden_size, num_labels)

# Custom Dataset for fine-tuning (remaining code remains the same)
class CustomDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]

        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
            truncation=True,
        )

        input_ids = encoding["input_ids"].squeeze()
        attention_mask = encoding["attention_mask"].squeeze()
        #print(label)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": torch.tensor(label, dtype=torch.int64),
        }

# Split the DataFrame into train and validation sets
train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)

# Create DataLoader for the datasets
max_length = 128
batch_size = 5

train_dataset = CustomDataset(train_df['text'].tolist(), train_df['label'].tolist(), tokenizer, max_length)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

val_dataset = CustomDataset(val_df['text'].tolist(), val_df['label'].tolist(), tokenizer, max_length)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)  # No need to shuffle validation data

# Training loop with AdamW optimizer
num_epochs = 7
learning_rate = 2e-5
criterion = nn.CrossEntropyLoss()
optimizer = AdamW(classifier.parameters(), lr=learning_rate)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
classifier.to(device)

for epoch in range(num_epochs):
    # Training
    roberta_model.train()
    total_train_loss = 0.0
    correct_train_predictions = 0
    total_train_predictions = 0
    progress_bar = tqdm_notebook(train_dataloader, ascii=True)
    # inputs, labels, attention_mask = train_dataloader

    for idx, batch in enumerate(progress_bar):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        outputs = roberta_model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        loss = criterion(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_train_loss += loss.item()
        progress_bar.set_description_str(
            "Batch: %d, Loss: %.4f" % ((idx + 1), loss.item()))

        _, predicted_labels = torch.max(logits, dim=1)
        correct_train_predictions += (predicted_labels == labels).sum().item()
        total_train_predictions += labels.size(0)

        # Evaluate the classifier
        # outputs = roberta_model(inputs, attention_mask=attention_mask)
        predictions = classifier(outputs).argmax(dim=1)
        accuracy = torch.sum(predictions == labels).item() / len(labels)
        print(f"Accuracy Classifier: {accuracy}")

    avg_train_loss = total_train_loss / len(train_dataloader)
    accuracy_train = correct_train_predictions / total_train_predictions
    print("Training Loss: %.4f." % (avg_train_loss))
    print("Training Perplexity: %.4f." % (np.exp(avg_train_loss)))
    print("Training Accuracy: %.4f." % (accuracy_train))



    # Validation
    roberta_model.eval()
    total_val_loss = 0.0
    correct_val_predictions = 0
    total_val_predictions = 0

    with torch.no_grad():
        progress_bar = tqdm_notebook(val_dataloader, ascii=True)

        for idx, batch in enumerate(progress_bar):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = roberta_model(input_ids=input_ids, attention_mask=attention_mask)
            pooled_output = outputs.pooler_output

            logits = classifier(pooled_output)

            loss = criterion(logits, labels)
            # loss = outputs.loss
            # logits = outputs.logits

            # optimizer.zero_grad()
            # loss.backward()
            # optimizer.step()

            total_val_loss += loss.item()
            progress_bar.set_description_str(
                "Batch: %d, Loss: %.4f" % ((idx + 1), loss.item()))

            _, predicted_labels = torch.max(logits, dim=1)
            correct_val_predictions += (predicted_labels == labels).sum().item()
            total_val_predictions += labels.size(0)

        avg_val_loss = total_val_loss / len(val_dataloader)
        accuracy_val = correct_val_predictions / total_val_predictions
        print("Validation Loss: %.4f." % (avg_val_loss))
        print("Validation Perplexity: %.4f." % (np.exp(avg_val_loss)))
        print("Validation Accuracy: %.4f." % (accuracy_val))

roberta_model.save_pretrained("..\\fine_tuned_arch")
tokenizer.save_pretrained("..\\fine_tuned_arch")
print("Fine-tuning complete.")
