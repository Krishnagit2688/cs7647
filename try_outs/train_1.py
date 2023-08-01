import pandas as pd
import torch
from transformers import RobertaTokenizer, RobertaForSequenceClassification, AdamW
from torch.utils.data import DataLoader, Dataset

df = pd.read_excel("..\\transcripts_text_clean\\consolidated_transcripts_labelled.xlsx")
print(df)

tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
num_labels = 3  # Number of output labels (Positive, Negative, Neutral)
model = RobertaForSequenceClassification.from_pretrained("roberta-base", num_labels=num_labels)

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
        print(label)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": torch.tensor(label, dtype=torch.long),
        }

# Map labels to numerical values (Positive: 0, Negative: 1, Neutral: 2)
#label_map = {'Negative': 0, 'Neutral': 1, 'Positive': 2}
#df['label'] = df['label'].map(label_map)

#default_label = 'Unknown'
#df['label'] = df['label'].map(lambda label: label_map.get(label, label_map[default_label]))

# Create DataLoader for the dataset
max_length = 128  # You can adjust this value based on your input length requirements
batch_size = 2
dataset = CustomDataset(list(df.text), list(df.label), tokenizer, max_length)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

num_epochs = 3
optimizer = AdamW(model.parameters(), lr=2e-5)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

for epoch in range(num_epochs):
    model.train()
    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}")

print("Fine-tuning complete.")
