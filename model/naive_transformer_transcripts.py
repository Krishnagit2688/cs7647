import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from torchtext.data import Field, TabularDataset, BucketIterator
from torchtext.vocab import GloVe
import torchtext
from tqdm import tqdm
import torch.nn.functional as F
import math


class TransformerModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_classes, num_heads, num_encoder_layers, dropout):
        super(TransformerModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=embedding_dim, nhead=num_heads, dim_feedforward=hidden_dim,
                                                   dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_encoder_layers)
        self.self_attention = nn.MultiheadAttention(embed_dim=embedding_dim, num_heads=num_heads)
        self.fc = nn.Linear(embedding_dim, num_classes)

    def forward(self, x):
        x = self.embedding(x)
        x = x.permute(1, 0, 2)  # (seq_len, batch_size, embedding_dim)
        # self_att_output, _ = self.self_attention(x, x, x)
        x = self.transformer_encoder(x)
        # x = self.transformer_encoder(self_att_output)
        x = x.mean(dim=0)  # Average pooling over the sequence length
        x = self.fc(x)
        return F.softmax(x, dim=1)

def preprocess_data(df):
    TEXT = Field(sequential=True, lower=True, include_lengths=True, batch_first=True)
    LABEL = Field(sequential=False, use_vocab=False)

    fields = [('text', TEXT), ('label', LABEL)]

    examples = [torchtext.data.Example.fromlist([text, label], fields) for text, label in zip(df['text'], df['label'])]
    dataset = torchtext.data.Dataset(examples, fields)

    TEXT.build_vocab(dataset, vectors=GloVe(name='6B', dim=100))  # You can choose a different GloVe dimension if you prefer

    return dataset, TEXT

def train_model(model, train_iterator, criterion, optimizer, num_epochs, epoch):
    model.train()

    # for epoch in range(num_epochs):
    total_loss = 0
    total_samples = 0
    correct = 0
    total = 0

    for batch in tqdm(train_iterator, desc=f"Epoch {epoch + 1}/{num_epochs}"):
        text, text_lengths = batch.text
        labels = batch.label

        optimizer.zero_grad()
        outputs = model(text)
        # print(labels)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * text.size(0)
        total_samples += text.size(0)

        # Calculate accuracy
        predicted_labels = torch.argmax(outputs, dim=1)
        # print(predicted_labels)
        correct += (predicted_labels == batch.label).sum().item()
        total += batch.label.size(0)

    epoch_loss = total_loss / total_samples
    accuracy = correct / total
    print(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {epoch_loss:.4f}")
    print(f"Epoch {epoch + 1}/{num_epochs}, Train Accuracy: {accuracy:.4f}")
    print(f"Epoch {epoch + 1}/{num_epochs}, Train PPL: {math.exp(epoch_loss):.4f}")


def val_model(model, val_iterator, criterion, num_epochs, epoch):
    model.eval()

    # for epoch in range(num_epochs):
    total_loss = 0
    total_samples = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in tqdm(val_iterator, desc=f"Epoch {epoch + 1}/{num_epochs}"):
            text, text_lengths = batch.text
            labels = batch.label
            outputs = model(text)
            # print(labels)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * text.size(0)
            total_samples += text.size(0)

            # Calculate accuracy
            predicted_labels = torch.argmax(outputs, dim=1)
            # print(predicted_labels)
            correct += (predicted_labels == batch.label).sum().item()
            total += batch.label.size(0)

    epoch_loss = total_loss / total_samples
    accuracy = correct / total
    print(f"Epoch {epoch + 1}/{num_epochs}, Val Loss: {epoch_loss:.4f}")
    print(f"Epoch {epoch + 1}/{num_epochs}, Val Accuracy: {accuracy:.4f}")
    print(f"Epoch {epoch + 1}/{num_epochs}, Val PPL: {math.exp(epoch_loss):.4f}")

if __name__ == "__main__":
    from sklearn.model_selection import train_test_split
    df = pd.read_excel("..\\transcripts_text_clean\\consolidated_transcripts_labelled_updated.xlsx")

    X = df['text']
    y = df['label']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    train_data = pd.concat([X_train, y_train], axis=1)
    test_data = pd.concat([X_test, y_test], axis=1)

    # Assuming your DataFrame is named 'df'
    train_dataset, train_TEXT = preprocess_data(train_data)
    val_dataset, val_TEXT = preprocess_data(test_data)

    vocab_size = len(train_TEXT.vocab)
    print(vocab_size)
    embedding_dim = 100
    hidden_dim = 256
    num_classes = 3
    num_heads = 4
    num_encoder_layers = 2
    dropout = 0.2

    model = TransformerModel(vocab_size, embedding_dim, hidden_dim, num_classes, num_heads, num_encoder_layers, dropout)

    optimizer = optim.Adam(model.parameters(), lr=1e-5)
    criterion = nn.CrossEntropyLoss()

    train_iterator = BucketIterator(train_dataset, batch_size=2, sort_key=lambda x: len(x.text), sort_within_batch=True)

    #train_model(model, train_iterator, criterion, optimizer, num_epochs)

    val_iterator = BucketIterator(val_dataset, batch_size=2, sort_key=lambda x: len(x.text), sort_within_batch=True)
    num_epochs = 10

    #val_model(model, val_iterator, criterion, optimizer, num_epochs)

    for epoch in range(num_epochs):
        train_model(model, train_iterator, criterion, optimizer, num_epochs, epoch)
        val_model(model, val_iterator, criterion, num_epochs, epoch)

    torch.save(model.state_dict(), "..\\naive_model\\trained_model.pth")

