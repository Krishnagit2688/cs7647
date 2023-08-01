import torch.nn as nn

class SentimentClassifier(nn.Module):
    def __init__(self, hidden_size, num_classes):
        super(SentimentClassifier, self).__init__()
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        return self.fc(x)

import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Assuming you have already preprocessed your data and have the input tensors
# train_inputs, train_masks, train_labels, val_inputs, val_masks, val_labels

# Define the classifier model
classifier = SentimentClassifier(hidden_size=roberta_model.config.hidden_size, num_classes=num_classes)

# Combine RoBERTa with the classifier
model = nn.Sequential(roberta_model, classifier)

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Train the model
num_epochs = 5
batch_size = 32

for epoch in range(num_epochs):
    model.train()
    for i in range(0, len(train_inputs), batch_size):
        inputs_batch = train_inputs[i:i+batch_size]
        masks_batch = train_masks[i:i+batch_size]
        labels_batch = train_labels[i:i+batch_size]

        optimizer.zero_grad()

        outputs = model(inputs_batch, attention_mask=masks_batch)
        loss = criterion(outputs, labels_batch)
        loss.backward()
        optimizer.step()

    # Validate the model
    model.eval()
    with torch.no_grad():
        val_loss = 0.0
        correct_preds = 0
        total_preds = 0

        for i in range(0, len(val_inputs), batch_size):
            inputs_batch = val_inputs[i:i+batch_size]
            masks_batch = val_masks[i:i+batch_size]
            labels_batch = val_labels[i:i+batch_size]

            outputs = model(inputs_batch, attention_mask=masks_batch)
            loss = criterion(outputs, labels_batch)
            val_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            correct_preds += (predicted == labels_batch).sum().item()
            total_preds += labels_batch.size(0)

        val_accuracy = correct_preds / total_preds
        val_loss /= len(val_inputs) / batch_size

    print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {val_loss:.4f}, Accuracy: {val_accuracy:.4f}")
