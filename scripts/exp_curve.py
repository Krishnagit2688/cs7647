#Transformer perplexity - best model
import matplotlib.pyplot as plt
valid_accuracy = [0.3615, 0.4521, 0.5591, 0.6012, 0.6596]
train_accuracy = [0.4815, 0.5012, 0.5891, 0.6321, 0.6897]

valid_loss = [0.9820, 0.9123, 0.8513, 0.7898, 0.7512]
train_loss = [0.9217, 0.8812, 0.8279, 0.7519, 0.7265]

# Overfitting Roberta

#train_accuracy = [0.5349, 0.5941, 0.6935, 0.7930, 0.8656, 0.9032, 0.9247, 0.9543, 0.9677, 0.9946]
#valid_accuracy = [0.4796, 0.5891, 0.6596, 0.6892, 0.7029, 0.7209, 0.7340, 0.7447, 0.6809, 0.6509]


epochs = range(1, len(valid_accuracy) + 1)

plt.plot(epochs, train_loss, marker='o', label='Training Loss')
plt.plot(epochs, valid_loss, marker='o', label='Validation Loss')
# plt.plot(epochs, train_accuracy, marker='o', label='Training Accuracy')
# plt.plot(epochs, valid_accuracy, marker='o', label='Validation Accuracy')

plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Custom Transformer Model - Accuracy')

# Adding a legend
plt.legend()

# Display the plot
plt.show()