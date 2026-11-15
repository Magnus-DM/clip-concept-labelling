import re

import matplotlib.pyplot as plt

# Initialize lists to store metrics
training_loss = []
training_accuracy = []
training_accuracy_0 = []
training_accuracy_1 = []
training_f1_score = []
validation_loss = []
validation_accuracy = []
validation_accuracy_0 = []
validation_accuracy_1 = []
validation_f1_score = []

# Read the content of the temp.txt file
with open('output.txt', 'r') as file:
    lines = file.readlines()

# Regular expression to extract metrics
pattern = re.compile(r"Epoch \[\d+/\d+\], (Training|Validation) Loss: ([\d.]+), (Training|Validation) Accuracy: ([\d.]+), (Training|Validation) Accuracy \(0\): ([\d.]+), (Training|Validation) Accuracy \(1\): ([\d.]+), (Training|Validation) F1 Score: ([\d.]+)")

# Extract metrics from each line
for line in lines:
    match = pattern.match(line)
    if match:
        if match.group(1) == 'Training':
            training_loss.append(float(match.group(2)))
            training_accuracy.append(float(match.group(4)))
            training_accuracy_0.append(float(match.group(6)))
            training_accuracy_1.append(float(match.group(8)))
            training_f1_score.append(float(match.group(10)))
        elif match.group(1) == 'Validation':
            validation_loss.append(float(match.group(2)))
            validation_accuracy.append(float(match.group(4)))
            validation_accuracy_0.append(float(match.group(6)))
            validation_accuracy_1.append(float(match.group(8)))
            validation_f1_score.append(float(match.group(10)))

# Ensure the epochs array matches the length of the metric lists
epochs = range(1, len(training_loss) + 1)

plt.figure(figsize=(15, 10))

plt.subplot(2, 2, 1)
plt.plot(epochs, training_loss, label='Training Loss')
plt.plot(epochs, validation_loss, label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.subplot(2, 2, 2)
plt.plot(epochs, training_accuracy, label='Training Accuracy')
plt.plot(epochs, validation_accuracy, label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(2, 2, 3)
plt.plot(epochs, training_accuracy_0, label='Training Accuracy (0)')
plt.plot(epochs, validation_accuracy_0, label='Validation Accuracy (0)')
plt.xlabel('Epochs')
plt.ylabel('Accuracy (0)')
plt.legend()

plt.subplot(2, 2, 4)
plt.plot(epochs, training_accuracy_1, label='Training Accuracy (1)')
plt.plot(epochs, validation_accuracy_1, label='Validation Accuracy (1)')
plt.xlabel('Epochs')
plt.ylabel('Accuracy (1)')
plt.legend()

plt.figure(figsize=(7, 5))
plt.plot(epochs, training_f1_score, label='Training F1 Score')
plt.plot(epochs, validation_f1_score, label='Validation F1 Score')
plt.xlabel('Epochs')
plt.ylabel('F1 Score')
plt.legend()

plt.tight_layout()
plt.show()