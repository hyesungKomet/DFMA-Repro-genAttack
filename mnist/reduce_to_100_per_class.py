import torch

# Load the full generated dataset
data = torch.load("generated_data.pt")
labels = torch.load("generated_label.pt")

print("Loaded:")
print(" data:", data.shape)
print(" labels:", labels.shape)

# Dict to track samples per class
count = {i: 0 for i in range(10)}

selected_data = []
selected_labels = []

for i in range(data.shape[0]):
    label = torch.argmax(labels[i]).item()

    if count[label] < 100:
        selected_data.append(data[i])
        selected_labels.append(labels[i])
        count[label] += 1

    if all(v == 100 for v in count.values()):
        break

print("Final counts:", count)

selected_data = torch.stack(selected_data)
selected_labels = torch.stack(selected_labels)

torch.save(selected_data, "generated_data_100.pt")
torch.save(selected_labels, "generated_label_100.pt")

print("Saved: generated_data_100.pt and generated_label_100.pt")
