import torch
import pandas as pd
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from sklearn.model_selection import train_test_split
from pso import PSO

def train_and_evaluate_distilbert_model(params, train_dataloader, val_dataloader, num_labels, save_model=Flase):
    learning_rate, dropout_prob, num_epochs = params

    # Load the DistilBERT model and tokenizer
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels)

    # Set up the optimizer and loss function
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    loss_fn = torch.nn.CrossEntropyLoss()

    # Train the model
    for epoch in range(num_epochs):
        for batch in train_dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            logits = outputs.logits
            loss.backward()
            optimizer.step()

        # Evaluate the model on the validation set
        val_loss = 0
        num_val_batches = 0
        num_correct = 0
        num_total = 0
        with torch.no_grad():
            for batch in val_dataloader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)

                outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
                batch_loss = outputs.loss
                batch_logits = outputs.logits
                val_loss += batch_loss.item()
                num_val_batches += 1

                _, predictions = torch.max(batch_logits, dim=1)
                num_correct += (predictions == labels).sum().item()
                num_total += labels.shape[0]

        val_accuracy = num_correct / num_total
        val_loss /= num_val_batches
        print(f"Epoch {epoch+1} | Train Loss: {loss.item():.4f} | Val Loss: {val_loss:.4f} | Val Accuracy: {val_accuracy:.4f}")

    if save_model:
        print("\nSaving the final model")
        model.save_pretrained('distilbert_model')
    return val_accuracy


# Define the text classification dataset
df = pd.read_csv("mtsamples.csv")
df.dropna(inplace=True) # Drop any rows with missing values
texts = df['transcriptions'].tolist() # a list of text samples
labels = df['medical_specialty'].astype('category').cat.codes.tolist() # a list of corresponding labels
num_labels = df['medical_specialty'].nunique()

"""
# Retrieve the medical specialty from an integer code
code = 1  # Example code
medical_specialty = categories.cat.categories[code]
print("Medical specialty:", medical_specialty)
"""

# Split the dataset into training and validation sets
train_texts, val_texts, train_labels, val_labels = train_test_split(texts, labels, test_size=0.2, random_state=42)

# Tokenize the training and validation sets
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
train_encodings = tokenizer(train_texts, truncation=True, padding=True)
val_encodings = tokenizer(val_texts, truncation=True, padding=True)
train_dataset = torch.utils.data.TensorDataset(
    torch.tensor(train_encodings['input_ids']),
    torch.tensor(train_encodings['attention_mask']),
    torch.tensor(train_labels)
)
val_dataset = torch.utils.data.TensorDataset(
    torch.tensor(val_encodings['input_ids']),
    torch.tensor(val_encodings['attention_mask']),
    torch.tensor(val_labels)
)
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=16, shuffle=True)
val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=16, shuffle=False)

# Define the objective function to optimize with PSO
def objective_function(params):
    return -train_and_evaluate_distilbert_model(params, train_dataloader, val_dataloader, num_labels)

# Define the hyperparameter ranges to search over
param_ranges = [
    (1e-5, 1e-3),  # learning rate
    (0.1, 0.5),    # dropout probability
    (2, 5)         # number of epochs
]

# Set up the PSO algorithm with the objective function and hyperparameter ranges
pso = PSO(objective_function, param_ranges, num_particles=10, max_iter=50)

# Run the PSO algorithm to find the optimal hyperparameters
best_params, best_score = pso.run()

# Train the DistilBERT model with the optimal hyperparameters on the entire dataset
train_and_evaluate_distilbert_model(params, train_dataloader, val_dataloader, num_labels, save_model=True)
