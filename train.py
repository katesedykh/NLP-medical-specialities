import torch
import numpy as np
import pandas as pd
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, DistilBertConfig
from sklearn.model_selection import train_test_split
from pso import PSO


def eval_model(model, val_dataloader):
    model.eval()
        with torch.no_grad():
            for batch in val_dataloader:
                batch = {k: v.to(device) for k, v in zip(['input_ids', 'attention_mask', 'labels'],batch)}

                outputs = model(**batch)
                batch_loss = outputs.loss
                batch_logits = outputs.logits
                val_loss += batch_loss.item()
                num_val_batches += 1

                _, predictions = torch.max(batch_logits, dim=1)
                num_correct += (predictions == batch["labels"]).sum().item()
                num_total += batch["labels"].shape[0]

        val_accuracy = num_correct / num_total
        val_loss /= num_val_batches
    return val_accuracy, val_loss


def train_and_evaluate_distilbert_model(params, train_dataloader, val_dataloader, num_labels, test_dataloader=None, save_model=False):
    '''Train and evaluate a DistilBERT model with the given hyperparameters.'''
    learning_rate, dropout_prob, num_epochs = params

    # Load the DistilBERT model
    config = DistilBertConfig(dropout=dropout_prob, attention_dropout=dropout_prob, seq_classif_dropout=dropout_prob*1.5)
    model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=num_labels, dropout=dropout_prob, attention_dropout=dropout_prob, seq_classif_dropout=dropout_prob*1.5)
    model.to(device)
    # Train the model
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    model.train()
    for epoch in range(int(num_epochs)):
        for batch in train_dataloader:
            batch = {k: v.to(device) for k, v in zip(['input_ids', 'attention_mask', 'labels'],batch)}
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
            optimizer.step()

        # Evaluate the model on the validation set each epoch
        val_loss = 0
        num_val_batches = 0
        num_correct = 0
        num_total = 0
        val_accuracy, val_loss = eval_model(model, val_dataloader)
        print(f"Epoch {epoch+1} | Train Loss: {loss.item():.4f} | Val Loss: {val_loss:.4f} | Val Accuracy: {val_accuracy:.4f}")

    if save_model:
        if test_dataloader is not None:
            print("\nEvaluating the final model on a test set")
            test_accuracy, _ = eval_model(model, test_dataloader)
            print(f"Accuracy on the test set: {test_accuracy:.4f}")
        print("\nSaving the final model")
        model.save_pretrained("distilbert_model_best_params")
    return val_accuracy


def objective_function(params):
    '''Objective function for the PSO algorithm.'''
    return -train_and_evaluate_distilbert_model(params, train_dataloader, val_dataloader, num_labels)



device = torch.device("cuda")

# Define the text classification dataset
df = pd.read_csv("mtsamples.csv")
df.dropna(inplace=True) # Drop any rows with missing values
texts = df['transcription'].tolist() # a list of text samples
labels = df['medical_specialty'].astype('category').cat.codes.tolist() # a list of corresponding labels
num_labels = df['medical_specialty'].nunique()

"""
# Retrieve the medical specialty from an integer code
code = 1  # Example code
medical_specialty = categories.cat.categories[code]
print("Medical specialty:", medical_specialty)
"""

# Split the dataset into training, validation and test sets
train_texts, val_texts, train_labels, val_labels = train_test_split(texts, labels, test_size=.3)
val_texts, test_texts, val_labels, test_labels = train_test_split(val_texts, val_labels, test_size=.3)

# Tokenize the training and validation sets
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
train_encodings = tokenizer(train_texts, truncation=True, padding=True)
val_encodings = tokenizer(val_texts, truncation=True, padding=True)
test_encodings = tokenizer(test_texts, truncation=True, padding=True)

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
test_dataset = torch.utils.data.TensorDataset(
    torch.tensor(test_encodings['input_ids']),
    torch.tensor(test_encodings['attention_mask']),
    torch.tensor(test_labels)
)
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=64, shuffle=False)
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False)


# Define the hyperparameter ranges to search over
param_ranges = np.array([
    [1e-6, 1e-3],  # learning rate
    [0, 0.3],    # dropout probability
    [5, 30]         # number of epochs
])

# Set up the PSO algorithm with the objective function and hyperparameter ranges
pso = PSO(objective_function, param_ranges, num_particles=2, max_iter=5)

# Run the PSO algorithm to find the optimal hyperparameters
best_params, best_score = pso.run()

# Train the DistilBERT model with the optimal hyperparameters on the entire dataset
print(f"Best params: \nLearning rate {best_params[0]} \nDropout probability {best_params[1]} \nNum epochs {best_params[2]}")
print(f"\nTraining the final model with best params")
train_and_evaluate_distilbert_model(best_params, train_dataloader, val_dataloader, num_labels, save_model=True)
