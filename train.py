import torch
import numpy as np
import pandas as pd
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, TrainingArguments, Trainer, DataCollatorWithPadding, EarlyStoppingCallback
from sklearn.model_selection import train_test_split
from pso import PSO
import evaluate


class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get("labels")
        # forward pass
        outputs = model(**inputs)
        logits = outputs.get("logits")
        # compute custom loss
        loss_fct = torch.nn.CrossEntropyLoss(weight=class_weights).to(device)
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss
    
accuracy = evaluate.load("accuracy")
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return accuracy.compute(predictions=predictions, references=labels)

def preprocessing(x):
    tokenized_x = tokenizer(x['text'], truncation=True, padding=True)
    tokenized_x['labels'] = x['label']
    return tokenized_x



def train_and_evaluate_distilbert_model(params, tokenizer, tokenized_dataset, num_labels, id2label, label2id, save_model=False):
    '''Train and evaluate a DistilBERT model with the given hyperparameters.'''
    learning_rate, dropout_prob, num_epochs = params[0], 0, 1

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=num_labels, id2label=id2label, label2id=label2id , dropout=dropout_prob, attention_dropout=dropout_prob, seq_classif_dropout=dropout_prob*1.5)
    for name, param in list(model.named_parameters())[:-4]:
        param.requires_grad = False
    #for name, param in model.named_parameters():
    #    print(name, param.requires_grad)
    #quit()

    training_args = TrainingArguments(
    output_dir="NLP_project",
    learning_rate=learning_rate,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    num_train_epochs=num_epochs,
    weight_decay=0.01,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="eval_accuracy",
    greater_is_better=True,
    )

    trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset['train'],
    eval_dataset=tokenized_dataset['val'],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)])

    trainer.train()
    
    if save_model:
        print("\nEvaluating the final model on a test set")
        test_accuracy = trainer.evaluate(tokenized_dataset['test'])['eval_accuracy']
        print(f"Accuracy on the test set: {test_accuracy:.4f}")
        print("\nSaving the final model")
        model.save_pretrained("distilbert_model_best_params")

    return trainer.evaluate()['eval_accuracy']
    

def objective_function(params):
    '''Objective function for the PSO algorithm.'''
    return -train_and_evaluate_distilbert_model(params, tokenizer, tokenized_dataset, num_labels, id2label, label2id)



device = torch.device("cuda")

df = pd.read_csv("truncated_mtsamples.csv")

df.dropna(inplace=True) # Drop any rows with missing values
texts = df['transcription'].tolist() # a list of text samples
labels = df['medical_specialty'].astype('category').cat.codes.tolist()

train_texts, val_texts, train_labels, val_labels = train_test_split(texts, labels, test_size=.3)
val_texts, test_texts, val_labels, test_labels = train_test_split(val_texts, val_labels, test_size=.3)

# make datasets of format dict(label=..., text=...)
dataset = {}
dataset['train'] = [{"label": label, "text": text} for label, text in zip(train_labels, train_texts)]
dataset['val'] = [{"label": label, "text": text} for label, text in zip(val_labels, val_texts)]
dataset['test'] = [{"label": label, "text": text} for label, text in zip(test_labels, test_texts)]

# tokenize datasets
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
tokenized_dataset = {}
tokenized_dataset['train'] = [preprocessing(x) for x in dataset['train']]
tokenized_dataset['val'] = [preprocessing(x) for x in dataset['val']]
tokenized_dataset['test'] = [preprocessing(x) for x in dataset['test']]

label2id = {v: i for i, v in enumerate(df['medical_specialty'].astype('category').cat.categories)}
id2label = {i: v for i, v in enumerate(df['medical_specialty'].astype('category').cat.categories)}

class_counts = df['medical_specialty'].value_counts()
num_labels = len(class_counts)
# class_weights = 1 / torch.tensor(class_counts, dtype=torch.float) # used for weighted loss

# Split the dataset into training, validation and test sets
train_texts, val_texts, train_labels, val_labels = train_test_split(texts, labels, test_size=.3)
val_texts, test_texts, val_labels, test_labels = train_test_split(val_texts, val_labels, test_size=.3)


# Define the hyperparameter ranges to search over
param_ranges = np.array([
    [1e-8, 1e-3],  # learning rate
#    [0, 0.1],    # dropout probability
#    [1, 3]         # number of epochs
])

# Set up the PSO algorithm with the objective function and hyperparameter ranges
pso = PSO(objective_function, param_ranges, num_particles=10, max_iter=5)

# Run the PSO algorithm to find the optimal hyperparameters
best_params, best_score = pso.run()
pso.plot_logs()

#best_params = [1e-4, 0, 5]

# Train the DistilBERT model with the optimal hyperparameters on the entire dataset
# print(f"Best params: \nLearning rate {best_params[0]} \nDropout probability {best_params[1]} \nNum epochs {best_params[2]}")
print(f"\nTraining the final model with best params")
train_and_evaluate_distilbert_model(best_params, tokenizer, tokenized_dataset, num_labels, id2label, label2id, save_model=True)
