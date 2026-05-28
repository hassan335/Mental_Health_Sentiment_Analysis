import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from datasets import Dataset
from transformers import DistilBertForSequenceClassification, DistilBertTokenizerFast, Trainer, TrainingArguments

# Load dataset
df = pd.read_csv('./content/Combined_Data.csv')
df['label'] = df['label'].astype(int)

df = df.dropna(subset=['text'])
df['text'] = df['text'].astype(str)

# Check device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Split dataset
A, B = train_test_split(df, test_size=0.1, random_state=42)
A1, A2 = train_test_split(A, test_size=0.1111, random_state=42)

# Compute class weights
class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(A1['label']), y=A1['label'])
class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)

# Custom model with class weights
class CustomDistilBERT(DistilBertForSequenceClassification):
    def __init__(self, config, class_weights):
        super().__init__(config)
        self.class_weights = class_weights
        self.loss_fn = torch.nn.CrossEntropyLoss(weight=self.class_weights)

    def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
        outputs = super().forward(input_ids=input_ids, attention_mask=attention_mask, **kwargs)
        logits = outputs.logits
        loss = None
        if labels is not None:
            loss = self.loss_fn(logits, labels)
        return {'loss': loss, 'logits': logits} if loss is not None else outputs

# Convert to Hugging Face Dataset
train_dataset = Dataset.from_pandas(A1)
eval_dataset = Dataset.from_pandas(A2)
test_dataset = Dataset.from_pandas(B)

# Tokenizer
tokenizer = DistilBertTokenizerFast.from_pretrained("./distilbert_base_uncased")


# Tokenization
def tokenize_function(examples):
    return tokenizer(examples['text'], padding='max_length', truncation=True, max_length=128)

train_dataset = train_dataset.map(tokenize_function, batched=True)
eval_dataset = eval_dataset.map(tokenize_function, batched=True)
test_dataset = test_dataset.map(tokenize_function, batched=True)

train_dataset = train_dataset.remove_columns(['text'])
eval_dataset = eval_dataset.remove_columns(['text'])
test_dataset = test_dataset.remove_columns(['text'])

train_dataset.set_format('torch')
eval_dataset.set_format('torch')
test_dataset.set_format('torch')

# Model
model = CustomDistilBERT.from_pretrained("./distilbert_base_uncased", num_labels=7, class_weights=class_weights)
model.to(device)

# Training arguments
training_args = TrainingArguments(
    output_dir='./results',
    evaluation_strategy='steps',
    save_strategy='steps',
    save_steps=500,
    eval_steps=500,
    learning_rate=5e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir='./logs',
    load_best_model_at_end=True,
    save_total_limit=2,
    fp16=True
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    compute_metrics=lambda pred: {
        'accuracy': accuracy_score(pred.label_ids, np.argmax(pred.predictions, axis=1)),
        'precision': precision_recall_fscore_support(pred.label_ids, np.argmax(pred.predictions, axis=1), average='weighted')[0],
        'recall': precision_recall_fscore_support(pred.label_ids, np.argmax(pred.predictions, axis=1), average='weighted')[1],
        'f1': precision_recall_fscore_support(pred.label_ids, np.argmax(pred.predictions, axis=1), average='weighted')[2],
    }
)

# Train and save
trainer.train()
model.save_pretrained('./fine_tuned_model')

# Evaluate on test dataset
test_results = trainer.predict(test_dataset)
print(test_results.metrics)

# Embedding extraction function
def extract_embeddings(dataset, model, batch_size=16):
    model.eval()
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)
    embeddings = []
    with torch.no_grad():
        for batch in dataloader:
            inputs = {k: v.to(device) for k, v in batch.items() if k != "label"}
            outputs = model.distilbert(**inputs)
            cls_embeddings = outputs.last_hidden_state[:, 0, :]  # CLS token
            embeddings.append(cls_embeddings.cpu().numpy())
    return np.vstack(embeddings)

# Extract embeddings for Dataset B
print("Extracting embeddings for the test dataset...")
embeddings_B = extract_embeddings(test_dataset, model)

# Save embeddings
embeddings_df = pd.DataFrame(embeddings_B)
embeddings_df['label'] = B['label'].values
embeddings_df['text'] = B['text'].values
embeddings_df.to_parquet('embeddings_B.parquet', index=False)
embeddings_df.to_csv('embeddings_B.csv', index=False)
print("Embeddings saved successfully.")
