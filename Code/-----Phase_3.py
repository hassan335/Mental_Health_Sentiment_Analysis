# Student Names

# Muhammad Hassan Siddiqui
# Zeeshan Ghaffar
# Shahid Afridi Patel


import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from datasets import Dataset
from transformers import DistilBertForSequenceClassification, DistilBertTokenizerFast, Trainer, TrainingArguments

# Load dataset
df = pd.read_csv('./content/Combined_Data.csv')  # Make sure the path is correct and accessible
df['label'] = df['label'].astype(int)  # Ensure labels are integers

# Check device (GPU or CPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# Split dataset into A (90%) and B (10%)
A, B = train_test_split(df, test_size=0.1, random_state=42)

# Further split A into A1 (80%) and A2 (10% of original)
A1, A2 = train_test_split(A, test_size=0.1111, random_state=42)  # 0.1111 ensures A2 is 10% of original

# Convert to Hugging Face Dataset format
train_dataset = Dataset.from_pandas(A1)
eval_dataset = Dataset.from_pandas(A2)
test_dataset = Dataset.from_pandas(B)

# Load tokenizer locally 
tokenizer = DistilBertTokenizerFast.from_pretrained('./distilbert_base_uncased')
print("Tokenizer loaded successfully!")


# Define the tokenization function
def tokenize_function(examples):
    # Convert all items in examples['text'] to strings to prevent errors
    texts = [str(text) for text in examples['text']]
    return tokenizer(texts, padding='max_length', truncation=True, max_length=100)





# Tokenize the datasets
train_dataset = train_dataset.map(tokenize_function, batched=True)
eval_dataset = eval_dataset.map(tokenize_function, batched=True)
test_dataset = test_dataset.map(tokenize_function, batched=True)

# Load pre-trained model from local directory, which includes model.safetensors
model = DistilBertForSequenceClassification.from_pretrained('./distilbert_base_uncased', num_labels=7, from_tf=False, use_safetensors=True)
model.to(device)
print("Model loaded successfully!")

# Set training arguments for the model
training_args = TrainingArguments(
    output_dir='./results',  # Directory to store model checkpoints and outputs
    report_to='none',  # Disable logging to external services (e.g., W&B)
    evaluation_strategy='epoch',  # Evaluate the model at the end of each epoch
    save_strategy='epoch',  # Save model checkpoints at the end of each epoch
    learning_rate=5e-5,  # Learning rate for model optimization
    per_device_train_batch_size=16,  # Batch size for training
    num_train_epochs=3,  # Number of training epochs
    load_best_model_at_end=True,  # Load the best model after training
    save_total_limit=2  # Keep only the two most recent model checkpoints
)

# Create Trainer instance with the specified arguments and datasets
trainer = Trainer(
    model=model,  # The model to be trained
    args=training_args,  # Training arguments defined above
    train_dataset=train_dataset,  # Training dataset
    eval_dataset=eval_dataset,  # Evaluation dataset
)


# Train the model
trainer.train()

# Save the fine-tuned model
model.save_pretrained('./fine_tuned_model')
print("Fine-tuned model saved successfully.")

# Extract embeddings for the test dataset
# Define a function to extract embeddings from a dataset
def extract_embeddings(dataset):
    model.eval()  # Set the model to evaluation mode (disables dropout, etc.)
    embeddings = []  # Initialize a list to store the embeddings

    # Iterate through each sample in the dataset
    for i in range(len(dataset)):
        # Extract input IDs and attention masks, and move them to the device (GPU or CPU)
        input_ids = torch.tensor(dataset[i]['input_ids']).unsqueeze(0).to(device)
        attention_mask = torch.tensor(dataset[i]['attention_mask']).unsqueeze(0).to(device)

        # Perform inference without updating gradients (no backpropagation)
        with torch.no_grad():
            # Pass input through the DistilBERT model to get the output
            outputs = model.distilbert(input_ids=input_ids, attention_mask=attention_mask)
            # Pool the last hidden states by averaging across the token dimension
            pooled_output = outputs.last_hidden_state.mean(dim=1)
            # Append the pooled output (embedding) to the embeddings list
            embeddings.append(pooled_output.squeeze().cpu().numpy())

    # Return the list of embeddings as a NumPy array
    return np.array(embeddings)


# Extract embeddings for Dataset B
print("Extracting embeddings for the test dataset...")
embeddings_B = extract_embeddings(test_dataset)

# Create a DataFrame for embeddings
embeddings_df = pd.DataFrame(embeddings_B)

# Add labels from Dataset B
embeddings_df['label'] = B['label'].values

# Add text from Dataset B
embeddings_df['text'] = B['text'].values

# Save embeddings as a Parquet file
embeddings_df.to_parquet('embeddings_B.parquet', index=False)
print("Embeddings saved as 'embeddings_B.parquet'.")

# Save embeddings as a CSV file
embeddings_df.to_csv('embeddings_B.csv', index=False)
print("Embeddings saved as 'embeddings_B.csv'.")

# Below are some of the articles we cover in order to get some insights  

# https://huggingface.co/docs/transformers/en/training
# https://mariamkilibechir.medium.com/fine-tuning-of-bert-and-distillbert-for-sentiment-analysis-9f3bb459767b
# https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.to_parquet.html
# https://brighteshun.medium.com/sentiment-analysis-part-1-finetuning-and-hosting-a-text-classification-model-on-huggingface-9d6da6fd856b