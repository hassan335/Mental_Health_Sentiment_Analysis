# Muhammad Hassan Siddiqui 202407171


import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from peft import LoraConfig, get_peft_model, TaskType
from transformers import logging
import os

# Suppress transformer warnings to reduce unnecessary output
logging.set_verbosity_error()

# Disable W&B (Weights and Biases) logging
os.environ["WANDB_DISABLED"] = "true"

# Check if a GPU is available and set the device accordingly (either GPU or CPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Step 1: Load and Prepare Dataset
df = pd.read_csv("/content/Combined_Data.csv")  # Load dataset (make sure the path is correct)
df = df.dropna(subset=["text"])  # Drop rows with missing text
df['label'] = df['label'].astype(int)  # Ensure that labels are in integer format
df['text'] = df['text'].astype(str)  # Ensure that text is in string format

# Step 2: Split Dataset into A and B for training and evaluation
A, B = train_test_split(df, test_size=0.5, random_state=42)  # Split into two halves

# Convert A and B to Hugging Face Dataset format for easy processing
from datasets import Dataset
dataset_A = Dataset.from_pandas(A)
dataset_B = Dataset.from_pandas(B)

# Step 3: Tokenize Datasets
tokenizer = AutoTokenizer.from_pretrained("/content/fine-tuned-model/")  # Load tokenizer from fine-tuned model

# Tokenization function for dataset text (padding, truncation, and max length setting)
def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=128)

# Apply tokenization to both datasets A and B
dataset_A = dataset_A.map(tokenize_function, batched=True)
dataset_B = dataset_B.map(tokenize_function, batched=True)

# Remove unnecessary columns (e.g., raw text) to keep only the relevant tokenized data
dataset_A = dataset_A.remove_columns(["text"])
dataset_B = dataset_B.remove_columns(["text"])

# Set the format of the datasets to "torch" for PyTorch compatibility
dataset_A.set_format("torch")
dataset_B.set_format("torch")

# Step 4: Load Pre-Fine-Tuned Model and Configure LoRA for PEFT
model = AutoModelForSequenceClassification.from_pretrained("/content/fine-tuned-model/", num_labels=len(df['label'].unique()))  # Load fine-tuned model
model.to(device)  # Move model to the correct device (GPU or CPU)

# Configure LoRA (Low-Rank Adaptation) for Parameter-Efficient Fine-Tuning (PEFT)
lora_config = LoraConfig(
    task_type=TaskType.SEQ_CLS,  # Sequence classification task
    r=8,  # LoRA rank
    lora_alpha=16,  # LoRA scaling factor
    lora_dropout=0.1,  # Dropout rate
    target_modules=["q_lin", "k_lin", "v_lin"]  # Layers to apply LoRA to (DistilBERT-compatible)
)

# Wrap the model with LoRA to apply PEFT
peft_model = get_peft_model(model, lora_config)

# Step 5: Train PEFT-Tuned Model on Dataset A
# Set up training arguments for the Trainer API
training_args = TrainingArguments(
    output_dir="./peft_results",  # Directory to save results
    eval_strategy="epoch",  # Evaluate at the end of each epoch
    learning_rate=2e-5,  # Learning rate for fine-tuning
    num_train_epochs=3,  # Number of epochs for training
    per_device_train_batch_size=16,  # Batch size per device during training
    save_strategy="epoch",  # Save model at the end of each epoch
    load_best_model_at_end=True,  # Load the best model after training
    logging_dir="./logs",  # Directory to store logs
    report_to="none"  # Disable W&B logging
)

# Create the Trainer instance for model training
trainer = Trainer(
    model=peft_model,  # Model to train
    args=training_args,  # Training arguments
    train_dataset=dataset_A,  # Training dataset
    eval_dataset=dataset_A,  # Evaluation dataset
    tokenizer=tokenizer  # Tokenizer used for processing text
)

# Start training
trainer.train()

# Save the PEFT-tuned model to disk
peft_model.save_pretrained("/content/peft_tuned_model")

# Step 6: Extract Embeddings from Dataset B using the PEFT-tuned Model
# Define a function to extract embeddings for the dataset using the trained model
def extract_embeddings(dataset, model, batch_size=16):
    model.eval()  # Set the model to evaluation mode
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)  # Create DataLoader for batching
    embeddings = []  # List to store embeddings
    with torch.no_grad():  # Disable gradient calculation for inference
        for batch in dataloader:
            inputs = {k: v.to(device) for k, v in batch.items() if k in ["input_ids", "attention_mask"]}
            outputs = model.distilbert(**inputs)  # Use DistilBERT model to get the outputs
            cls_embeddings = outputs.last_hidden_state[:, 0, :]  # Extract the CLS token embeddings (first token)
            embeddings.append(cls_embeddings.cpu().numpy())  # Append embeddings to list
    return np.vstack(embeddings)  # Return stacked embeddings as numpy array

# Extract embeddings from Dataset B using the PEFT-tuned model
embeddings_B_peft = extract_embeddings(dataset_B, peft_model)

# Save the embeddings along with labels and text for later analysis
embeddings_df_peft = pd.DataFrame(embeddings_B_peft)
embeddings_df_peft['label'] = B['label'].values
embeddings_df_peft['text'] = B['text'].values
embeddings_df_peft.to_csv("/content/embeddings_B_peft.csv", index=False)

# Step 7: Extract Baseline Embeddings (Without PEFT)
# Load the original fine-tuned model (baseline) to compare against PEFT
baseline_model = AutoModelForSequenceClassification.from_pretrained("/content/peft_tuned_model", num_labels=len(df['label'].unique()))
baseline_model.to(device)

# Extract embeddings from Dataset B using the baseline model
embeddings_B_baseline = extract_embeddings(dataset_B, baseline_model)

# Save baseline embeddings for further analysis
embeddings_df_baseline = pd.DataFrame(embeddings_B_baseline)
embeddings_df_baseline['label'] = B['label'].values
embeddings_df_baseline['text'] = B['text'].values
embeddings_df_baseline.to_csv("/content/embeddings_B_baseline.csv", index=False)

# Step 8: Comparison Using `df-analyze`
# Print a message indicating the results are saved for comparison
print("Embeddings for PEFT model saved as embeddings_B_peft.csv")

# The results can now be compared using df-analyze or any other downstream analysis tools.
