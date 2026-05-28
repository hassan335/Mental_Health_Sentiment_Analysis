# Student Names

# Muhammad Hassan Siddiqui
# Zeeshan Ghaffar
# Shahid Afridi Patel



import pandas as pd
import re
import string
import nltk
from sklearn.preprocessing import LabelEncoder
from nltk.corpus import stopwords

# Download NLTK stopwords
nltk.download('stopwords')

# Load your sentiment analysis dataset
df = pd.read_csv("/content/Combined_Data.csv")  

# Drop unnecessary columns
df = df.drop(df.columns[[0]], axis=1)

# Drop missing values and reset index
df.dropna(inplace=True)
df.reset_index(drop=True, inplace=True)

# Text cleaning function
def clean_text(text):
    text = re.sub(r'https?://\S+|www\.\S+|\[.*?\]\(.*?\)', '', text)  # Remove URLs
    text = re.sub(r'<.*?>+', '', text)  # Remove HTML tags
    text = re.sub(r'@\w+', '', text)  # Remove handles
    text = re.sub(f'[{re.escape(string.punctuation)}]', '', text)  # Remove punctuation
    text = re.sub(r'\n', ' ', text)  # Remove newline characters
    text = re.sub(r'\w*\d\w*', '', text)  # Remove words containing numbers
    text = re.sub(r'\s+', ' ', text)  # Remove extra spaces
    return text.strip()

# Clean the text data
df['statement'] = df['statement'].str.lower()
df['statement'] = df['statement'].apply(clean_text)

# Remove stopwords
stopWords = stopwords.words('english')
df['statement'] = df['statement'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stopWords)]))

# Rename columns
df.rename(columns={'statement': 'text', 'status': 'label'}, inplace=True)

# Reorder the columns
df = df[['label', 'text']]

# Correct label mapping
label_mapping = {
    'Stress': 0,
    'Normal': 1,
    'Bipolar': 2,
    'Anxiety': 3,
    'Suicidal': 4,
    'Depression': 5,
    'Personality disorder': 6
}

# Map labels
df['label'] = df['label'].map(label_mapping)

# Check for unmapped labels
if df['label'].isnull().any():
    raise ValueError("There are still unmapped labels. Please update the label mapping accordingly.")

# Encode labels
df['label'] = df['label'].astype(int)

# Save the preprocessed DataFrame to a CSV file
df.to_csv('preprocessed_data.csv', index=False)

# Confirm the file has been saved
print("Preprocessed data saved as 'preprocessed_data.csv'")





