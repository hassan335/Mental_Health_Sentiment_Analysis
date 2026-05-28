# Group Member Names

Muhammad Hassan Siddiqui





import pandas as pd  # Importing Pandas module alias as pd 

df = pd.read_csv("/content/Combined_Data.csv")  # Reading a CSV File using the method available in pandas module

df = df.drop(df.columns[[0]], axis=1) # excluding a unique numerical column from dataset


df.rename(columns={'statement': 'text', 'status': 'label'}, inplace=True) # Renaming the column name statement to text and status to label in original dataframe

df = df[['label', 'text']]  # Reorder the columns to have 'label' first and 'text' second



# Define a mapping from class names to integers



label_mapping = {
    'Stress': 0,
    'Normal': 1,
    'Bi-Polar': 2,
    'Anxiety': 3,
    'Suicidal': 4,
    'Depression': 5,
    'Anxiety': 6,
    'Personality Disorder': 7

}



df['label'] = df['label'].map(label_mapping)  # converting the class names into int by applying Map method


df.dropna(inplace=True) # removing any row that has missing values    
label_encoder = LabelEncoder() #instantiating the object of LabelEncoder

 #Fit and transform the label column to convert string labels to integers
df['label'] = label_encoder.fit_transform(df['label'])

# Ensure the label column is of integer type
df['label'] = df['label'].astype(int)

df.to_parquet('Sentiment_Analysis.parquet', index=False)  # Save as Parquet
