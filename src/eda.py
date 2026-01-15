import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import re
import os

# Ensure NLTK data is downloaded
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
    nltk.data.find('corpora/wordnet')
    nltk.data.find('corpora/omw-1.4')
except LookupError:
    print("Downloading necessary NLTK data...")
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
    nltk.download('omw-1.4', quiet=True)

class DataPrep:
    def __init__(self, file_path):
        self.file_path = file_path
        self.df = None
        self.filtered_products = [
            "Credit card",
            "Personal loan",
            "Savings account",
            "Money transfers"
        ]
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))

    def load_data(self, **kwargs):
        """
        Loads the dataset efficiently. 
        Accepts kwargs like nrows for testing.
        """
        print(f"Loading data from {self.file_path}...")
        try:
            self.df = pd.read_csv(self.file_path, low_memory=False, **kwargs)
            print(f"Data loaded successfully. Shape: {self.df.shape}")
        except Exception as e:
            print(f"Error loading data: {e}")
            raise e

    def sanity_check(self):
        """
        Performs initial sanity checks for NaN and Empty strings.
        """
        if self.df is None:
            raise ValueError("Dataframe not loaded. Call load_data() first.")
            
        print("\n--- Sanity Check ---")
        total_rows = len(self.df)
        
        # Check 'Product'
        if 'Product' in self.df.columns:
            nan_products = self.df['Product'].isna().sum()
            print(f"Missing 'Product': {nan_products} rows")
        
        # Check 'Consumer complaint narrative'
        if 'Consumer complaint narrative' in self.df.columns:
            nan_narratives = self.df['Consumer complaint narrative'].isna().sum()
            empty_narratives = (self.df['Consumer complaint narrative'] == "").sum()
            whitespace_narratives = (self.df['Consumer complaint narrative'].str.strip() == "").sum()
            
            print(f"Missing 'Consumer complaint narrative': {nan_narratives} rows")
            print(f"Empty string narratives: {empty_narratives} rows") 
            print(f"Whitespace-only narratives: {whitespace_narratives} rows")
            
            valid_count = total_rows - nan_narratives
            print(f"Valid narratives available for analysis: {valid_count}")
        else:
            print("CRITICAL: 'Consumer complaint narrative' column not found!")

    def clean_text(self, text):
        """
        Applies strict normalization: lowercasing, boilerplate removal, tokenization,
        stopwords removal, and lemmatization.
        """
        if not isinstance(text, str):
            return ""
        
        # 1. Lowercase
        text = text.lower()
        
        # 2. Remove boilerplate
        text = text.replace("i am writing to file a complaint", "")
        text = text.replace("xxxx", "") # Common redaction in financial datasets
        
        # 3. Remove Special Characters & Digits
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # 4. Tokenization
        tokens = word_tokenize(text)
        
        # 5. Stopwords & Lemmatization
        clean_tokens = [
            self.lemmatizer.lemmatize(word) 
            for word in tokens 
            if word not in self.stop_words and len(word) > 2
        ]
        
        return " ".join(clean_tokens)

    def preprocess_data(self):
        """
        Filters by product, drops empty narratives, and applies text cleaning.
        Returns the cleaned DataFrame.
        """
        if self.df is None:
            load_data()

        print("\n--- Preprocessing ---")
        initial_count = len(self.df)
        
        # 1. Product Filter
        # Note: Normalizing product names if they are slightly different in raw data
        # Using simple containment check or exact match
        self.df = self.df[self.df['Product'].isin(self.filtered_products)]
        print(f"After filtering for 5 target products: {len(self.df)} rows")
        
        # 2. Drop Empty Narratives
        self.df = self.df.dropna(subset=['Consumer complaint narrative'])
        self.df = self.df[self.df['Consumer complaint narrative'].str.strip() != ""]
        print(f"After dropping missing/empty narratives: {len(self.df)} rows")
        
        # 3. Text Normalization
        print("Applying advanced text normalization (this may take a while)...")
        # Using a sample if testing, but assume full run for production
        # Vectorized apply is slow for cleaning, but robust.
        self.df['cleaned_narrative'] = self.df['Consumer complaint narrative'].apply(self.clean_text)
        
        print("Preprocessing complete.")
        return self.df

    def save_to_csv(self, output_path):
        """
        Saves the current dataframe to a CSV file.
        """
        if self.df is None:
            print("No data to save.")
            return

        try:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            self.df.to_csv(output_path, index=False)
            print(f"Saved processed data to {output_path}")
        except Exception as e:
            print(f"Error saving to CSV: {e}")

    def save_to_parquet(self, output_path):
        """
        Saves the current dataframe to a Parquet file (more efficient).
        """
        if self.df is None:
            print("No data to save.")
            return

        try:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            self.df.to_parquet(output_path, index=False)
            print(f"Saved processed data to {output_path}")
        except Exception as e:
            print(f"Error saving to Parquet: {e}")