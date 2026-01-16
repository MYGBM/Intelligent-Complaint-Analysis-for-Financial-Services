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
        # Product mapping: target product -> list of source product names
        self.product_mapping = {
            "Credit card": ["Credit card"],
            "Personal loan": ["Payday loan, title loan, personal loan, or advance loan", 
                            "Payday loan, title loan, or personal loan"],
            "Savings account": ["Checking or savings account"],
            "Money transfers": ["Money transfers"]
        }
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
        
        # Check 'Issue'
        if 'Issue' in self.df.columns:
            nan_issues = self.df['Issue'].isna().sum()
            print(f"Missing 'Issue': {nan_issues} rows")
        else:
            print("WARNING: 'Issue' column not found!")
        
        # Check 'Sub-issue'
        if 'Sub-issue' in self.df.columns:
            nan_subissues = self.df['Sub-issue'].isna().sum()
            print(f"Missing 'Sub-issue': {nan_subissues} rows")
        else:
            print("WARNING: 'Sub-issue' column not found!")
        
        #Check missing 'Complaint ID'
        if 'Complaint ID' in self.df.columns:
            nan_complaint_id = self.df['Complaint ID'].isna().sum()
            print(f"Missing 'Complaint ID': {nan_complaint_id} rows")
        
        # Check for duplicate Complaint IDs
        if 'Complaint ID' in self.df.columns:
            duplicate_ids = self.df['Complaint ID'].duplicated().sum()
            print(f"Duplicate 'Complaint ID': {duplicate_ids} rows")
            if duplicate_ids > 0:
                print("WARNING: Duplicate Complaint IDs found! Consider investigating.")
        else:
            print("WARNING: 'Complaint ID' column not found!")

    def clean_text(self, text):
        """
        Applies strict normalization: lowercasing, boilerplate removal, tokenization,
        stopwords removal, and lemmatization.
        Keeps numbers as they are part of complaints (e.g., $200, 90%, 300).
        """
        if not isinstance(text, str):
            return ""
        
        # 1. Lowercase
        text = text.lower()
        
        # 2. Remove boilerplate
        text = text.replace("i am writing to file a complaint", "")
        # Remove both uppercase and lowercase xxxx redactions
        text = re.sub(r'x{2,}', '', text, flags=re.IGNORECASE)
        
        # 3. Remove Special Characters but KEEP numbers and basic punctuation
        # Keep alphanumeric, spaces, dollar signs, percent signs, and periods
        text = re.sub(r'[^a-zA-Z0-9\s$%.,-]', '', text)
        
        # 4. Tokenization
        tokens = word_tokenize(text)
        
        # 5. Stopwords & Lemmatization (but skip tokens that contain numbers)
        clean_tokens = []
        for word in tokens:
            # If the word contains any digit, keep it as-is
            if any(char.isdigit() for char in word):
                clean_tokens.append(word)
            # Otherwise apply stopwords filter and lemmatization
            elif word not in self.stop_words and len(word) > 2:
                clean_tokens.append(self.lemmatizer.lemmatize(word))
        
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
        
        # 1. Product Filter using mapping
        print("Mapping products to target categories...")
        # Create a reverse mapping for easier lookup
        product_to_category = {}
        for target, sources in self.product_mapping.items():
            for source in sources:
                product_to_category[source] = target
        
        # Map products to standardized names
        self.df['Product_Original'] = self.df['Product']
        self.df['Product'] = self.df['Product'].map(product_to_category)
        
        # Filter only rows that got mapped (non-null after mapping)
        self.df = self.df.dropna(subset=['Product'])
        print(f"After filtering for target products: {len(self.df)} rows")
        
        # Verify filtering worked - show unique products
        unique_products = self.df['Product'].unique()
        print(f"\n✓ Products in filtered data: {sorted(unique_products)}")
        print(f"Product counts after filtering:")
        print(self.df['Product'].value_counts())
        
        # 2. Drop Empty Narratives
        self.df = self.df.dropna(subset=['Consumer complaint narrative'])
        self.df = self.df[self.df['Consumer complaint narrative'].str.strip() != ""]
        print(f"\nAfter dropping missing/empty narratives: {len(self.df)} rows")
        
        # 3. Text Normalization
        print("\nApplying advanced text normalization (this may take a while)...")
        self.df['cleaned_narrative'] = self.df['Consumer complaint narrative'].apply(self.clean_text)
        
        # 4. Post-cleaning validation: Check for missing critical fields
        print("\n--- Post-Cleaning Validation ---")
        missing_product = self.df['Product'].isna().sum()
        missing_complaint_id = self.df['Complaint ID'].isna().sum() if 'Complaint ID' in self.df.columns else 0
        missing_issue = self.df['Issue'].isna().sum() if 'Issue' in self.df.columns else 0
        missing_subissue = self.df['Sub-issue'].isna().sum() if 'Sub-issue' in self.df.columns else 0
        
        print(f"Rows with missing 'Product': {missing_product}")
        print(f"Rows with missing 'Complaint ID': {missing_complaint_id}")
        print(f"Rows with missing 'Issue': {missing_issue}")
        print(f"Rows with missing 'Sub-issue': {missing_subissue}")
        
        if missing_product > 0 or missing_complaint_id > 0 or missing_issue > 0 or missing_subissue > 0:
            print("\n⚠️ WARNING: Some rows have missing critical fields!")
            print("Please review and drop these rows before saving if needed.")
        
        print("\nPreprocessing complete.")
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