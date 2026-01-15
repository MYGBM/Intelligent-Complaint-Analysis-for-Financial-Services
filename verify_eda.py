from src.eda import DataPrep
import os

# Path to data
DATA_PATH = r'c:\Users\yeget\Intelligent-Complaint-Analysis-for-Financial-Services\data\raw\complaints.csv'

def run_verification():
    print("Initializing DataPrep...")
    eda = DataPrep(DATA_PATH)
    
    print("Loading sample data (nrows=5000)...")
    eda.load_data(nrows=5000)
    
    print("Running Sanity Check...")
    eda.sanity_check()
    
    print("Running Preprocessing (Cleaning)...")
    clean_df = eda.preprocess_data()
    
    print("Saving processed data...")
    eda.save_to_csv(r'data/processed/complaints_preprocessed.csv')
    eda.save_to_parquet(r'data/processed/complaints_preprocessed.parquet')
    
    print("\nVerification Results:")
    print(f"Final DataFrame Shape: {clean_df.shape}")
    print("Sample Cleaned Narrative:")
    if not clean_df.empty:
        print(clean_df[['Consumer complaint narrative', 'cleaned_narrative']].iloc[0])
    else:
        print("DataFrame is empty after filtering (might happen with small sample if no target products found).")

if __name__ == "__main__":
    run_verification()
