"""
Data preparation and preprocessing for sentiment analysis
Downloads IMDb dataset, cleans text, and creates train/val/test splits
"""

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from datasets import load_dataset
import pickle

# Download required NLTK data
print("Downloading NLTK resources...")
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('punkt', quiet=True)

class DataPreprocessor:
    """Text preprocessing and cleaning"""
    
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        print("✅ DataPreprocessor initialized")
    
    def clean_text(self, text):
        """
        Clean and preprocess text:
        - Remove HTML tags
        - Convert to lowercase
        - Remove special characters
        - Tokenize and lemmatize
        - Remove stopwords
        """
        if not isinstance(text, str):
            return ""
        
        # Remove HTML tags
        text = re.sub(r'<[^>]+>', '', text)
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+', '', text)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        
        # Remove special characters and numbers
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Skip if empty
        if len(text) == 0:
            return ""
        
        # Tokenize
        tokens = text.split()
        
        # Remove stopwords and lemmatize (keep words > 2 chars)
        tokens = [
            self.lemmatizer.lemmatize(word) 
            for word in tokens 
            if word not in self.stop_words and len(word) > 2
        ]
        
        return ' '.join(tokens)
    
    def prepare_dataset(self, texts, labels, test_size=0.2):
        """
        Prepare dataset with train/val/test split
        - 70% train
        - 15% validation
        - 15% test
        """
        print("\nCleaning text...")
        cleaned_texts = []
        valid_labels = []
        
        for i, text in enumerate(texts):
            cleaned = self.clean_text(text)
            if cleaned:  # Only keep non-empty texts
                cleaned_texts.append(cleaned)
                valid_labels.append(labels[i])
                if (i + 1) % 5000 == 0:
                    print(f"  Processed {i + 1} texts...")
        
        print(f"✅ Cleaned {len(cleaned_texts)} texts")
        
        # Split: 70% train, 15% val, 15% test
        print("\nSplitting data...")
        X_temp, X_test, y_temp, y_test = train_test_split(
            cleaned_texts, valid_labels, 
            test_size=0.15, random_state=42, stratify=valid_labels
        )
        
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, 
            test_size=0.1765, random_state=42, stratify=y_temp
        )
        
        print(f"  Train: {len(X_train)}")
        print(f"  Val: {len(X_val)}")
        print(f"  Test: {len(X_test)}")
        
        return {
            'train': (X_train, y_train),
            'val': (X_val, y_val),
            'test': (X_test, y_test)
        }


def download_imdb_dataset():
    """Download IMDb dataset from Hugging Face"""
    print("\n" + "="*60)
    print("DOWNLOADING IMDb DATASET")
    print("="*60)
    
    try:
        print("Downloading IMDb dataset from Hugging Face...")
        dataset = load_dataset('imdb', trust_remote_code=True)
        
        # Convert to lists
        train_texts = dataset['train']['text']
        train_labels = dataset['train']['label']
        test_texts = dataset['test']['text']
        test_labels = dataset['test']['label']
        
        # Combine train and test for re-splitting
        all_texts = list(train_texts) + list(test_texts)
        all_labels = list(train_labels) + list(test_labels)
        
        print(f"✅ Downloaded {len(all_texts)} reviews")
        print(f"   Positive: {sum(all_labels)}")
        print(f"   Negative: {len(all_labels) - sum(all_labels)}")
        
        return all_texts, all_labels
    
    except Exception as e:
        print(f"❌ Error downloading dataset: {e}")
        raise


def save_data_splits(prepared_data):
    """Save prepared data to CSV files"""
    print("\n" + "="*60)
    print("SAVING DATA SPLITS")
    print("="*60)
    
    # Create data directory if it doesn't exist
    os.makedirs('data', exist_ok=True)
    
    for split, (X, y) in prepared_data.items():
        df = pd.DataFrame({
            'text': X,
            'label': y
        })
        
        filepath = f'data/{split}_data.csv'
        df.to_csv(filepath, index=False)
        print(f"✅ Saved {filepath} ({len(df)} samples)")
    
    # Also save as pickle for faster loading
    with open('data/prepared_data.pkl', 'wb') as f:
        pickle.dump(prepared_data, f)
    print(f"✅ Saved pickle file for faster loading")


def main():
    print("\n" + "="*60)
    print("🎬 SENTIMENT ANALYSIS - DATA PREPARATION")
    print("="*60)
    
    # Download dataset
    texts, labels = download_imdb_dataset()
    
    # Preprocess
    preprocessor = DataPreprocessor()
    prepared_data = preprocessor.prepare_dataset(texts, labels)
    
    # Save
    save_data_splits(prepared_data)
    
    # Print statistics
    print("\n" + "="*60)
    print("DATA STATISTICS")
    print("="*60)
    for split, (X, y) in prepared_data.items():
        pos_count = sum(y)
        neg_count = len(y) - pos_count
        print(f"\n{split.upper()}:")
        print(f"  Total: {len(y)}")
        print(f"  Positive: {pos_count} ({100*pos_count/len(y):.1f}%)")
        print(f"  Negative: {neg_count} ({100*neg_count/len(y):.1f}%)")
        if len(X) > 0:
            print(f"  Avg text length: {np.mean([len(t.split()) for t in X]):.0f} words")
    
    print("\n" + "="*60)
    print("✅ DATA PREPARATION COMPLETE!")
    print("="*60)


if __name__ == "__main__":
    main()
