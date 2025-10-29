#!/usr/bin/env python3
"""
Preprocessing script for names_data_candidate.csv
Implements stratified train/test split and basic text preprocessing
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
import re


def clean_text(text):
    """
    Basic text cleaning function
    """
    if pd.isna(text):
        return text
    
    # Convert to string and strip whitespace
    text = str(text).strip()
    
    # Normalize whitespace (replace multiple spaces with single space)
    text = re.sub(r'\s+', ' ', text)
    
    return text


def extract_features(df):
    """
    Extract basic features from the dirty_name column
    """
    df = df.copy()
    
    # Basic text features
    df['name_length'] = df['dirty_name'].str.len()
    df['word_count'] = df['dirty_name'].str.split().str.len()
    df['has_prefix'] = df['dirty_name'].str.match(r'^[A-Z]{2,}\.', na=False)
    df['has_suffix'] = df['dirty_name'].str.contains(r'\b(PTY|LTD|INC|CORP|LLC|PTY LTD)\b', case=False, na=False)
    df['is_uppercase'] = df['dirty_name'].str.isupper()
    df['has_numbers'] = df['dirty_name'].str.contains(r'\d', na=False)
    df['has_special_chars'] = df['dirty_name'].str.contains(r'[^\w\s]', na=False)
    
    # Extract common prefixes
    df['prefix'] = df['dirty_name'].str.extract(r'^([A-Z]{2,}\.)', expand=False)
    df['prefix'] = df['prefix'].fillna('None')
    
    return df


def run():
    """Main preprocessing function"""
    
    print("=" * 60)
    print("DATA PREPROCESSING WITH STRATIFIED SPLIT")
    print("=" * 60)
    
    # Load the data
    data_path = Path("data/names_data_candidate.csv")
    
    try:
        df = pd.read_csv(data_path)
        print(f"✅ Loaded dataset: {len(df):,} rows")
        
        # Check class distribution before split
        print(f"\n📊 Original class distribution:")
        class_counts = df['dirty_label'].value_counts()
        for label, count in class_counts.items():
            percentage = (count / len(df)) * 100
            print(f"  {label}: {count:,} ({percentage:.2f}%)")
        
        # Stratified train/test split
        print(f"\n🔄 Performing stratified train/test split...")
        
        X = df['dirty_name']
        y = df['dirty_label']
        
        # 80/20 split with stratification
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=0.2,
            random_state=42,
            stratify=y  # Maintains class proportions
        )
        
        print(f"  Train set: {len(X_train):,} samples")
        print(f"  Test set: {len(X_test):,} samples")
        
        # Verify stratification worked
        print(f"\n📈 Train set class distribution:")
        train_counts = y_train.value_counts()
        for label, count in train_counts.items():
            percentage = (count / len(y_train)) * 100
            print(f"  {label}: {count:,} ({percentage:.2f}%)")
        
        print(f"\n📈 Test set class distribution:")
        test_counts = y_test.value_counts()
        for label, count in test_counts.items():
            percentage = (count / len(y_test)) * 100
            print(f"  {label}: {count:,} ({percentage:.2f}%)")
        
        # Create dataframes
        train_df = pd.DataFrame({'dirty_name': X_train, 'dirty_label': y_train})
        test_df = pd.DataFrame({'dirty_name': X_test, 'dirty_label': y_test})
        
        # Clean the text data
        print(f"\n🧹 Cleaning text data...")
        train_df['dirty_name'] = train_df['dirty_name'].apply(clean_text)
        test_df['dirty_name'] = test_df['dirty_name'].apply(clean_text)
        
        # Extract features from training set
        print(f"\n🔧 Extracting features from training set...")
        train_df_processed = extract_features(train_df)
        
        # Apply same feature extraction to test set (using training set statistics)
        print(f"🔧 Extracting features from test set...")
        test_df_processed = extract_features(test_df)
        
        # Create output directory
        output_dir = Path("data")
        output_dir.mkdir(exist_ok=True)
        
        # Save the splits
        print(f"\n💾 Saving processed data...")
        
        # Save raw splits (before feature extraction)
        train_df.to_csv(output_dir / "train_raw.csv", index=False)
        test_df.to_csv(output_dir / "test_raw.csv", index=False)
        
        # Save processed splits (with features)
        train_df_processed.to_csv(output_dir / "train_processed.csv", index=False)
        test_df_processed.to_csv(output_dir / "test_processed.csv", index=False)
        
        print(f"  ✅ Saved train_raw.csv ({len(train_df):,} rows)")
        print(f"  ✅ Saved test_raw.csv ({len(test_df):,} rows)")
        print(f"  ✅ Saved train_processed.csv ({len(train_df_processed):,} rows)")
        print(f"  ✅ Saved test_processed.csv ({len(test_df_processed):,} rows)")
        
        # Show sample of processed data
        print(f"\n📋 Sample of processed training data:")
        print(train_df_processed[['dirty_name', 'dirty_label', 'name_length', 'word_count', 'has_prefix', 'has_suffix']].head(10).to_string(index=False))
        
        # Feature summary
        print(f"\n📊 Feature Summary:")
        feature_cols = ['name_length', 'word_count', 'has_prefix', 'has_suffix', 'is_uppercase', 'has_numbers', 'has_special_chars']
        for col in feature_cols:
            if col in train_df_processed.columns:
                if col in ['has_prefix', 'has_suffix', 'is_uppercase', 'has_numbers', 'has_special_chars']:
                    true_count = train_df_processed[col].sum()
                    print(f"  {col}: {true_count:,} samples ({true_count/len(train_df_processed)*100:.1f}%)")
                else:
                    mean_val = train_df_processed[col].mean()
                    print(f"  {col}: mean = {mean_val:.2f}")
        
        print(f"\n✅ Preprocessing complete!")
        print(f"📁 All files saved to: {output_dir.absolute()}")
        
    except FileNotFoundError:
        print(f"❌ Error: Could not find the file at {data_path}")
        print("Please make sure the CSV file exists in the correct location.")
    except Exception as e:
        print(f"❌ Error: {str(e)}")


if __name__ == "__main__":
    run()
