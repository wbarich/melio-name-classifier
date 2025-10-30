#!/usr/bin/env python3
"""
Custom sklearn transformers for data preprocessing
Designed to be modular and reusable for both training and inference
"""

import pandas as pd
import numpy as np
import re
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from typing import Union, List
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class TextCleaner(BaseEstimator, TransformerMixin):
    """
    Custom transformer for text cleaning operations
    - Convert to lowercase
    - Remove leading/trailing whitespace
    - Remove special characters (keeping alphanumeric and spaces)
    """
    
    def __init__(self, remove_special_chars: bool = True):
        self.remove_special_chars = remove_special_chars
        
    def fit(self, X, y=None):
        """Fit method - no parameters to learn"""
        return self
    
    def transform(self, X):
        """Transform text data"""
        if isinstance(X, pd.Series):
            X = X.copy()
        elif isinstance(X, np.ndarray):
            X = pd.Series(X)
        else:
            X = pd.Series(X)
        
        # Convert to string and handle NaN values
        X = X.astype(str)
        X = X.replace('nan', '')
        
        # Convert to lowercase
        X = X.str.lower()
        
        # Remove leading and trailing whitespace
        X = X.str.strip()
        
        # Remove special characters if requested
        if self.remove_special_chars:
            # Keep only alphanumeric characters and spaces
            X = X.str.replace(r'[^a-z0-9\s]', '', regex=True)
            
            # Normalize multiple spaces to single space
            X = X.str.replace(r'\s+', ' ', regex=True)
            X = X.str.strip()
        
        return X
    
    def fit_transform(self, X, y=None):
        """Fit and transform in one step"""
        return self.fit(X, y).transform(X)


class DuplicateRemover(BaseEstimator, TransformerMixin):
    """
    Custom transformer to remove duplicate records based on input names
    Keeps the first occurrence of each duplicate
    """
    
    def __init__(self, column_name: str = 'dirty_name'):
        self.column_name = column_name
        self.duplicate_indices_ = None
        
    def fit(self, X, y=None):
        """Identify duplicate indices to remove"""
        if isinstance(X, pd.DataFrame):
            text_series = X[self.column_name]
        else:
            text_series = pd.Series(X)
        
        # Find duplicates, keep first occurrence
        self.duplicate_indices_ = text_series.duplicated()
        return self
    
    def transform(self, X):
        """Remove duplicate records"""
        if isinstance(X, pd.DataFrame):
            # Remove rows where duplicates were found
            X_cleaned = X[~self.duplicate_indices_].copy()
        else:
            # For array-like input, convert to DataFrame first
            X_df = pd.DataFrame({self.column_name: X})
            X_cleaned = X_df[~self.duplicate_indices_].copy()
            X_cleaned = X_cleaned[self.column_name].values
        
        return X_cleaned
    
    def fit_transform(self, X, y=None):
        """Fit and transform in one step"""
        return self.fit(X, y).transform(X)


class FeatureExtractor(BaseEstimator, TransformerMixin):
    """
    Custom transformer to extract features from cleaned text
    """
    
    def __init__(self, column_name: str = 'dirty_name'):
        self.column_name = column_name
        self.feature_names_ = None
        
    def fit(self, X, y=None):
        """Fit method - determine feature names"""
        self.feature_names_ = [
            'name_length', 'word_count', 'has_prefix', 'has_suffix',
            'is_uppercase', 'has_numbers', 'has_special_chars',
            'starts_with_vowel', 'ends_with_vowel',
            # New domain-specific features
            'has_university_keyword', 'has_company_suffix_extended',
            'consecutive_capitals', 'capital_ratio', 'first_word_length',
            'last_word_length'
        ]
        return self
    
    def transform(self, X):
        """Extract features from text data"""
        if isinstance(X, pd.DataFrame):
            text_series = X[self.column_name]
        else:
            text_series = pd.Series(X)
        
        # Ensure we have a pandas Series
        if not isinstance(text_series, pd.Series):
            text_series = pd.Series(text_series)
        
        # Initialize feature matrix
        features = pd.DataFrame(index=text_series.index)
        
        # Basic text features
        features['name_length'] = text_series.str.len()
        features['word_count'] = text_series.str.split().str.len()
        
        # Pattern detection
        features['has_prefix'] = text_series.str.match(r'^(dr|prof|mr|ms|mrs|hon|rev)\.', na=False)
        features['has_suffix'] = text_series.str.contains(r'\b(pty|ltd|inc|corp|llc|pty ltd)\b', case=False, na=False)
        features['is_uppercase'] = text_series.str.isupper()
        features['has_numbers'] = text_series.str.contains(r'\d', na=False)
        features['has_special_chars'] = text_series.str.contains(r'[^\w\s]', na=False)

        # Vowel patterns
        features['starts_with_vowel'] = text_series.str.match(r'^[aeiou]', case=False, na=False)
        features['ends_with_vowel'] = text_series.str.contains(r'[aeiou]$', case=False, na=False)

        # New domain-specific features
        # University keyword detection
        university_keywords = r'\b(university|college|institute|school|academy|polytechnic|conservatory)\b'
        features['has_university_keyword'] = text_series.str.contains(university_keywords, case=False, na=False)

        # Extended company suffix detection
        company_suffixes = r'\b(pty|ltd|inc|corp|llc|pty ltd|limited|co|group|holdings|plc|gmbh|sa|ag)\b'
        features['has_company_suffix_extended'] = text_series.str.contains(company_suffixes, case=False, na=False)

        # Character-level features
        # Count consecutive capital letters (e.g., IBM, UCLA)
        def count_consecutive_capitals(text):
            import re
            if pd.isna(text) or not isinstance(text, str):
                return 0
            # Find all sequences of 2+ consecutive capitals
            matches = re.findall(r'[A-Z]{2,}', text)
            return sum(len(m) for m in matches)

        features['consecutive_capitals'] = text_series.apply(count_consecutive_capitals)

        # Ratio of capital letters to total letters
        def calculate_capital_ratio(text):
            if pd.isna(text) or not isinstance(text, str) or len(text) == 0:
                return 0.0
            letters = [c for c in text if c.isalpha()]
            if len(letters) == 0:
                return 0.0
            capitals = [c for c in letters if c.isupper()]
            return len(capitals) / len(letters)

        features['capital_ratio'] = text_series.apply(calculate_capital_ratio)

        # First and last word lengths (patterns like "The University" or "Smith LLC")
        def get_first_word_length(text):
            if pd.isna(text) or not isinstance(text, str):
                return 0
            words = text.split()
            return len(words[0]) if words else 0

        def get_last_word_length(text):
            if pd.isna(text) or not isinstance(text, str):
                return 0
            words = text.split()
            return len(words[-1]) if words else 0

        features['first_word_length'] = text_series.apply(get_first_word_length)
        features['last_word_length'] = text_series.apply(get_last_word_length)
        
        # Fill NaN values with appropriate defaults
        features = features.fillna(0)
        
        return features
    
    def fit_transform(self, X, y=None):
        """Fit and transform in one step"""
        return self.fit(X, y).transform(X)
    
    def get_feature_names_out(self, input_features=None):
        """Return feature names for sklearn compatibility"""
        return self.feature_names_


class CharNgramFeatureExtractor(BaseEstimator, TransformerMixin):
    """
    Extract character n-gram TF-IDF features
    Captures patterns like 'LLC', 'University', 'Inc.' at the character level
    """

    def __init__(self, ngram_range=(3, 4), max_features=50, column_name='dirty_name'):
        """
        Args:
            ngram_range: tuple of (min_n, max_n) for character n-grams
            max_features: maximum number of features to keep (to control memory)
            column_name: name of the column to extract features from
        """
        self.ngram_range = ngram_range
        self.max_features = max_features
        self.column_name = column_name
        self.vectorizer = None

    def fit(self, X, y=None):
        """Fit the TF-IDF vectorizer on character n-grams"""
        if isinstance(X, pd.DataFrame):
            text_series = X[self.column_name]
        else:
            text_series = pd.Series(X)

        # Initialize TF-IDF vectorizer for character n-grams
        self.vectorizer = TfidfVectorizer(
            analyzer='char',
            ngram_range=self.ngram_range,
            max_features=self.max_features,
            lowercase=True,
            strip_accents='unicode'
        )

        # Fit on the text data
        self.vectorizer.fit(text_series.fillna(''))

        logger.debug(f"Fitted CharNgramFeatureExtractor with {len(self.vectorizer.get_feature_names_out())} features")

        return self

    def transform(self, X):
        """Transform text data into character n-gram TF-IDF features"""
        if self.vectorizer is None:
            raise ValueError("CharNgramFeatureExtractor must be fitted before transform")

        if isinstance(X, pd.DataFrame):
            text_series = X[self.column_name]
        else:
            text_series = pd.Series(X)

        # Transform to TF-IDF features
        tfidf_matrix = self.vectorizer.transform(text_series.fillna(''))

        # Convert to DataFrame with proper feature names
        feature_names = [f'char_ngram_{i}' for i in range(tfidf_matrix.shape[1])]
        tfidf_df = pd.DataFrame(
            tfidf_matrix.toarray(),
            columns=feature_names,
            index=text_series.index
        )

        return tfidf_df

    def fit_transform(self, X, y=None):
        """Fit and transform in one step"""
        return self.fit(X, y).transform(X)

    def get_feature_names_out(self, input_features=None):
        """Return feature names for sklearn compatibility"""
        if self.vectorizer is None:
            return []
        return [f'char_ngram_{i}' for i in range(len(self.vectorizer.get_feature_names_out()))]


class EmbeddingTransformer(BaseEstimator, TransformerMixin):
    """
    Custom transformer for generating sentence embeddings
    Uses sentence-transformers for semantic feature extraction
    """
    
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2', 
                 model_dir: str = 'models/embedding_models',
                 column_name: str = 'dirty_name',
                 cache_embeddings: bool = True):
        self.model_name = model_name
        self.model_dir = Path(model_dir)
        self.column_name = column_name
        self.cache_embeddings = cache_embeddings
        self.model = None
        self.embedding_dim = 384  # all-MiniLM-L6-v2 dimension
        self.embedding_cache = {}
        
    def fit(self, X, y=None):
        """Load the embedding model"""
        try:
            from sentence_transformers import SentenceTransformer
            
            # Check if model exists locally
            model_path = self.model_dir / self.model_name
            if model_path.exists():
                logger.info(f"Loading embedding model from {model_path}")
                self.model = SentenceTransformer(str(model_path))
            else:
                logger.info(f"Downloading embedding model: {self.model_name}")
                self.model = SentenceTransformer(self.model_name)
                
                # Save model locally for future use
                self.model.save(str(model_path))
                logger.info(f"Saved embedding model to {model_path}")
                
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            raise
            
        return self
    
    def transform(self, X):
        """Generate embeddings for input text"""
        if self.model is None:
            raise ValueError("EmbeddingTransformer must be fitted before transform")
            
        if isinstance(X, pd.DataFrame):
            text_series = X[self.column_name]
        else:
            text_series = pd.Series(X)
        
        # Generate embeddings
        texts = text_series.fillna('').tolist()
        embeddings = self.model.encode(texts, convert_to_tensor=False)
        
        # Convert to DataFrame
        embedding_df = pd.DataFrame(
            embeddings,
            columns=[f'embedding_{i}' for i in range(self.embedding_dim)],
            index=text_series.index
        )
        
        return embedding_df
    
    def fit_transform(self, X, y=None):
        """Fit and transform in one step"""
        return self.fit(X, y).transform(X)
    
    def get_feature_names_out(self, input_features=None):
        """Return embedding feature names"""
        return [f'embedding_{i}' for i in range(self.embedding_dim)]


class LabelEncoder(BaseEstimator, TransformerMixin):
    """
    Custom label encoder that preserves label mapping for inference
    """
    
    def __init__(self):
        self.label_mapping_ = {}
        self.inverse_mapping_ = {}
        
    def fit(self, y):
        """Fit label encoder"""
        unique_labels = sorted(pd.Series(y).unique())
        self.label_mapping_ = {label: idx for idx, label in enumerate(unique_labels)}
        self.inverse_mapping_ = {idx: label for label, idx in self.label_mapping_.items()}
        return self
    
    def transform(self, y):
        """Transform labels to numeric"""
        y_series = pd.Series(y)
        return y_series.map(self.label_mapping_).values
    
    def inverse_transform(self, y):
        """Transform numeric labels back to original"""
        y_series = pd.Series(y)
        return y_series.map(self.inverse_mapping_).values
    
    def fit_transform(self, y):
        """Fit and transform in one step"""
        return self.fit(y).transform(y)
    
    def get_classes(self):
        """Get the list of classes"""
        return list(self.label_mapping_.keys())


class DataPreprocessor(BaseEstimator, TransformerMixin):
    """
    Combined preprocessor that handles the full data cleaning pipeline
    """
    
    def __init__(self, text_column: str = 'dirty_name', 
                 label_column: str = 'dirty_label', 
                 use_char_ngrams: bool = True,
                 use_embeddings: bool = True,  # New parameter
                 embedding_model: str = 'all-MiniLM-L6-v2'):  # New parameter
        self.text_column = text_column
        self.label_column = label_column
        self.use_char_ngrams = use_char_ngrams
        self.use_embeddings = use_embeddings  # New
        self.embedding_model = embedding_model  # New
        
        # Existing transformers
        self.text_cleaner = TextCleaner()
        self.duplicate_remover = DuplicateRemover(text_column)
        self.feature_extractor = FeatureExtractor(text_column)
        self.char_ngram_extractor = CharNgramFeatureExtractor(column_name=text_column) if use_char_ngrams else None
        self.label_encoder = LabelEncoder()
        
        # New embedding transformer
        self.embedding_extractor = EmbeddingTransformer(
            model_name=embedding_model,
            column_name=text_column
        ) if use_embeddings else None
        
    def fit(self, X, y=None):
        """Fit all transformers including embeddings"""
        if isinstance(X, pd.DataFrame):
            # Clean text
            X_cleaned = X.copy()
            X_cleaned[self.text_column] = self.text_cleaner.fit_transform(X[self.text_column])

            # Remove duplicates
            X_cleaned = self.duplicate_remover.fit_transform(X_cleaned)

            # Extract basic features
            self.feature_extractor.fit(X_cleaned)

            # Fit character n-gram extractor if enabled
            if self.use_char_ngrams and self.char_ngram_extractor is not None:
                self.char_ngram_extractor.fit(X_cleaned)

            # Fit embedding extractor if enabled
            if self.use_embeddings and self.embedding_extractor is not None:
                logger.info("Fitting embedding transformer...")
                self.embedding_extractor.fit(X_cleaned)
                logger.info("Embedding transformer fitted successfully")

            # Encode labels if provided
            if y is not None:
                self.label_encoder.fit(y)

        return self
    
    def transform(self, X, y=None):
        """Transform data including embeddings"""
        if isinstance(X, pd.DataFrame):
            # Clean text
            X_cleaned = X.copy()
            X_cleaned[self.text_column] = self.text_cleaner.transform(X[self.text_column])

            # Don't remove duplicates during inference - only during training
            # This ensures consistent behavior between train and test

            # Extract basic features
            features = self.feature_extractor.transform(X_cleaned)

            # Extract character n-gram features if enabled
            if self.use_char_ngrams and self.char_ngram_extractor is not None:
                char_ngram_features = self.char_ngram_extractor.transform(X_cleaned)
                # Combine features horizontally
                features = pd.concat([features, char_ngram_features], axis=1)

            # Extract embedding features if enabled
            if self.use_embeddings and self.embedding_extractor is not None:
                logger.info("Generating embeddings...")
                embedding_features = self.embedding_extractor.transform(X_cleaned)
                features = pd.concat([features, embedding_features], axis=1)
                logger.info(f"Generated {embedding_features.shape[1]} embedding features")

            # Encode labels if provided
            if y is not None:
                y_encoded = self.label_encoder.transform(y)
                return features, y_encoded

            return features

        return X
    
    def fit_transform(self, X, y=None):
        """Fit and transform in one step including embeddings"""
        if isinstance(X, pd.DataFrame):
            # Clean text
            X_cleaned = X.copy()
            X_cleaned[self.text_column] = self.text_cleaner.fit_transform(X[self.text_column])

            # Remove duplicates
            X_cleaned = self.duplicate_remover.fit_transform(X_cleaned)

            # Extract basic features
            features = self.feature_extractor.fit_transform(X_cleaned)

            # Extract character n-gram features if enabled
            if self.use_char_ngrams and self.char_ngram_extractor is not None:
                char_ngram_features = self.char_ngram_extractor.fit_transform(X_cleaned)
                # Combine features horizontally
                features = pd.concat([features, char_ngram_features], axis=1)

            # Extract embedding features if enabled
            if self.use_embeddings and self.embedding_extractor is not None:
                logger.info("Fitting and generating embeddings...")
                embedding_features = self.embedding_extractor.fit_transform(X_cleaned)
                features = pd.concat([features, embedding_features], axis=1)
                logger.info(f"Generated {embedding_features.shape[1]} embedding features")

            # Encode labels if provided
            if y is not None:
                # Remove corresponding labels for duplicates
                y_series = pd.Series(y)
                y_cleaned = y_series[~self.duplicate_remover.duplicate_indices_]
                y_encoded = self.label_encoder.fit_transform(y_cleaned)
                return features, y_encoded

            return features

        return X
    
    def get_feature_names_out(self, input_features=None):
        """Return all feature names including embeddings"""
        feature_names = list(self.feature_extractor.get_feature_names_out())

        # Add character n-gram feature names if enabled
        if self.use_char_ngrams and self.char_ngram_extractor is not None:
            feature_names.extend(self.char_ngram_extractor.get_feature_names_out())

        # Add embedding feature names if enabled
        if self.use_embeddings and self.embedding_extractor is not None:
            feature_names.extend(self.embedding_extractor.get_feature_names_out())

        return feature_names
    
    def get_label_mapping(self):
        """Return label mapping for inference"""
        return self.label_encoder.label_mapping_
    
    def get_inverse_label_mapping(self):
        """Return inverse label mapping for inference"""
        return self.label_encoder.inverse_mapping_
