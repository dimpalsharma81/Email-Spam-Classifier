"""Feature extraction and engineering module for spam classifier."""

import pandas as pd
from typing import Optional
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
from pathlib import Path

from src.logger import setup_logger
from src.exceptions import FeatureExtractionError

logger = setup_logger(__name__)


class FeatureExtractor:
    """
    Handles feature extraction from text messages using TF-IDF vectorization.
    
    Attributes:
        min_df: Minimum document frequency
        max_df: Maximum document frequency
        stop_words: Stop words to use
        max_features: Maximum number of features
        vectorizer: Fitted TfidfVectorizer instance
    """
    
    def __init__(
        self,
        min_df: int = 1,
        max_df: float = 0.95,
        stop_words: str = 'english',
        max_features: Optional[int] = None
    ):
        """
        Initialize FeatureExtractor.
        
        Args:
            min_df: Minimum document frequency (default: 1)
            max_df: Maximum document frequency (default: 0.95)
            stop_words: Stop words to filter (default: 'english')
            max_features: Maximum number of features to extract (default: None)
        """
        self.min_df = min_df
        self.max_df = max_df
        self.stop_words = stop_words
        self.max_features = max_features
        self.vectorizer = None
        logger.info(
            f"FeatureExtractor initialized with min_df={min_df}, "
            f"max_df={max_df}, max_features={max_features}"
        )
    
    def fit(self, X_train: pd.Series) -> 'FeatureExtractor':
        """
        Fit the vectorizer on training data.
        
        Args:
            X_train: Training messages
            
        Returns:
            Self for method chaining
            
        Raises:
            FeatureExtractionError: If fitting fails
        """
        try:
            self.vectorizer = TfidfVectorizer(
                min_df=self.min_df,
                max_df=self.max_df,
                stop_words=self.stop_words,
                max_features=self.max_features,
                lowercase=True
            )
            self.vectorizer.fit(X_train)
            logger.info(
                f"Vectorizer fitted successfully. "
                f"Vocabulary size: {len(self.vectorizer.vocabulary_)}"
            )
            return self
        except Exception as e:
            raise FeatureExtractionError(f"Failed to fit vectorizer: {str(e)}")
    
    def transform(self, X):
        """
        Transform text data to TF-IDF features.
        
        Args:
            X: Messages to transform
            
        Returns:
            Sparse matrix of TF-IDF features
            
        Raises:
            FeatureExtractionError: If vectorizer not fitted or transform fails
        """
        if self.vectorizer is None:
            raise FeatureExtractionError("Vectorizer not fitted. Call fit() first.")
        
        try:
            features = self.vectorizer.transform(X)
            logger.info(f"Transformed {len(X)} messages to features")
            return features
        except Exception as e:
            raise FeatureExtractionError(f"Failed to transform data: {str(e)}")
    
    def fit_transform(self, X_train: pd.Series):
        """
        Fit vectorizer and transform training data in one step.
        
        Args:
            X_train: Training messages
            
        Returns:
            Sparse matrix of TF-IDF features
            
        Raises:
            FeatureExtractionError: If fit or transform fails
        """
        try:
            self.fit(X_train)
            return self.transform(X_train)
        except FeatureExtractionError:
            raise
        except Exception as e:
            raise FeatureExtractionError(f"Failed fit_transform: {str(e)}")
    
    def get_feature_names(self):
        """
        Get the names of all extracted features.
        
        Returns:
            Array of feature names
            
        Raises:
            FeatureExtractionError: If vectorizer not fitted
        """
        if self.vectorizer is None:
            raise FeatureExtractionError("Vectorizer not fitted. Call fit() first.")
        
        try:
            return self.vectorizer.get_feature_names_out()
        except Exception as e:
            raise FeatureExtractionError(f"Failed to get feature names: {str(e)}")
    
    def get_vocabulary_size(self) -> int:
        """
        Get the size of the vocabulary.
        
        Returns:
            Number of features in vocabulary
            
        Raises:
            FeatureExtractionError: If vectorizer not fitted
        """
        if self.vectorizer is None:
            raise FeatureExtractionError("Vectorizer not fitted. Call fit() first.")
        
        return len(self.vectorizer.vocabulary_)
    
    def save(self, filepath: str) -> None:
        """
        Save the fitted vectorizer to disk.
        
        Args:
            filepath: Path to save the vectorizer
            
        Raises:
            FeatureExtractionError: If save fails or vectorizer not fitted
        """
        if self.vectorizer is None:
            raise FeatureExtractionError("Vectorizer not fitted. Cannot save.")
        
        try:
            Path(filepath).parent.mkdir(parents=True, exist_ok=True)
            joblib.dump(self.vectorizer, filepath)
            logger.info(f"Vectorizer saved to {filepath}")
        except Exception as e:
            raise FeatureExtractionError(f"Failed to save vectorizer: {str(e)}")
    
    def load(self, filepath: str) -> 'FeatureExtractor':
        """
        Load a fitted vectorizer from disk.
        
        Args:
            filepath: Path to load the vectorizer from
            
        Returns:
            Self for method chaining
            
        Raises:
            FeatureExtractionError: If load fails
        """
        try:
            self.vectorizer = joblib.load(filepath)
            logger.info(f"Vectorizer loaded from {filepath}")
            return self
        except Exception as e:
            raise FeatureExtractionError(f"Failed to load vectorizer: {str(e)}")
