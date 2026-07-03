"""Data processing and preprocessing module for spam classifier."""

import pandas as pd
from pathlib import Path
from typing import Tuple
from sklearn.model_selection import train_test_split

from src.logger import setup_logger
from src.exceptions import DataProcessingError

logger = setup_logger(__name__)


class DataProcessor:
    """
    Handles data loading, cleaning, and preprocessing for the spam classifier.
    
    Attributes:
        data_path: Path to the CSV data file
        test_size: Proportion of data to use for testing
        random_state: Random seed for reproducibility
    """
    
    SPAM_LABEL = 'spam'
    HAM_LABEL = 'ham'
    SPAM_ENCODED = 0
    HAM_ENCODED = 1
    
    def __init__(self, data_path: str, test_size: float = 0.2, random_state: int = 42):
        """
        Initialize DataProcessor.
        
        Args:
            data_path: Path to CSV file with columns 'Category' and 'Message'
            test_size: Fraction of data to use for testing (default: 0.2)
            random_state: Random seed for reproducibility
            
        Raises:
            DataProcessingError: If data_path doesn't exist
        """
        if not Path(data_path).exists():
            raise DataProcessingError(f"Data file not found: {data_path}")
        
        self.data_path = data_path
        self.test_size = test_size
        self.random_state = random_state
        self.df = None
        logger.info(f"DataProcessor initialized with data_path={data_path}")
    
    def load_data(self) -> pd.DataFrame:
        """
        Load data from CSV file.
        
        Returns:
            Loaded DataFrame
            
        Raises:
            DataProcessingError: If loading fails
        """
        try:
            self.df = pd.read_csv(self.data_path)
            logger.info(f"Data loaded successfully. Shape: {self.df.shape}")
            return self.df
        except Exception as e:
            raise DataProcessingError(f"Failed to load data: {str(e)}")
    
    def clean_data(self) -> pd.DataFrame:
        """
        Clean data by handling null values.
        
        Returns:
            Cleaned DataFrame
            
        Raises:
            DataProcessingError: If no data is loaded
        """
        if self.df is None:
            raise DataProcessingError("No data loaded. Call load_data() first.")
        
        try:
            # Replace NaN values with empty string
            self.df = self.df.where((pd.notnull(self.df)), '')
            logger.info(f"Data cleaned. Null values handled.")
            return self.df
        except Exception as e:
            raise DataProcessingError(f"Failed to clean data: {str(e)}")
    
    def encode_labels(self) -> None:
        """
        Encode categorical labels (spam/ham) to numerical (0/1).
        
        Raises:
            DataProcessingError: If encoding fails or required columns missing
        """
        if self.df is None:
            raise DataProcessingError("No data loaded. Call load_data() first.")
        
        try:
            if 'Category' not in self.df.columns:
                raise DataProcessingError("'Category' column not found in data")
            
            self.df.loc[self.df['Category'] == self.SPAM_LABEL, 'Category'] = self.SPAM_ENCODED
            self.df.loc[self.df['Category'] == self.HAM_LABEL, 'Category'] = self.HAM_ENCODED
            self.df['Category'] = self.df['Category'].astype(int)
            logger.info("Labels encoded successfully")
        except Exception as e:
            raise DataProcessingError(f"Failed to encode labels: {str(e)}")
    
    def prepare_features_and_labels(self) -> Tuple[pd.Series, pd.Series]:
        """
        Extract features (messages) and labels (categories) from data.
        
        Returns:
            Tuple of (X, y) where X is messages and y is labels
            
        Raises:
            DataProcessingError: If required columns are missing
        """
        if self.df is None:
            raise DataProcessingError("No data loaded. Call load_data() first.")
        
        try:
            if 'Message' not in self.df.columns or 'Category' not in self.df.columns:
                raise DataProcessingError("Required columns 'Message' or 'Category' not found")
            
            X = self.df['Message']
            y = self.df['Category']
            
            logger.info(f"Features and labels prepared. X shape: {X.shape}, y shape: {y.shape}")
            return X, y
        except Exception as e:
            raise DataProcessingError(f"Failed to prepare features and labels: {str(e)}")
    
    def split_data(self, X: pd.Series, y: pd.Series) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
        """
        Split data into training and testing sets.
        
        Args:
            X: Features (messages)
            y: Labels (categories)
            
        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
            
        Raises:
            DataProcessingError: If split fails
        """
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y,
                test_size=self.test_size,
                random_state=self.random_state,
                stratify=y
            )
            logger.info(
                f"Data split successfully. "
                f"Train: {X_train.shape[0]}, Test: {X_test.shape[0]}"
            )
            return X_train, X_test, y_train, y_test
        except Exception as e:
            raise DataProcessingError(f"Failed to split data: {str(e)}")
    
    def preprocess(self) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
        """
        Full preprocessing pipeline: load -> clean -> encode -> split.
        
        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
            
        Raises:
            DataProcessingError: If any step fails
        """
        try:
            self.load_data()
            self.clean_data()
            self.encode_labels()
            X, y = self.prepare_features_and_labels()
            return self.split_data(X, y)
        except DataProcessingError:
            raise
        except Exception as e:
            raise DataProcessingError(f"Preprocessing failed: {str(e)}")
