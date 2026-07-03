"""Unit tests for the DataProcessor class."""

import pytest
import pandas as pd
import tempfile
from pathlib import Path

from src.data_processor import DataProcessor
from src.exceptions import DataProcessingError


# Fixtures
@pytest.fixture
def sample_data():
    """
    Create sample data for testing.
    
    Returns:
        Path to temporary CSV file with sample data
    """
    data = {
        'Category': ['spam', 'ham', 'spam', 'ham', 'spam'],
        'Message': [
            'Win free money now!',
            'Hi, how are you?',
            'Click here for prizes',
            'Let me know when you are free',
            'Limited time offer!!!'
        ]
    }
    df = pd.DataFrame(data)
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        df.to_csv(f.name, index=False)
        return f.name


@pytest.fixture
def data_processor(sample_data):
    """
    Create a DataProcessor instance with sample data.
    
    Returns:
        DataProcessor instance
    """
    return DataProcessor(data_path=sample_data, test_size=0.2, random_state=42)


# Tests
class TestDataProcessorInitialization:
    """Tests for DataProcessor initialization."""
    
    def test_initialization_with_valid_file(self, sample_data):
        """Test initialization with valid data file."""
        processor = DataProcessor(data_path=sample_data)
        assert processor.data_path == sample_data
        assert processor.test_size == 0.2
        assert processor.random_state == 42
    
    def test_initialization_with_invalid_file(self):
        """Test initialization with non-existent file raises error."""
        with pytest.raises(DataProcessingError):
            DataProcessor(data_path='non_existent_file.csv')
    
    def test_initialization_with_custom_parameters(self, sample_data):
        """Test initialization with custom parameters."""
        processor = DataProcessor(data_path=sample_data, test_size=0.3, random_state=123)
        assert processor.test_size == 0.3
        assert processor.random_state == 123


class TestDataLoading:
    """Tests for data loading."""
    
    def test_load_data_successfully(self, data_processor, sample_data):
        """Test successful data loading."""
        df = data_processor.load_data()
        assert df is not None
        assert len(df) == 5
        assert list(df.columns) == ['Category', 'Message']
    
    def test_load_data_stores_df(self, data_processor):
        """Test that load_data stores the dataframe internally."""
        data_processor.load_data()
        assert data_processor.df is not None
        assert isinstance(data_processor.df, pd.DataFrame)


class TestDataCleaning:
    """Tests for data cleaning."""
    
    def test_clean_data_without_loading(self, data_processor):
        """Test that clean_data raises error if no data loaded."""
        with pytest.raises(DataProcessingError):
            data_processor.clean_data()
    
    def test_clean_data_successfully(self, data_processor):
        """Test successful data cleaning."""
        data_processor.load_data()
        df = data_processor.clean_data()
        assert df is not None
        assert df.isnull().sum().sum() == 0


class TestLabelEncoding:
    """Tests for label encoding."""
    
    def test_encode_labels_without_loading(self, data_processor):
        """Test that encode_labels raises error if no data loaded."""
        with pytest.raises(DataProcessingError):
            data_processor.encode_labels()
    
    def test_encode_labels_successfully(self, data_processor):
        """Test successful label encoding."""
        data_processor.load_data()
        data_processor.encode_labels()
        
        # Check that labels are encoded as 0 and 1
        unique_labels = set(data_processor.df['Category'].unique())
        assert unique_labels == {0, 1}
    
    def test_encode_labels_correctness(self, data_processor):
        """Test that spam maps to 0 and ham maps to 1."""
        data_processor.load_data()
        data_processor.encode_labels()
        
        # Original first row was 'spam', should be 0
        assert data_processor.df.iloc[0]['Category'] == 0
        # Original second row was 'ham', should be 1
        assert data_processor.df.iloc[1]['Category'] == 1


class TestFeaturePreparation:
    """Tests for feature and label preparation."""
    
    def test_prepare_features_without_loading(self, data_processor):
        """Test that prepare_features_and_labels raises error if no data loaded."""
        with pytest.raises(DataProcessingError):
            data_processor.prepare_features_and_labels()
    
    def test_prepare_features_successfully(self, data_processor):
        """Test successful feature and label preparation."""
        data_processor.load_data()
        data_processor.clean_data()
        data_processor.encode_labels()
        
        X, y = data_processor.prepare_features_and_labels()
        
        assert isinstance(X, pd.Series)
        assert isinstance(y, pd.Series)
        assert len(X) == len(y) == 5
    
    def test_prepare_features_correct_shapes(self, data_processor):
        """Test that prepared features have correct shapes."""
        data_processor.load_data()
        data_processor.clean_data()
        data_processor.encode_labels()
        
        X, y = data_processor.prepare_features_and_labels()
        
        assert X.shape == (5,)
        assert y.shape == (5,)


class TestDataSplitting:
    """Tests for data splitting."""
    
    def test_split_data_successfully(self, data_processor):
        """Test successful data splitting."""
        X = pd.Series(['msg1', 'msg2', 'msg3', 'msg4', 'msg5'])
        y = pd.Series([0, 1, 0, 1, 0])
        
        X_train, X_test, y_train, y_test = data_processor.split_data(X, y)
        
        assert len(X_train) + len(X_test) == len(X)
        assert len(y_train) + len(y_test) == len(y)
    
    def test_split_data_respects_test_size(self, data_processor):
        """Test that split respects test_size parameter."""
        X = pd.Series(['msg' + str(i) for i in range(100)])
        y = pd.Series([i % 2 for i in range(100)])
        
        X_train, X_test, y_train, y_test = data_processor.split_data(X, y)
        
        # With test_size=0.2, test should have ~20 samples
        assert 15 <= len(X_test) <= 25
        assert 75 <= len(X_train) <= 85


class TestPreprocessingPipeline:
    """Tests for the full preprocessing pipeline."""
    
    def test_preprocess_successfully(self, data_processor):
        """Test successful full preprocessing."""
        X_train, X_test, y_train, y_test = data_processor.preprocess()
        
        assert isinstance(X_train, pd.Series)
        assert isinstance(X_test, pd.Series)
        assert isinstance(y_train, pd.Series)
        assert isinstance(y_test, pd.Series)
        
        assert len(X_train) + len(X_test) == 5
        assert len(y_train) + len(y_test) == 5
    
    def test_preprocess_returns_correct_types(self, data_processor):
        """Test that preprocess returns correct data types."""
        X_train, X_test, y_train, y_test = data_processor.preprocess()
        
        # Messages should be strings
        assert all(isinstance(msg, str) for msg in X_train)
        assert all(isinstance(msg, str) for msg in X_test)
        
        # Labels should be integers
        assert y_train.dtype in [int, 'int64']
        assert y_test.dtype in [int, 'int64']
    
    def test_preprocess_no_data_leakage(self, data_processor):
        """Test that there's no data leakage between train and test."""
        X_train, X_test, y_train, y_test = data_processor.preprocess()
        
        # No message should appear in both train and test
        train_messages = set(X_train.values)
        test_messages = set(X_test.values)
        
        assert len(train_messages & test_messages) == 0
