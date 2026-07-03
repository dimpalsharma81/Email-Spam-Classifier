"""Unit tests for the FeatureExtractor class."""

import pytest
import pandas as pd
import tempfile
from pathlib import Path
from scipy.sparse import csr_matrix

from src.feature_extractor import FeatureExtractor
from src.exceptions import FeatureExtractionError


# Fixtures
@pytest.fixture
def sample_messages():
    """
    Create sample messages for testing.
    
    Returns:
        Series of sample messages
    """
    return pd.Series([
        'Win free money now',
        'Hi how are you today',
        'Click here for prizes',
        'Let me know when free',
        'Limited time offer',
        'Hello friend',
        'Great deals here',
        'Contact us today'
    ])


@pytest.fixture
def feature_extractor():
    """
    Create a FeatureExtractor instance.
    
    Returns:
        FeatureExtractor instance
    """
    return FeatureExtractor(
        min_df=1,
        max_df=0.95,
        stop_words='english',
        max_features=None
    )


# Tests
class TestFeatureExtractorInitialization:
    """Tests for FeatureExtractor initialization."""
    
    def test_initialization_default_parameters(self):
        """Test initialization with default parameters."""
        extractor = FeatureExtractor()
        assert extractor.min_df == 1
        assert extractor.max_df == 0.95
        assert extractor.stop_words == 'english'
        assert extractor.max_features is None
        assert extractor.vectorizer is None
    
    def test_initialization_custom_parameters(self):
        """Test initialization with custom parameters."""
        extractor = FeatureExtractor(
            min_df=2,
            max_df=0.8,
            stop_words='english',
            max_features=100
        )
        assert extractor.min_df == 2
        assert extractor.max_df == 0.8
        assert extractor.max_features == 100


class TestFeatureExtractorFitting:
    """Tests for vectorizer fitting."""
    
    def test_fit_successfully(self, feature_extractor, sample_messages):
        """Test successful fitting."""
        result = feature_extractor.fit(sample_messages)
        assert result is feature_extractor  # Check method chaining
        assert feature_extractor.vectorizer is not None
    
    def test_fit_creates_vocabulary(self, feature_extractor, sample_messages):
        """Test that fitting creates vocabulary."""
        feature_extractor.fit(sample_messages)
        vocab_size = len(feature_extractor.vectorizer.vocabulary_)
        assert vocab_size > 0
    
    def test_fit_on_empty_data(self, feature_extractor):
        """Test fitting on empty data."""
        empty_messages = pd.Series([])
        with pytest.raises(FeatureExtractionError):
            feature_extractor.fit(empty_messages)


class TestFeatureTransformation:
    """Tests for feature transformation."""
    
    def test_transform_without_fitting(self, feature_extractor, sample_messages):
        """Test that transform raises error if not fitted."""
        with pytest.raises(FeatureExtractionError):
            feature_extractor.transform(sample_messages)
    
    def test_transform_after_fitting(self, feature_extractor, sample_messages):
        """Test successful transformation after fitting."""
        feature_extractor.fit(sample_messages)
        transformed = feature_extractor.transform(sample_messages)
        
        assert transformed is not None
        assert isinstance(transformed, csr_matrix)
        assert transformed.shape[0] == len(sample_messages)
    
    def test_transform_returns_correct_shape(self, feature_extractor, sample_messages):
        """Test that transform returns correct shape."""
        feature_extractor.fit(sample_messages)
        transformed = feature_extractor.transform(sample_messages)
        
        assert transformed.shape[0] == len(sample_messages)
        assert transformed.shape[1] > 0  # Number of features
    
    def test_transform_single_message(self, feature_extractor, sample_messages):
        """Test transforming a single message."""
        feature_extractor.fit(sample_messages)
        single_message = pd.Series(['Hello world'])
        transformed = feature_extractor.transform(single_message)
        
        assert transformed.shape[0] == 1


class TestFitTransform:
    """Tests for combined fit_transform method."""
    
    def test_fit_transform_successfully(self, feature_extractor, sample_messages):
        """Test successful fit_transform."""
        transformed = feature_extractor.fit_transform(sample_messages)
        
        assert transformed is not None
        assert isinstance(transformed, csr_matrix)
        assert feature_extractor.vectorizer is not None
    
    def test_fit_transform_same_result_as_separate_calls(
        self, feature_extractor, sample_messages
    ):
        """Test that fit_transform gives same result as fit then transform."""
        # Method 1: fit_transform
        extractor1 = FeatureExtractor()
        result1 = extractor1.fit_transform(sample_messages)
        
        # Method 2: fit then transform
        extractor2 = FeatureExtractor()
        extractor2.fit(sample_messages)
        result2 = extractor2.transform(sample_messages)
        
        # Results should be equal
        assert (result1 != result2).nnz == 0  # No differences in sparse matrices


class TestFeatureNames:
    """Tests for getting feature names."""
    
    def test_get_feature_names_without_fitting(self, feature_extractor):
        """Test that get_feature_names raises error if not fitted."""
        with pytest.raises(FeatureExtractionError):
            feature_extractor.get_feature_names()
    
    def test_get_feature_names_after_fitting(
        self, feature_extractor, sample_messages
    ):
        """Test getting feature names after fitting."""
        feature_extractor.fit(sample_messages)
        feature_names = feature_extractor.get_feature_names()
        
        assert len(feature_names) > 0
        assert isinstance(feature_names, (list, tuple))
        # Feature names should be strings
        assert all(isinstance(name, str) for name in feature_names)


class TestVocabularySize:
    """Tests for vocabulary size."""
    
    def test_get_vocabulary_size_without_fitting(self, feature_extractor):
        """Test that get_vocabulary_size raises error if not fitted."""
        with pytest.raises(FeatureExtractionError):
            feature_extractor.get_vocabulary_size()
    
    def test_get_vocabulary_size_after_fitting(
        self, feature_extractor, sample_messages
    ):
        """Test getting vocabulary size after fitting."""
        feature_extractor.fit(sample_messages)
        vocab_size = feature_extractor.get_vocabulary_size()
        
        assert vocab_size > 0
        assert isinstance(vocab_size, int)
    
    def test_vocabulary_size_respects_max_features(
        self, sample_messages
    ):
        """Test that vocabulary size respects max_features parameter."""
        extractor_no_limit = FeatureExtractor(max_features=None)
        extractor_no_limit.fit(sample_messages)
        size_no_limit = extractor_no_limit.get_vocabulary_size()
        
        extractor_with_limit = FeatureExtractor(max_features=5)
        extractor_with_limit.fit(sample_messages)
        size_with_limit = extractor_with_limit.get_vocabulary_size()
        
        assert size_with_limit <= 5
        assert size_with_limit <= size_no_limit


class TestVectorizerPersistence:
    """Tests for saving and loading vectorizer."""
    
    def test_save_without_fitting(self, feature_extractor):
        """Test that save raises error if not fitted."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = f"{tmpdir}/vectorizer.pkl"
            with pytest.raises(FeatureExtractionError):
                feature_extractor.save(filepath)
    
    def test_save_successfully(self, feature_extractor, sample_messages):
        """Test successful vectorizer saving."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = f"{tmpdir}/vectorizer.pkl"
            feature_extractor.fit(sample_messages)
            feature_extractor.save(filepath)
            
            assert Path(filepath).exists()
    
    def test_load_successfully(self, sample_messages):
        """Test successful vectorizer loading."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = f"{tmpdir}/vectorizer.pkl"
            
            # Save
            extractor1 = FeatureExtractor()
            extractor1.fit(sample_messages)
            extractor1.save(filepath)
            
            # Load
            extractor2 = FeatureExtractor()
            result = extractor2.load(filepath)
            assert result is extractor2  # Check method chaining
            assert extractor2.vectorizer is not None
    
    def test_loaded_vectorizer_produces_same_results(
        self, sample_messages
    ):
        """Test that loaded vectorizer produces same results."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = f"{tmpdir}/vectorizer.pkl"
            
            # Original transformation
            extractor1 = FeatureExtractor()
            extractor1.fit(sample_messages)
            result1 = extractor1.transform(sample_messages)
            extractor1.save(filepath)
            
            # Load and transform
            extractor2 = FeatureExtractor()
            extractor2.load(filepath)
            result2 = extractor2.transform(sample_messages)
            
            # Results should be identical
            assert (result1 != result2).nnz == 0


class TestFeatureExtractorIntegration:
    """Integration tests for FeatureExtractor."""
    
    def test_full_workflow(self, feature_extractor, sample_messages):
        """Test complete feature extraction workflow."""
        # Fit
        feature_extractor.fit(sample_messages)
        
        # Get vocabulary info
        vocab_size = feature_extractor.get_vocabulary_size()
        feature_names = feature_extractor.get_feature_names()
        
        # Transform
        transformed = feature_extractor.transform(sample_messages)
        
        # Verify
        assert vocab_size == len(feature_names)
        assert transformed.shape[1] == vocab_size
        assert transformed.shape[0] == len(sample_messages)
    
    def test_different_stop_words_produce_different_vocabularies(
        self, sample_messages
    ):
        """Test that different stop words parameters affect vocabulary size."""
        extractor_with_stop = FeatureExtractor(stop_words='english')
        extractor_with_stop.fit(sample_messages)
        vocab_with_stop = extractor_with_stop.get_vocabulary_size()
        
        extractor_without_stop = FeatureExtractor(stop_words=None)
        extractor_without_stop.fit(sample_messages)
        vocab_without_stop = extractor_without_stop.get_vocabulary_size()
        
        # Without stop words should have larger vocabulary
        assert vocab_without_stop >= vocab_with_stop
