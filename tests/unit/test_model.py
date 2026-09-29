"""
Unit tests for MovieRatingModel class.

Run tests:
    pytest tests/unit/test_model.py -v
"""

import pytest

from app.model import MovieRatingModel


class TestMovieRatingModel:
    """Unit tests for MovieRatingModel class."""

    # =========================================================================
    # Model Loading Tests
    # =========================================================================

    def test_model_loads_successfully(self, trained_model):
        """Test that model loads without errors."""
        assert trained_model is not None
        assert trained_model.is_loaded()

    def test_model_instance_has_model_attribute(self, trained_model):
        """Test that model instance has the model attribute."""
        assert hasattr(trained_model, "model")
        assert trained_model.model is not None

    # =========================================================================
    # Implement Prediction Return Type Tests
    # =========================================================================

    def test_predict_returns_float(self, trained_model):
        """
        Test that predict() returns a float value.
        - Call trained_model.predict() with valid user_id and movie_id
        - Assert that the result is an instance of float
        """
        result = trained_model.predict("196", "242")
        assert isinstance(result, float)

    # =========================================================================
    # Implement Rating Range Tests
    # =========================================================================

    def test_predict_returns_value_in_valid_range(self, trained_model):
        """
        Test that predictions are within 1-5 range.
        - Call predict() with a valid user-movie pair
        - Assert that result is >= 1.0 and <= 5.0
        """
        result = trained_model.predict("196", "242")
        assert 1.0 <= result <= 5.0

    def test_predict_multiple_pairs_all_in_range(self, trained_model, known_user_movie_pairs):
        """
        Test that all predictions are in valid range.
        - Loop through known_user_movie_pairs
        - For each pair, call predict() and verify range
        """
        for pair in known_user_movie_pairs:
            result = trained_model.predict(pair["user_id"], pair["movie_id"])
            assert 1.0 <= result <= 5.0

    # =========================================================================
    # Implement Batch Prediction Tests
    # =========================================================================

    def test_predict_batch_returns_list(self, trained_model):
        """
        Test that predict_batch() returns a list.
        - Create list of (user_id, movie_id) tuples
        - Call predict_batch()
        - Assert result is a list
        """
        results = trained_model.predict_batch([("196", "242"), ("186", "302")])
        assert isinstance(results, list)

    def test_predict_batch_returns_correct_length(self, trained_model):
        """
        Test that predict_batch() returns correct number of results.
        - Create list of pairs
        - Assert len(results) == len(pairs)
        """
        pairs = [("196", "242"), ("186", "302"), ("22", "377")]
        assert len(trained_model.predict_batch(pairs)) == len(pairs)

    def test_predict_batch_all_values_in_range(self, trained_model):
        """
        Test that all batch predictions are in valid range.
        """
        pairs = [("196", "242"), ("186", "302"), ("22", "377"), ("244", "51")]
        assert all(1.0 <= r <= 5.0 for r in trained_model.predict_batch(pairs))

    # =========================================================================
    # Implement is_loaded() Tests
    # =========================================================================

    def test_is_loaded_returns_bool(self, trained_model):
        """
        Test that is_loaded() returns a boolean.
        """
        assert isinstance(trained_model.is_loaded(), bool)

    def test_is_loaded_returns_true_for_loaded_model(self, trained_model):
        """
        Test that is_loaded() returns True for loaded model.
        """
        assert trained_model.is_loaded() is True

    # =========================================================================
    # Implement Error Handling Tests (BONUS)
    # =========================================================================

    def test_predict_with_none_user_id(self, trained_model):
        """
        Test behavior when user_id is None.
        - This might raise an exception or return a default value
        """
        # Surprise treats an unknown id as "unknown user" and falls back to the
        # global mean, so None degrades gracefully instead of raising.
        result = trained_model.predict(None, "242")
        assert 1.0 <= result <= 5.0

    def test_predict_with_empty_string(self, trained_model):
        """
        Test behavior when IDs are empty strings.
        """
        result = trained_model.predict("", "")
        assert 1.0 <= result <= 5.0


class TestUnloadedModel:
    """A wrapper whose model is missing must refuse to predict."""

    def test_predict_raises_when_not_loaded(self, trained_model):
        unloaded = object.__new__(MovieRatingModel)
        unloaded.model = None
        assert unloaded.is_loaded() is False
        with pytest.raises(RuntimeError):
            unloaded.predict("196", "242")
        with pytest.raises(RuntimeError):
            unloaded.predict_batch([("196", "242")])

    def test_singleton_is_reused_and_resettable(self):
        from app.model import get_model, reset_model

        reset_model()
        first = get_model()
        assert get_model() is first
        reset_model()
        assert get_model() is not first


class TestModelFileHandling:
    """Tests for model file handling."""

    def test_model_raises_error_for_missing_file(self):
        """Test that missing model file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            MovieRatingModel(model_path="/nonexistent/path/model.pkl")

    def test_model_raises_error_for_corrupt_file(self, tmp_path):
        """A file that is not a pickle must raise, not load a broken model."""
        bad = tmp_path / "bad.pkl"
        bad.write_bytes(b"not a pickle")
        with pytest.raises(Exception):
            MovieRatingModel(model_path=str(bad))


# =============================================================================
# Run tests
# =============================================================================
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
