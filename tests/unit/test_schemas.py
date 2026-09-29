"""
Unit tests for Pydantic schemas.

Run tests:
    pytest tests/unit/test_schemas.py -v
"""

import pytest
from pydantic import ValidationError

from app.schemas import (
    BatchPredictionRequest,
    HealthResponse,
    PredictionItem,
    PredictionRequest,
    PredictionResponse,
)


class TestPredictionRequest:
    """Tests for PredictionRequest schema."""

    # =========================================================================
    # Valid Input Tests (PROVIDED)
    # =========================================================================

    def test_valid_request(self):
        """Test that valid request passes validation."""
        request = PredictionRequest(user_id="196", movie_id="242")
        assert request.user_id == "196"
        assert request.movie_id == "242"

    def test_valid_request_with_numeric_strings(self):
        """Test numeric string IDs are valid."""
        request = PredictionRequest(user_id="123", movie_id="456")
        assert request.user_id == "123"
        assert request.movie_id == "456"

    # =========================================================================
    # Implement Missing Field Tests
    # =========================================================================

    def test_missing_user_id_raises_error(self):
        """
        Test that missing user_id raises ValidationError.
        - Use pytest.raises(ValidationError)
        - Try to create PredictionRequest without user_id
        """
        with pytest.raises(ValidationError):
            PredictionRequest(movie_id="242")

    def test_missing_movie_id_raises_error(self):
        """
        Test that missing movie_id raises ValidationError.
        """
        with pytest.raises(ValidationError):
            PredictionRequest(user_id="196")

    def test_missing_both_fields_raises_error(self):
        """
        Test that missing both fields raises ValidationError.
        """
        with pytest.raises(ValidationError):
            PredictionRequest()

    # =========================================================================
    # Implement Empty/Invalid Input Tests
    # =========================================================================

    def test_empty_user_id_raises_error(self):
        """
        Test that empty user_id raises ValidationError.
        """
        with pytest.raises(ValidationError):
            PredictionRequest(user_id="", movie_id="242")

    def test_whitespace_only_user_id_raises_error(self):
        """
        Test that whitespace-only user_id raises ValidationError.
        """
        with pytest.raises(ValidationError):
            PredictionRequest(user_id="   ", movie_id="242")

    def test_none_values_raise_error(self):
        """
        Test that None values raise ValidationError.
        """
        with pytest.raises(ValidationError):
            PredictionRequest(user_id=None, movie_id=None)

    # =========================================================================
    # Implement Type Validation Tests
    # =========================================================================

    def test_integer_user_id_converted_to_string(self):
        """
        Test how integer user_id is handled.
        - Pydantic may convert int to string or raise error
        - Test the actual behavior
        """
        # Pydantic v2 does not coerce int -> str by default: the contract says
        # IDs are strings, so an integer is rejected rather than silently cast.
        with pytest.raises(ValidationError):
            PredictionRequest(user_id=196, movie_id="242")


class TestPredictionResponse:
    """Tests for PredictionResponse schema."""

    # =========================================================================
    # Implement Response Validation Tests
    # =========================================================================

    def test_valid_response(self):
        """
        Test that valid response passes validation.
        """
        response = PredictionResponse(
            user_id="196", movie_id="242", predicted_rating=3.5, model_version="1.0.0"
        )
        assert response.predicted_rating == 3.5

    def test_rating_below_minimum_raises_error(self):
        """
        Test that rating below 1.0 raises ValidationError.
        """
        with pytest.raises(ValidationError):
            PredictionResponse(
                user_id="196", movie_id="242", predicted_rating=0.5, model_version="1.0.0"
            )

    def test_rating_above_maximum_raises_error(self):
        """
        Test that rating above 5.0 raises ValidationError.
        """
        with pytest.raises(ValidationError):
            PredictionResponse(
                user_id="196", movie_id="242", predicted_rating=5.5, model_version="1.0.0"
            )

    def test_rating_at_boundaries(self):
        """
        Test ratings at exact boundaries (1.0 and 5.0).
        """
        for rating in (1.0, 5.0):
            response = PredictionResponse(
                user_id="196", movie_id="242", predicted_rating=rating, model_version="1.0.0"
            )
            assert response.predicted_rating == rating


class TestHealthResponse:
    """Tests for HealthResponse schema."""

    # =========================================================================
    # Implement Health Response Tests
    # =========================================================================

    def test_valid_health_response(self):
        """
        Test that valid health response passes validation.
        """
        health = HealthResponse(status="healthy", model_loaded=True)
        assert health.status == "healthy"
        assert health.model_loaded is True

    def test_health_response_status_types(self):
        """
        Test various status values.
        """
        for status, loaded in (("healthy", True), ("unhealthy", False)):
            health = HealthResponse(status=status, model_loaded=loaded)
            assert health.status == status
            assert health.model_loaded is loaded


class TestBatchPredictionRequest:
    """Tests for BatchPredictionRequest schema."""

    # =========================================================================
    # Implement Batch Request Tests (BONUS)
    # =========================================================================

    def test_valid_batch_request(self):
        """
        Test that valid batch request passes validation.
        """
        batch = BatchPredictionRequest(predictions=[PredictionItem(user_id="196", movie_id="242")])
        assert len(batch.predictions) == 1

    def test_empty_predictions_list_raises_error(self):
        """
        Test that empty predictions list raises ValidationError.
        """
        with pytest.raises(ValidationError):
            BatchPredictionRequest(predictions=[])

    def test_too_many_predictions_raises_error(self):
        """
        Test that too many predictions raises ValidationError.
        - The schema has max_length=100
        """
        items = [{"user_id": "1", "movie_id": "1"}] * 101
        with pytest.raises(ValidationError):
            BatchPredictionRequest(predictions=items)


# =============================================================================
# Run tests
# =============================================================================
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
