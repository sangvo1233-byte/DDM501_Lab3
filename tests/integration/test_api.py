"""
Integration tests for API endpoints.

Run tests:
    pytest tests/integration/test_api.py -v
"""

import pytest


class TestHealthEndpoint:
    """Integration tests for /health endpoint."""

    # =========================================================================
    # Provided Tests
    # =========================================================================

    def test_health_returns_200(self, test_client):
        """Test that health endpoint returns 200 status code."""
        response = test_client.get("/health")
        assert response.status_code == 200

    def test_health_response_has_status_field(self, test_client):
        """Test that health response has status field."""
        response = test_client.get("/health")
        data = response.json()
        assert "status" in data

    # =========================================================================
    # Implement Additional Health Tests
    # =========================================================================

    def test_health_response_has_model_loaded_field(self, test_client):
        """
        Test that health response has model_loaded field.
        """
        assert "model_loaded" in test_client.get("/health").json()

    def test_health_model_loaded_is_boolean(self, test_client):
        """
        Test that model_loaded is a boolean value.
        """
        assert isinstance(test_client.get("/health").json()["model_loaded"], bool)


class TestRootEndpoint:
    """Integration tests for / endpoint."""

    def test_root_returns_200(self, test_client):
        """Test that root endpoint returns 200 status code."""
        response = test_client.get("/")
        assert response.status_code == 200

    # =========================================================================
    # Implement Root Endpoint Tests
    # =========================================================================

    def test_root_contains_api_info(self, test_client):
        """
        Test that root response contains API information.
        - Check for 'name', 'version', 'docs' fields
        """
        data = test_client.get("/").json()
        assert {"name", "version", "docs"} <= set(data)


class TestPredictEndpoint:
    """Integration tests for /predict endpoint."""

    # =========================================================================
    # Provided Tests
    # =========================================================================

    def test_predict_valid_request_returns_200(self, test_client, sample_prediction_request):
        """Test that valid prediction request returns 200."""
        response = test_client.post("/predict", json=sample_prediction_request)
        assert response.status_code == 200

    # =========================================================================
    # Implement Response Structure Tests
    # =========================================================================

    def test_predict_response_has_predicted_rating(self, test_client, sample_prediction_request):
        """
        Test that response contains predicted_rating field.
        """
        data = test_client.post("/predict", json=sample_prediction_request).json()
        assert "predicted_rating" in data

    def test_predict_response_has_user_id(self, test_client, sample_prediction_request):
        """
        Test that response contains user_id field.
        """
        data = test_client.post("/predict", json=sample_prediction_request).json()
        assert data["user_id"] == sample_prediction_request["user_id"]

    def test_predict_response_has_movie_id(self, test_client, sample_prediction_request):
        """
        Test that response contains movie_id field.
        """
        data = test_client.post("/predict", json=sample_prediction_request).json()
        assert data["movie_id"] == sample_prediction_request["movie_id"]

    def test_predict_response_rating_in_valid_range(self, test_client, sample_prediction_request):
        """
        Test that predicted_rating is between 1.0 and 5.0.
        """
        data = test_client.post("/predict", json=sample_prediction_request).json()
        assert 1.0 <= data["predicted_rating"] <= 5.0

    # =========================================================================
    # Implement Validation Error Tests
    # =========================================================================

    def test_predict_missing_user_id_returns_422(self, test_client):
        """
        Test that missing user_id returns 422 Unprocessable Entity.
        """
        assert test_client.post("/predict", json={"movie_id": "242"}).status_code == 422

    def test_predict_missing_movie_id_returns_422(self, test_client):
        """
        Test that missing movie_id returns 422.
        """
        assert test_client.post("/predict", json={"user_id": "196"}).status_code == 422

    def test_predict_empty_body_returns_422(self, test_client):
        """
        Test that empty request body returns 422.
        """
        assert test_client.post("/predict", json={}).status_code == 422

    def test_predict_invalid_json_returns_422(self, test_client):
        """
        Test that invalid JSON returns 422.
        """
        response = test_client.post(
            "/predict", content="{not json", headers={"Content-Type": "application/json"}
        )
        assert response.status_code == 422

    # =========================================================================
    # Implement Multiple Request Tests
    # =========================================================================

    def test_predict_multiple_valid_requests(self, test_client, known_user_movie_pairs):
        """
        Test multiple prediction requests all succeed.
        - Loop through known_user_movie_pairs
        - Make prediction request for each
        - Assert all return 200
        """
        for pair in known_user_movie_pairs:
            body = {"user_id": pair["user_id"], "movie_id": pair["movie_id"]}
            assert test_client.post("/predict", json=body).status_code == 200


class TestBatchPredictEndpoint:
    """Integration tests for /predict/batch endpoint."""

    # =========================================================================
    # Implement Batch Prediction Tests
    # =========================================================================

    def test_batch_predict_returns_200(self, test_client, sample_batch_request):
        """
        Test that batch prediction returns 200.
        """
        assert test_client.post("/predict/batch", json=sample_batch_request).status_code == 200

    def test_batch_predict_returns_correct_count(self, test_client, sample_batch_request):
        """
        Test that batch prediction returns correct number of results.
        """
        data = test_client.post("/predict/batch", json=sample_batch_request).json()
        expected = len(sample_batch_request["predictions"])
        assert data["total_count"] == expected
        assert len(data["predictions"]) == expected

    def test_batch_predict_all_ratings_in_range(self, test_client, sample_batch_request):
        """
        Test that all batch predictions are in valid range.
        """
        data = test_client.post("/predict/batch", json=sample_batch_request).json()
        assert all(1.0 <= p["predicted_rating"] <= 5.0 for p in data["predictions"])


class TestErrorHandling:
    """Tests for API error handling."""

    # =========================================================================
    # Implement Error Handling Tests
    # =========================================================================

    def test_404_for_unknown_endpoint(self, test_client):
        """
        Test that unknown endpoint returns 404.
        """
        assert test_client.get("/unknown").status_code == 404

    def test_method_not_allowed_get_predict(self, test_client):
        """
        Test that GET /predict returns 405 Method Not Allowed.
        """
        assert test_client.get("/predict").status_code == 405

    def test_method_not_allowed_post_health(self, test_client):
        """
        Test that POST /health returns 405.
        """
        assert test_client.post("/health").status_code == 405


class TestModelInfoEndpoint:
    """Tests for /model/info endpoint."""

    def test_model_info_returns_200(self, test_client):
        """Test that model info endpoint returns 200."""
        response = test_client.get("/model/info")
        assert response.status_code == 200

    # =========================================================================
    # Implement Model Info Tests
    # =========================================================================

    def test_model_info_has_version(self, test_client):
        """
        Test that model info has version field.
        """
        assert "model_version" in test_client.get("/model/info").json()

    def test_model_info_has_is_loaded(self, test_client):
        """
        Test that model info has is_loaded field.
        """
        assert test_client.get("/model/info").json()["is_loaded"] is True


class TestDegradedService:
    """The API must answer 503/500 cleanly when the model is missing or fails."""

    def test_predict_returns_503_without_model(
        self, test_client, monkeypatch, sample_prediction_request
    ):
        import app.main as main

        monkeypatch.setattr(main, "model", None)
        assert test_client.post("/predict", json=sample_prediction_request).status_code == 503
        assert test_client.get("/health").json()["status"] == "unhealthy"

    def test_batch_returns_503_without_model(self, test_client, monkeypatch, sample_batch_request):
        import app.main as main

        monkeypatch.setattr(main, "model", None)
        assert test_client.post("/predict/batch", json=sample_batch_request).status_code == 503

    def test_predict_returns_500_when_model_raises(
        self, test_client, monkeypatch, sample_prediction_request, sample_batch_request
    ):
        import app.main as main

        class Broken:
            def is_loaded(self):
                return True

            def predict(self, *_):
                raise ValueError("boom")

        monkeypatch.setattr(main, "model", Broken())
        assert test_client.post("/predict", json=sample_prediction_request).status_code == 500
        assert test_client.post("/predict/batch", json=sample_batch_request).status_code == 500


# =============================================================================
# Run tests
# =============================================================================
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
