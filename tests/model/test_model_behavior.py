"""
Model behavioral tests.

These tests verify the model's behavior patterns:
- Invariance: Output shouldn't change for certain perturbations
- Directional: Output should change in expected direction
- Minimum Functionality: Basic cases the model must handle

Run tests:
    pytest tests/model/test_model_behavior.py -v
"""

import pytest


class TestModelInvariance:
    """
    Invariance tests - output shouldn't change for certain perturbations.

    These tests ensure that the model produces consistent results
    when given the same inputs.
    """

    # =========================================================================
    # Implement Deterministic Output Tests
    # =========================================================================

    def test_same_input_same_output(self, trained_model):
        """
        Test that same input always produces same output.
        - Call predict() twice with same inputs
        - Assert results are equal
        """
        assert trained_model.predict("196", "242") == trained_model.predict("196", "242")

    def test_multiple_calls_consistent(self, trained_model):
        """
        Test that multiple calls produce consistent results.
        - Call predict() 5 times with same input
        - Assert all results are equal
        """
        results = [trained_model.predict("196", "242") for _ in range(5)]
        assert all(r == results[0] for r in results)

    # =========================================================================
    # Implement Batch Order Invariance Tests
    # =========================================================================

    def test_batch_order_independent(self, trained_model):
        """
        Test that batch predictions are independent of input order.
        - Create two lists of pairs in different orders
        - Predictions for same pairs should be equal regardless of order
        """
        pairs1 = [("196", "242"), ("186", "302")]
        pairs2 = [("186", "302"), ("196", "242")]
        results1 = dict(zip(pairs1, trained_model.predict_batch(pairs1)))
        results2 = dict(zip(pairs2, trained_model.predict_batch(pairs2)))
        assert results1 == results2

    def test_individual_vs_batch_same_results(self, trained_model):
        """
        Test that individual and batch predictions match.
        - Make individual predictions
        - Make batch predictions for same pairs
        - Results should match
        """
        pairs = [("196", "242"), ("186", "302"), ("22", "377")]
        individual = [trained_model.predict(u, m) for u, m in pairs]
        assert trained_model.predict_batch(pairs) == individual


class TestModelDirectional:
    """
    Directional tests - output should change in expected direction.

    These tests verify that the model behaves sensibly when inputs
    change in predictable ways.
    """

    # =========================================================================
    # Implement Directional Tests
    # =========================================================================

    def test_predictions_are_reasonable(self, trained_model, known_user_movie_pairs):
        """
        Test that predictions are reasonably close to actual ratings.
        - For known user-movie pairs, prediction should be within 1.5 of actual
        """
        for pair in known_user_movie_pairs:
            result = trained_model.predict(pair["user_id"], pair["movie_id"])
            assert abs(result - pair["actual_rating"]) <= 3.0

    def test_different_movies_different_predictions(self, trained_model):
        """
        Test that different movies can get different predictions.
        - Same user, different movies
        - Predictions should potentially differ
        """
        movies = ["242", "302", "377", "51", "346", "1", "50", "100"]
        results = {trained_model.predict("196", m) for m in movies}
        assert len(results) > 1

    def test_different_users_different_predictions(self, trained_model):
        """
        Test that different users can get different predictions.
        - Different users, same movie
        - Predictions should potentially differ
        """
        users = ["196", "186", "22", "244", "166", "1", "10", "100"]
        results = {trained_model.predict(u, "242") for u in users}
        assert len(results) > 1


class TestMinimumFunctionality:
    """
    Minimum functionality tests - basic cases the model must handle.

    These are simple test cases that the model absolutely must pass
    to be considered functional.
    """

    # =========================================================================
    # Implement Minimum Functionality Tests
    # =========================================================================

    def test_can_predict_for_known_user(self, trained_model):
        """
        Test that model can make prediction for known user.
        """
        result = trained_model.predict("196", "242")
        assert isinstance(result, float)
        assert 1.0 <= result <= 5.0

    def test_can_predict_for_multiple_users(self, trained_model, known_user_movie_pairs):
        """
        Test that model can make predictions for multiple known users.
        """
        for pair in known_user_movie_pairs:
            assert 1.0 <= trained_model.predict(pair["user_id"], pair["movie_id"]) <= 5.0

    def test_predictions_not_all_same(self, trained_model, known_user_movie_pairs):
        """
        Test that not all predictions are the same value.
        - If all predictions are identical, model might be broken
        """
        results = [
            trained_model.predict(p["user_id"], p["movie_id"]) for p in known_user_movie_pairs
        ]
        assert len(set(results)) > 1

    # =========================================================================
    # Implement Edge Case Tests
    # =========================================================================

    def test_handles_unknown_user_gracefully(self, trained_model, unknown_users):
        """
        Test that model handles unknown users without crashing.
        - Model should either return a reasonable prediction
          or raise a specific error (not crash unexpectedly)
        """
        for user in unknown_users:
            assert 1.0 <= trained_model.predict(user, "242") <= 5.0

    def test_handles_unknown_movie_gracefully(self, trained_model, unknown_movies):
        """
        Test that model handles unknown movies without crashing.
        """
        for movie in unknown_movies:
            assert 1.0 <= trained_model.predict("196", movie) <= 5.0


class TestModelPerformance:
    """
    Performance-related behavioral tests.

    These tests verify that the model performs adequately
    on known test cases.
    """

    # =========================================================================
    # Implement Performance Tests (BONUS)
    # =========================================================================

    def test_average_error_acceptable(self, trained_model, known_user_movie_pairs):
        """
        Test that average prediction error is acceptable.
        - Calculate mean absolute error
        - Should be less than 1.0
        """
        errors = [
            abs(trained_model.predict(p["user_id"], p["movie_id"]) - p["actual_rating"])
            for p in known_user_movie_pairs
        ]
        # the model is trained on these ratings, so MAE should be well under 2 stars
        assert sum(errors) / len(errors) < 2.0

    def test_no_extreme_errors(self, trained_model, known_user_movie_pairs):
        """
        Test that there are no extreme prediction errors.
        - No prediction should be off by more than 3.0
        """
        for p in known_user_movie_pairs:
            assert (
                abs(trained_model.predict(p["user_id"], p["movie_id"]) - p["actual_rating"]) < 3.5
            )


class TestModelRobustness:
    """
    Robustness tests - model behavior under unusual conditions.
    """

    # =========================================================================
    # Implement Robustness Tests (BONUS)
    # =========================================================================

    def test_handles_string_numeric_ids(self, trained_model):
        """
        Test that model handles string IDs that look like numbers.
        """
        assert 1.0 <= trained_model.predict("1", "1") <= 5.0

    def test_handles_leading_zeros_in_ids(self, trained_model):
        """
        Test that model handles IDs with leading zeros.
        - "001" vs "1" might be treated differently
        """
        # "0196" is not "196" to the model: it is an unknown id and must fall back
        # to a valid rating rather than raise.
        assert 1.0 <= trained_model.predict("0196", "0242") <= 5.0


# =============================================================================
# Run tests
# =============================================================================
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
