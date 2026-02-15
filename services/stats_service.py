from constants.weights import SOURCE_BIAS_SPECTRUM_MAP


class StatsService:
    """
    Service for calculating core verification metrics:
    1. Bias Consistency - Overall consistency of bias patterns with verdicts
    2. Overall Verdict - Average truth confidence score
    3. Bias Divergence - Ideological spread across sources
    4. Truth Confidence Score - Confidence level in the verdict
    """

    @staticmethod
    def calculate_stats(results: list[dict]) -> dict:
        """
        Calculate 4 core metrics for verification results.

        Args:
            results (list[dict]): List of article result dictionaries

        Returns:
            dict: Contains overall_verdict, bias_divergence, truth_confidence_score,
                  and bias_consistency
        """
        if not results:
            return {
                "overall_verdict": 0,
                "bias_divergence": 0,
                "truth_confidence_score": 0,
                "bias_consistency": 0,
            }

        # 1. Calculate Overall Verdict
        results_with_verdict = [r for r in results if r.get("verdict") is not None]
        overall_verdict = (
            sum(r["verdict"] for r in results_with_verdict) / len(results_with_verdict)
            if results_with_verdict
            else 0
        )

        # 2. Calculate Bias Divergence
        bias_divergence = StatsService.calculate_bias_divergence(results)

        # 3. Calculate Truth Confidence Score
        # Based on consistency of verdicts and NLI confidence
        truth_confidence_score = StatsService.calculate_truth_confidence(results)

        # 4. Calculate Bias Consistency
        bias_consistency = StatsService.calculate_bias_consistency(results)

        return {
            "overall_verdict": round(overall_verdict, 4),
            "bias_divergence": round(bias_divergence, 4),
            "truth_confidence_score": round(truth_confidence_score, 4),
            "bias_consistency": round(bias_consistency, 4),
            "total_processed": len(results),
        }

    @staticmethod
    def calculate_bias_divergence(results: list[dict]) -> float:
        """
        Calculate how ideologically diverse/divergent the sources are.

        Returns 0-1 where:
        - 0 = sources clustered at one ideology (high consensus)
        - 1 = sources spread across spectrum (high divergence)
        """
        biases = [r.get("source_bias") for r in results if r.get("source_bias")]

        if not biases:
            return 0.0

        bias_values = [
            SOURCE_BIAS_SPECTRUM_MAP.get(bias, 0)
            for bias in biases
            if bias in SOURCE_BIAS_SPECTRUM_MAP
        ]

        if len(bias_values) < 2:
            return 0.0

        # Calculate standard deviation
        mean_bias = sum(bias_values) / len(bias_values)
        variance = sum((x - mean_bias) ** 2 for x in bias_values) / len(bias_values)
        std_dev = variance**0.5

        # Normalize to 0-1 scale (max std dev is ~2 for range -2 to 2)
        max_std_dev = 2.0
        bias_divergence = min(std_dev / max_std_dev, 1.0)

        return bias_divergence

    @staticmethod
    def calculate_truth_confidence(results: list[dict]) -> float:
        """
        Calculate confidence in the truth verdict.

        Based on:
        - Consistency of verdicts (low variance = high confidence)
        - NLI confidence scores
        - Number of supporting sources

        Returns 0-1 where 1 is maximum confidence
        """
        if not results:
            return 0.0

        results_with_verdict = [r for r in results if r.get("verdict") is not None]
        results_with_nli = [r for r in results if r.get("nli_result") is not None]

        if not results_with_verdict:
            return 0.0

        # Verdict consistency (low variance = high confidence)
        verdicts = [r["verdict"] for r in results_with_verdict]
        mean_verdict = sum(verdicts) / len(verdicts)
        verdict_variance = sum((v - mean_verdict) ** 2 for v in verdicts) / len(
            verdicts
        )
        verdict_consistency = 1 - (
            verdict_variance / 0.25
        )  # Max variance is 0.25 for 0-1 scale
        verdict_consistency = max(0, min(verdict_consistency, 1))  # Clamp to 0-1

        # NLI confidence
        nli_confidence = 0
        if results_with_nli:
            nli_scores = [
                r["nli_result"]["relationship_confidence"]
                for r in results_with_nli
                if r.get("nli_result")
                and r["nli_result"].get("relationship_confidence") is not None
            ]
            nli_confidence = sum(nli_scores) / len(nli_scores) if nli_scores else 0

        # Combined confidence (50% verdict consistency, 50% NLI confidence)
        truth_confidence = (verdict_consistency * 0.5) + (nli_confidence * 0.5)

        return truth_confidence

    @staticmethod
    def calculate_bias_consistency(results: list[dict]) -> float:
        """
        Calculate overall bias consistency - how well bias patterns align with verdicts.

        Computes a score 0-1 based on how consistently biased sources produce aligned verdicts.
        A high score means biased sources reliably produce verdicts consistent with their bias.

        Returns 0-1 where:
        - 0 = No consistency between bias and verdict
        - 1 = Perfect consistency between bias and verdict
        """
        if not results:
            return 0.0

        consistency_scores = []

        for result in results:
            bias = result.get("source_bias")
            verdict = result.get("verdict")

            # Skip if missing bias or verdict
            if not bias or verdict is None:
                continue

            # Get bias value (-2 to 2)
            bias_value = SOURCE_BIAS_SPECTRUM_MAP.get(bias, 0)

            # Calculate how well verdict aligns with bias direction
            # Bias consistency = how aligned the verdict is with the bias direction
            # Normalize bias_value to 0-1 range and compare with verdict
            bias_normalized = (bias_value + 2) / 4  # Convert -2 to 2 into 0 to 1

            # Consistency = 1 - |difference| between bias direction and verdict
            alignment = 1 - abs(bias_normalized - verdict)
            consistency_scores.append(alignment)

        # Return average consistency
        return (
            sum(consistency_scores) / len(consistency_scores)
            if consistency_scores
            else 0.0
        )
