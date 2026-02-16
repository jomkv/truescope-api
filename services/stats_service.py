from constants.weights import SOURCE_BIAS_SPECTRUM_MAP


class StatsService:
    """
    Service for calculating core verification metrics:
    1. Final Verdict - Overall verdict adjusted by confidence
    2. Overall Verdict - Average truth score across articles
    3. Truth Confidence Score - Confidence level in the verdict
    4. Bias Divergence - Ideological spread across sources
    5. Bias Consistency - Overall consistency of bias patterns with verdicts
    """

    @staticmethod
    def calculate_stats(results: list[dict]) -> dict:
        """
        Calculate core verification metrics.

        Args:
            results (list[dict]): List of article result dictionaries

        Returns:
            dict: Contains final_verdict, overall_verdict, truth_confidence_score,
                  bias_divergence, and bias_consistency on a -1 to 1 scale
        """
        if not results:
            return {
                "final_verdict": 0,
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

        # 5. Calculate Final Verdict using both overall_verdict and truth_confidence_score
        final_verdict = StatsService.calculate_final_verdict(
            overall_verdict, truth_confidence_score
        )

        return {
            "final_verdict": round(final_verdict, 4),
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

        Returns -1 to 1 where:
        - -1 = sources clustered at one ideology (high consensus)
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

        return (bias_divergence * 2) - 1

    @staticmethod
    def calculate_truth_confidence(results: list[dict]) -> float:
        """
        Calculate confidence in the truth verdict.

        Based on:
        - Consistency of verdicts (low variance = high confidence)
        - NLI confidence scores
        - Number of supporting sources

        Returns -1 to 1 where 1 is maximum confidence
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
        verdict_consistency = 1 - (verdict_variance / 1.0)  # Max variance is 1.0
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

        # Map from 0..1 to -1..1 for external consistency
        return (truth_confidence * 2) - 1

    @staticmethod
    def calculate_bias_consistency(results: list[dict]) -> float:
        """
        Calculate overall bias consistency - how well bias patterns align with verdicts.

        Computes a score on a -1 to 1 scale based on how consistently biased sources produce aligned verdicts.
        A high score means biased sources reliably produce verdicts consistent with their bias.

        Returns -1 to 1 where:
        - -1 = No consistency between bias and verdict
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

            # Normalize verdict from -1..1 to 0..1 for alignment comparison
            verdict_normalized = (verdict + 1) / 2

            # Consistency = 1 - |difference| between bias direction and verdict
            alignment = 1 - abs(bias_normalized - verdict_normalized)
            consistency_scores.append(alignment)

        # Return average consistency
        consistency = (
            sum(consistency_scores) / len(consistency_scores)
            if consistency_scores
            else 0.0
        )
        return (consistency * 2) - 1

    @staticmethod
    def calculate_final_verdict(
        overall_verdict: float, truth_confidence_score: float
    ) -> float:
        """
                Calculate final verdict (-1 to 1) using simplified formula.

        Logic:
                - If TCS is low (<0.33): Return neutral (0.0) - unclear case
                - If OV is at extremes (abs(OV) > 0.33):
                    - High TCS (>0.66): Trust OV fully
                    - Mid TCS (0.33-0.66): Lean toward OV (scale down)
                - If OV is in middle (abs(OV) <= 0.33): Pull toward neutral

        Args:
            overall_verdict (float): Average verdict score (-1 to 1)
            truth_confidence_score (float): Confidence in verdict (-1 to 1)

        Returns:
            float: Final verdict (-1 to 1)
        """
        HIGH_THRESHOLD = 0.66
        LOW_THRESHOLD = 0.33
        MID_POINT = 0.0
        OV_EXTREME_THRESHOLD = 0.33

        # Map signed confidence (-1..1) to unsigned (0..1) for thresholding
        truth_confidence_unsigned = (truth_confidence_score + 1) / 2

        # Low confidence → pull to neutral
        if truth_confidence_unsigned < LOW_THRESHOLD:
            return MID_POINT

        # Distance from neutral determines if we're at extremes or in middle
        ov_distance_from_middle = abs(overall_verdict)

        # At extremes (>0.33 from neutral)
        if ov_distance_from_middle > OV_EXTREME_THRESHOLD:
            # High TCS at extremes: trust OV fully
            if truth_confidence_unsigned > HIGH_THRESHOLD:
                return overall_verdict
            # Mid TCS at extremes: lean toward OV
            return overall_verdict * 0.7

        # In middle range: pull toward neutral
        return overall_verdict * 0.5
