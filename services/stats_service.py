from constants.weights import SOURCE_BIAS_SPECTRUM_MAP
from models.article_result_model import ArticleResultModel


class StatsService:
    """
    Service for calculating core verification metrics:
    1. Overall Verdict - Average truth score across articles
    2. Truth Confidence Score - Confidence level in the verdict
    3. Bias Divergence - Ideological spread across sources
    4. Bias Consistency - Overall consistency of bias patterns with verdicts
    """

    @staticmethod
    def calculate_stats(results: list[ArticleResultModel]) -> dict:
        """
        Calculate core verification metrics.

        Args:
            results (list[dict]): List of article result dictionaries

        Returns:
            dict: Contains overall_verdict, truth_confidence_score,
                  bias_divergence, and bias_consistency on a -1 to 1 scale
        """
        if not results:
            return {
                "overall_verdict": 0,
                "bias_divergence": 0,
                "truth_confidence_score": 0,
                "bias_consistency": 0,
            }

        # 1. Calculate Overall Verdict using NLI-confidence-weighted average.
        # This ensures high-confidence NLI results have more influence on the final
        # verdict than low-confidence ones, preventing weak evidence from distorting outcomes.
        verdicts = []
        weights = []
        for r in results:
            if r.verdict is not None:
                verdicts.append(r.verdict)
                # Use NLI confidence as weight if available, otherwise use 0.5 as default
                nli_conf = (
                    r.nli_result.relationship_confidence
                    if r.nli_result and r.nli_result.relationship_confidence is not None
                    else 0.5
                )
                weights.append(nli_conf)

        if verdicts:
            total_weight = sum(weights)
            overall_verdict = (
                sum(v * w for v, w in zip(verdicts, weights)) / total_weight
                if total_weight > 0
                else 0
            )

            # --- Neutral Dampening Logic ---
            # If the weighted average is near-neutral but contains strong signals,
            # ensure that a high-confidence Support or Refute isn't completely
            # washed out by a sea of low-signal Neutral evidence.
            # We calculate an "evidence-only" average (ignoring 0.0s) to see if
            # there's a clear consensus among non-neutral sources.
            non_neutral_verdicts = [v for v in verdicts if v != 0]
            non_neutral_weights = [w for v, w in zip(verdicts, weights) if v != 0]

            if non_neutral_verdicts:
                non_neutral_avg = sum(
                    v * w for v, w in zip(non_neutral_verdicts, non_neutral_weights)
                ) / sum(non_neutral_weights)

                # If there's clear non-neutral evidence, pull the overall verdict
                # towards it, reducing the "gravity" of the 0.0 Neutral results.
                # We blend the overall average with the non-neutral average (70/30)
                # if the non-neutral signal is strong.
                if abs(non_neutral_avg) > 0.4:
                    overall_verdict = (overall_verdict * 0.6) + (non_neutral_avg * 0.4)
        else:
            overall_verdict = 0

        # 2. Calculate Bias Divergence
        bias_divergence = StatsService.calculate_bias_divergence(results)

        # 3. Calculate Truth Confidence Score
        # Based on consistency of verdicts and NLI confidence
        truth_confidence_score = StatsService.calculate_truth_confidence(results)

        # 4. Calculate Bias Consistency
        bias_consistency, raw_r = StatsService.calculate_bias_consistency(results)

        return {
            "overall_verdict": round(overall_verdict, 4),
            "bias_divergence": round(bias_divergence, 4),
            "truth_confidence_score": round(truth_confidence_score, 4),
            "bias_consistency": round(bias_consistency, 4),
            "pearson_r": round(raw_r, 4),
            "total_processed": len(results),
        }

    @staticmethod
    def calculate_bias_divergence(results: list[ArticleResultModel]) -> float:
        """
        Calculate how ideologically diverse/divergent the sources are.

        Returns -1 to 1 where:
        - -1 = sources clustered at one ideology (high consensus)
        - 1 = sources spread across spectrum (high divergence)
        """
        biases = [r.source_bias for r in results if r.source_bias]

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

        # Normalize to 0-1 scale
        # Max standard deviation with our current midpoint extremities (-4.5 to 4.5) is 4.5
        max_std_dev = 4.5
        bias_divergence = min(std_dev / max_std_dev, 1.0)

        return (bias_divergence * 2) - 1

    @staticmethod
    def calculate_truth_confidence(results: list[ArticleResultModel]) -> float:
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

        # Verdict consistency (low variance = high confidence)
        results_with_nli = [r for r in results if r.nli_result is not None]
        verdicts = [r.verdict for r in results if r.verdict is not None]

        if not verdicts:
            return 0.0

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
                r.nli_result.relationship_confidence
                for r in results_with_nli
                if r.nli_result and r.nli_result.relationship_confidence is not None
            ]
            nli_confidence = sum(nli_scores) / len(nli_scores) if nli_scores else 0

        # Combined confidence (50% verdict consistency, 50% NLI confidence)
        truth_confidence = (verdict_consistency * 0.5) + (nli_confidence * 0.5)

        # Map from 0..1 to -1..1 for external consistency
        return (truth_confidence * 2) - 1

    @staticmethod
    def calculate_bias_consistency(
        results: list[ArticleResultModel],
    ) -> tuple[float, float]:
        """
        Calculate overall bias consistency - how well bias patterns align with verdicts.

        Uses the Absolute Pearson Correlation Coefficient to determine if a source's
        political ideology reliably predicts the verdict it will give.

        Returns a tuple (consistency, raw_r) where:
        - consistency: -1 to 1 (how polarized/predictable the topic is)
        - raw_r: -1 to 1 (the raw Pearson Correlation Coefficient)
        """
        if not results:
            return 0.0

        bias_values = []
        verdicts = []

        for result in results:
            bias = result.source_bias
            verdict = result.verdict

            # Skip if missing bias or verdict
            if not bias or verdict is None:
                continue

            bias_value = SOURCE_BIAS_SPECTRUM_MAP.get(bias, 0.0)
            bias_values.append(bias_value)
            verdicts.append(verdict)

        n = len(bias_values)
        if n < 2:
            return -1.0, 0.0

        mean_bias = sum(bias_values) / n
        mean_verdict = sum(verdicts) / n

        var_bias = sum((x - mean_bias) ** 2 for x in bias_values)
        var_verdict = sum((y - mean_verdict) ** 2 for y in verdicts)

        # If there is no variance in bias (all checking sources share the same ideology)
        # or no variance in verdicts (everyone perfectly agrees), correlation is undefined.
        # This implies bias is NOT dividing the results.
        if var_bias == 0 or var_verdict == 0:
            # Return -1 to represent "No bias-driven polarization/consistency"
            return -1.0, 0.0

        cov_xy = sum(
            (x - mean_bias) * (y - mean_verdict) for x, y in zip(bias_values, verdicts)
        )

        # Pearson Correlation Coefficient (-1 to 1)
        r = cov_xy / ((var_bias * var_verdict) ** 0.5)

        # We care about the absolute consistency. Both perfect negative correlation and
        # perfect positive correlation indicate that the ideology is highly polarized regarding the topic.
        abs_r = abs(r)

        # Map absolute correlation from [0, 1] to the [-1, 1] scale
        return (abs_r * 2) - 1, r
