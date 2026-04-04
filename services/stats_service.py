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
                non_neutral_avg = sum(v * w for v, w in zip(non_neutral_verdicts, non_neutral_weights)) / sum(non_neutral_weights)
                
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
        bias_consistency = StatsService.calculate_bias_consistency(results)

        return {
            "overall_verdict": round(overall_verdict, 4),
            "bias_divergence": round(bias_divergence, 4),
            "truth_confidence_score": round(truth_confidence_score, 4),
            "bias_consistency": round(bias_consistency, 4),
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

        # Normalize to 0-1 scale (max std dev is ~2 for range -2 to 2)
        max_std_dev = 2.0
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
    def calculate_bias_consistency(results: list[ArticleResultModel]) -> float:
        """
        Calculate Cross-Spectrum Consensus - how well sources with different biases agree on the verdict.
        
        A high score means sources from across the ideological spectrum (e.g., Left and Right) 
        are producing the SAME verdict, indicating a strong, bias-independent consensus.
        
        Returns -1 to 1 where:
        - 1 = Perfect consensus across different biases
        - 0 = Neutral / Mixed signals
        - -1 = High polarization (sources agree only within their own bias groups)
        """
        if not results or len(results) < 2:
            return 0.0

        # Group results by verdict polarity (Support, Refute, Neutral)
        # We use a small threshold to avoid 0.0001 being treated as non-neutral
        support_group = []
        refute_group = []
        
        for r in results:
            if r.verdict is None:
                continue
            
            bias_val = SOURCE_BIAS_SPECTRUM_MAP.get(r.source_bias, 0)
            
            if r.verdict > 0.1:
                support_group.append(bias_val)
            elif r.verdict < -0.1:
                refute_group.append(bias_val)

        def calculate_group_consensus(bias_values: list[float]) -> float:
            if len(bias_values) < 2:
                return 0.0
            
            # Consensus is high if the spread (std dev) of biases is high
            # i.e., different types of sources agree.
            mean_bias = sum(bias_values) / len(bias_values)
            variance = sum((x - mean_bias) ** 2 for x in bias_values) / len(bias_values)
            std_dev = variance ** 0.5
            
            # Normalize: max std dev is ~2.0 for range [-2, 2]
            return min(std_dev / 1.5, 1.0) # Use 1.5 as "high diversity" threshold

        support_consensus = calculate_group_consensus(support_group)
        refute_consensus = calculate_group_consensus(refute_group)
        
        # Weighted average based on group sizes
        total_non_neutral = len(support_group) + len(refute_group)
        if total_non_neutral == 0:
            return 0.0
            
        consensus_score = (
            (support_consensus * len(support_group)) + 
            (refute_consensus * len(refute_group))
        ) / total_non_neutral

        # Map 0..1 to -1..1
        return (consensus_score * 2) - 1

    @staticmethod
    def map_to_percentage(value: float) -> float:
        """
        Maps a -1 to 1 metric to a 0 to 100 percentage.
        Used for thesis reporting (Accuracy, Consistency, etc).
        """
        return round(((value + 1) / 2) * 100, 2)
