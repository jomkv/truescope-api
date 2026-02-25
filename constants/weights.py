from constants.enums import Verdict, SourceBias, NLILabel


def score_to_label(score: float) -> str:
    # Map score to closest verdict label
    mapping = {v: k.value for k, v in VERDICT_WEIGHT_MAP.items()}
    closest = min(mapping.keys(), key=lambda x: abs(score - x))
    return mapping[closest]


VERDICT_WEIGHT_MAP: dict[Verdict, float] = {
    Verdict.TRUE: 1.0,
    Verdict.MOSTLY_TRUE: 0.8,
    Verdict.HALF_TRUE: 0.65,
    Verdict.UNPROVEN: 0.5,
    Verdict.UNKNOWN: 0.5,
    Verdict.MISSING_CONTEXT: 0.45,
    Verdict.MISLEADING: -0.3,
    Verdict.MOSTLY_FALSE: -0.8,
    Verdict.FALSE: -1.0,
    Verdict.OUTDATED: 0.5,
    Verdict.SATIRE: 0.0,
}

# How much influence the source bias should have on the overall verdict confidence
SOURCE_BIAS_WEIGHT_MAP: dict[SourceBias, float] = {
    SourceBias.LEAST_BIASED: 1.0,
    SourceBias.LEFT_CENTER: 0.9,
    SourceBias.RIGHT_CENTER: 0.9,
    SourceBias.LEFT: 0.8,
    SourceBias.RIGHT: 0.8,
    SourceBias.NEUTRAL: 0.7,
}

# Political spectrum positioning for bias divergence calculations
SOURCE_BIAS_SPECTRUM_MAP: dict[SourceBias, int] = {
    SourceBias.LEFT: -2,
    SourceBias.LEFT_CENTER: -1,
    SourceBias.LEAST_BIASED: 0,
    SourceBias.RIGHT_CENTER: 1,
    SourceBias.RIGHT: 2,
    SourceBias.NEUTRAL: 0,
}

NLI_LABEL_WEIGHT_MAP: dict[NLILabel, float] = {
    NLILabel.SUPPORT: 1.0,
    NLILabel.NEUTRAL: 0.5,
    NLILabel.REFUTE: -1.0,
}
