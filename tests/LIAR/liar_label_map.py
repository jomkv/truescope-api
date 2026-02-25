from constants.enums import Verdict


# Mapping LIAR dataset string labels to internal Verdict enum
# This is based on the LIAR paper's label definitions and our internal label schema - we may need to adjust based on how our system's output maps to these categories
LIAR_LABEL_TO_VERDICT = {
    "true": Verdict.TRUE,
    "mostly-true": Verdict.MOSTLY_TRUE,
    "half-true": Verdict.HALF_TRUE,
    "barely-true": Verdict.MOSTLY_FALSE,  # LIAR uses 'barely-true'
    "false": Verdict.FALSE,
    "pants-fire": Verdict.MISLEADING,  # extreme false
}


def score_to_verdict(score: float) -> Verdict:
    """
    Maps a system score in range [-1, 1] to internal Verdict enum.

    Assumptions:
    -1.0  -> strongly false
     0.0  -> neutral / mixed
    +1.0  -> strongly true
    """

    if score >= 0.8:
        return Verdict.TRUE

    elif score >= 0.4:
        return Verdict.MOSTLY_TRUE

    elif score > -0.4:
        return Verdict.HALF_TRUE

    elif score > -0.8:
        return Verdict.MOSTLY_FALSE

    elif score > -0.95:
        return Verdict.FALSE

    else:
        return Verdict.MISLEADING
