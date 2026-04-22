import re

# Negation PHRASE patterns to detect and strip (order matters — longer first)
NEGATION_PHRASES = [
    "it is false that ",
    "it's false that ",
    "this is false that ",
    "false claim that ",
    "false that ",
    "it is not true that ",
    "it's not true that ",
    "not true that ",
    "no truth to the claim that ",
    "there is no truth that ",
    "there is no evidence that ",
    "no evidence that ",
    "no proof that ",
    "has been debunked that ",
    "debunked: ",
    "debunked — ",
    "misleading claim that ",
    "fabricated claim that ",
    "baseless claim that ",
    "recycled claim that ",
    "dismissed the claim that ",
    "dismisses the claim that ",
]

# Negation WORD/PATTERN pairs — regex to detect inline negations in sentences
# Maps a pattern to a replacement that strips the negation
NEGATION_WORD_PATTERNS = [
    # "was not", "were not", "is not", "are not", "has not", "have not", "did not", "does not"
    (r"\b(was|were|is|are|has|have|had|did|does|will|would|should|could|can)\s+not\b", True),
    # Contractions: wasn't, weren't, isn't, aren't, hasn't, haven't, didn't, doesn't, won't, wouldn't
    (r"\b(wasn't|weren't|isn't|aren't|hasn't|haven't|hadn't|didn't|doesn't|won't|wouldn't|shouldn't|couldn't|can't)\b", True),
    # "never"
    (r"\bnever\b", True),
    # "no [noun]" used at start — e.g. "No proof Marcos..."
    (r"^no\s+(proof|evidence|sign|indication|record|footage|video|photo)\s+", True),
]

# Individual negation tokens for keyword-based polarity checking
NEGATION_TOKENS = {
    "not", "no", "never", "none", "cannot", "isnt", "hasnt", "didnt", "wasnt",
    "baseless", "unbothered", "recycled", "fabricated", "hoax"
}
