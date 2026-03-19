# Minimum length for stem matching (e.g. "aggress" -> "aggressive")
MIN_STEM_MATCH_LENGTH = 5

# Demonym groups and related terms for fuzzy matching
# Each set contains terms that should be treated as equivalent
DEMONYM_GROUPS = [
    {"china", "chinese"},
    {"philippine", "philippines", "filipino", "filipinos", "pinoy", "ph"},
    {"america", "american", "us", "usa", "united states"},
    {"japan", "japanese"},
    {"korea", "korean", "south korea", "north korea"},
    {"russia", "russian"},
    {"britain", "british", "uk", "united kingdom", "england", "english"},
    {"france", "french"},
    {"germany", "german"},
    {"italy", "italian"},
    {"canada", "canadian"},
    {"australia", "australian"},
    {"india", "indian"},
    {"mexico", "mexican"},
    {"brazil", "brazilian"},
    {"israel", "israeli"},
    {"palestine", "palestinian"},
    {"ukraine", "ukrainian"},
    {"taiwan", "taiwanese"},
    {"vietnam", "vietnamese"},
]

# Common suffixes for plural/variant matching
COMMON_PLURAL_SUFFIXES = ["s", "es", "ies"]

# Antonym pairs for detecting polarity mismatches in quantitative/directional claims
ANTONYM_PAIRS = [
    ("high", "low"),
    ("increase", "decrease"),
    ("increased", "decreased"),
    ("increasing", "decreasing"),
    ("rise", "fall"),
    ("rising", "falling"),
    ("more", "less"),
    ("above", "below"),
    ("up", "down"),
    ("support", "against"),
    ("supports", "against"),
    ("supporting", "against"),
    ("legal", "illegal"),
    ("true", "false"),
    ("win", "lose"),
    ("won", "lost"),
    ("beaten", "lost"),
]
