"""
Keywords, patterns, and constants for entity extraction and text processing.
"""

# Organization patterns
ORGANIZATION_PATTERNS = [
    r'\b(Iglesia ni Cristo|INC)\b',
    r'\b([A-Z][a-z]+(?: [A-Z][a-z]+)*)(?:\s+(?:Church|Foundation|Organization|Coalition|Party|Alliance|Movement|University|Institute|Commission|Department|Office|Bureau|Agency|Council))\b',
    r'\b(?:BAYAN|CBCP|UP)\b',
    r'\b(?:Philippine Coast Guard|PCG|China Coast Guard|CCG|Chinese Coast Guard|Coast Guard)\b',
    r'\b(?:Philippine Navy|Armed Forces of the Philippines|AFP|Philippine Marines|Bureau of Fisheries and Aquatic Resources|BFAR)\b',
]

# People patterns
PEOPLE_PATTERNS = [
    r'\b(?:President|Vice President|Senator|Congressman|Rep\.|Cardinal|Archbishop|Bishop|Father|Pastor|Minister|Secretary|Chairman|Director|Spokesperson|Spox)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z\.]+)+)\b',
    r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z\.]+)+)\s+(?:said|announced|stated|claimed|reported|confirmed)\b',
    r'\b(fishermen|fisherfolk|fishers|sailors|troops|personnel|crew|crewmen)\b',
    r'^([A-Z][a-z]+)\b',  # Standalone proper noun at start (captures "Trump", "Biden", "Duterte")
    r'\b([A-Z][a-z]+)\s+(?:banned|ordered|signed|announced|declared|issued|enacted|implemented)\b',  # Name + action verb
]

# Location patterns (Philippine context - provinces, cities, regions)
LOCATION_PATTERNS = [
    r'\b(Manila|Quezon City|Luneta|EDSA|Quirino Grandstand|Philippines|Malacañang|Senate|Congress|House of Representatives|Ombudsman)\b',
    r'\b(South China Sea|West Philippine Sea|WPS|Ayungin Shoal|Second Thomas Shoal|Scarborough Shoal|Bajo de Masinloc|BDM|Panatag Shoal|Spratly Islands|Kalayaan)\b',
    r'\b(National Capital Region|NCR|Metro Manila|Cordillera Administrative Region|CAR|Ilocos Region|Cagayan Valley|Central Luzon|Calabarzon|Mimaropa|Bicol|Western Visayas|Central Visayas|Eastern Visayas|Zamboanga Peninsula|Northern Mindanao|Davao Region|Soccsksargen|Caraga|Bangsamoro)\b',
    r'\b(Region [IVX]+(?:-[AB])?|Luzon|Visayas|Mindanao)\b',
    r'\b(Abra|Apayao|Benguet|Ifugao|Kalinga|Mountain Province|Batanes|Cagayan|Isabela|Nueva Vizcaya|Quirino)\b',
    r'\b(Ilocos Norte|Ilocos Sur|La Union|Pangasinan)\b',
    r'\b(Bataan|Bulacan|Nueva Ecija|Pampanga|Tarlac|Zambales|Aurora)\b',
    r'\b(Batangas|Cavite|Laguna|Quezon|Rizal)\b',
    r'\b(Marinduque|Occidental Mindoro|Oriental Mindoro|Palawan|Romblon)\b',
    r'\b(Albay|Camarines Norte|Camarines Sur|Catanduanes|Masbate|Sorsogon)\b',
    r'\b(Aklan|Antique|Capiz|Guimaras|Iloilo|Negros Occidental)\b',
    r'\b(Bohol|Cebu|Negros Oriental|Siquijor)\b',
    r'\b(Biliran|Eastern Samar|Leyte|Northern Samar|Samar|Southern Leyte)\b',
    r'\b(Davao del Norte|Davao del Sur|Davao Oriental|Davao de Oro|Davao Occidental)\b',
    r'\b(Agusan del Norte|Agusan del Sur|Surigao del Norte|Surigao del Sur|Dinagat Islands)\b',
]

# Keyword patterns (actions and subjects)
KEYWORD_PATTERNS = [
    r'\b(protest|rally|demonstration|march|accountability|corruption|flood control|scandal)s?\b',
]

# INC-specific subject patterns (for main actor detection)
INC_SUBJECT_PATTERNS = [
    r'\b(?:iglesia ni cristo|inc)\b.*?(?:led|leads|organized|organizes|held|holds|staged|stages|called|calls|rallied|rallies)',
    r'^(?:iglesia ni cristo|inc)\b',  # Starts with INC
    r'\b(?:protest|rally|demonstration)s?\s+(?:led by|organized by|by)\s+(?:the\s+)?(?:iglesia ni cristo|inc)\b',
]

# Name suffixes for proper last name extraction
NAME_SUFFIXES = ['jr.', 'jr', 'sr.', 'sr', 'ii', 'iii', 'iv', 'v']

# Statement/attribution claim patterns (for detecting "X said Y" type claims)
# Includes both reporting verbs and action verbs for demands/calls
STATEMENT_CLAIM_PATTERNS = [
    r'\b(said|says|stated|states|claimed|claims|announced|announces|declared|declares|confirmed|confirms|reported|reports|asserted|asserts)\b',
    r'\b(demanded|demands|called for|calls for|urged|urges|pushed for|pushes for|appealed for|appeals for|pressed for|presses for)\b',
    r'\baccording to\b',
    r'\bquoted as saying\b',
    r'\bin a statement\b'
]

# Negation patterns (for detecting negative claims)
NEGATION_PATTERNS = [
    r'\b(did not|didn\'t|does not|doesn\'t|do not|don\'t|has not|hasn\'t|have not|haven\'t|had not|hadn\'t|will not|won\'t|would not|wouldn\'t|cannot|can\'t|could not|couldn\'t|should not|shouldn\'t)\b',
    r'\b(never|no|not|neither|nor|nobody|nothing|nowhere|none)\b',
    r'\b(without|lacking|absent|failed to|refuses to|refused to|denies|denied|rejects|rejected)\b',
    r'\b(false|untrue|incorrect|inaccurate|baseless|unfounded)\b'
]

# Quantifier/completeness patterns (for detecting full vs partial completion)
QUANTIFIER_PATTERNS = {
    'full': [
        r'\b(full|fully|complete|completely|entire|entirely|total|totally|all|100%|one hundred percent)\b',
        r'\ball\s+\w+\s+(restored|completed|finished|done|fixed|repaired)\b'
    ],
    'partial': [
        r'\b(partial|partially|some|somewhat|incomplete|incompletely)\b',
        r'\b(\d{1,2}%|\d{1,2}\s*percent)\b', 
        r'\b(half|quarter|third|two-thirds|three-quarters)\b'
    ],
    'in_progress': [
        r'\b(ongoing|continues|continuing|still|working on|in progress|underway)\b',
        r'\b(may take|expected to take|could take|will take)\b'
    ]
}

# Month mapping for temporal extraction
MONTH_MAP = {
    'january': 1, 'february': 2, 'march': 3, 'april': 4,
    'may': 5, 'june': 6, 'july': 7, 'august': 8,
    'september': 9, 'october': 10, 'november': 11, 'december': 12
}

# Month extraction patterns
MONTH_PATTERNS = [
    (r'\b(early|mid|late)\s+(January|February|March|April|May|June|July|August|September|October|November|December)\b', True),
    (r'\b(January|February|March|April|May|June|July|August|September|October|November|December)\b', False)
]

# Date extraction patterns
DATE_PATTERNS = [
    r'\b(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\s+\d{1,2},?\s+\d{4}\b',
    r'\b(?:September|October|November|December)\s+\d{1,2}\b',
]

# Date range patterns (for extracting start and end dates from claims)
DATE_RANGE_PATTERNS = [
    # "November 9 to November 12, 2025" or "from November 9 to November 12, 2025"
    r'\b(?:from\s+)?(January|February|March|April|May|June|July|August|September|October|November|December)\s+(\d{1,2})(?:\s+to|-|–|—)(?:\s+)(January|February|March|April|May|June|July|August|September|October|November|December)\s+(\d{1,2}),?\s+(\d{4})\b',
    # "from November 9 to 12, 2025" (same month, abbreviated)
    r'\b(?:from\s+)?(January|February|March|April|May|June|July|August|September|October|November|December)\s+(\d{1,2})(?:\s+to|-|–|—)(?:\s+)(\d{1,2}),?\s+(\d{4})\b',
    # "November 9-12, 2025" (date span with dash)
    r'\b(January|February|March|April|May|June|July|August|September|October|November|December)\s+(\d{1,2})-(\d{1,2}),?\s+(\d{4})\b',
]

# Temporal relationship patterns (for detecting "after X in month" or "before X in month")
TEMPORAL_RELATIONSHIP_PATTERNS = [
    # "after [event] in [month]" or "after [event] hit in [month]"
    r'\bafter\s+(?:[\w\s]+?)\s+(?:in|during)\s+(January|February|March|April|May|June|July|August|September|October|November|December)',
    # "following [event] in [month]"
    r'\bfollowing\s+(?:[\w\s]+?)\s+(?:in|during)\s+(January|February|March|April|May|June|July|August|September|October|November|December)',
    # "before [event] in [month]"
    r'\bbefore\s+(?:[\w\s]+?)\s+(?:in|during)\s+(January|February|March|April|May|June|July|August|September|October|November|December)',
]

# Number qualifier patterns (qualifier_text, regex_pattern, qualifier_type)
NUMBER_QUALIFIER_PATTERNS = [
    (r'\b(over|more than|above|exceeding|upwards of)\s+(\d+(?:,\d{3})*(?:\.\d+)?)\s*k?\b', 'over'),
    (r'\b(at least|minimum|no fewer than)\s+(\d+(?:,\d{3})*(?:\.\d+)?)\s*k?\b', 'at_least'),
    (r'\b(nearly|almost|close to|approximately|around|about)\s+(\d+(?:,\d{3})*(?:\.\d+)?)\s*k?\b', 'approximately'),
    (r'\b(under|less than|fewer than|below)\s+(\d+(?:,\d{3})*(?:\.\d+)?)\s*k?\b', 'under'),
    (r'\b(\d+(?:,\d{3})*(?:\.\d+)?)\s*k?\b', 'exact'),  # No qualifier
]

# Temporal normalization patterns (pattern, replacement)
TEMPORAL_NORMALIZATION_PATTERNS = [
    # Future tense to present
    (r'\bwill call(?:s)? for\b', 'calls for'),
    (r'\bto call for\b', 'calls for'),
    (r'\baims? to call for\b', 'calls for'),
    (r'\bwill hold\b', 'holds'),
    (r'\bwill organize\b', 'organizes'),
    (r'\bto hold\b', 'holds'),
    (r'\bto organize\b', 'organizes'),
    (r'\bwill demand\b', 'demands'),
    (r'\bto demand\b', 'demands'),
    # Past tense to present
    (r'\bcalled for\b', 'calls for'),
    (r'\bheld\b', 'holds'),
    (r'\borganized\b', 'organizes'),
    (r'\bled\b', 'leads'),
    (r'\bstaged\b', 'stages'),
    (r'\brallied\b', 'rallies'),
    (r'\bdemanded\b', 'demands'),
    (r'\bdemanding\b', 'demands'),
    (r'\bcalling for\b', 'calls for'),
    # Special patterns
    (r'\b(?:protests|rallies|demonstrations)\s+led by\s+([^,.]+)', r'\1 leads protests'),
    (r'\bdrew? (?:huge |massive |large )?crowd to\b', 'holds rally for'),
    (r'\bdrew? (?:huge |massive |large )?crowd (?:to|at|in)\b', 'holds rally at'),
]

SPECIFIC_PLACES = [
    'nueva vizcaya', 'nueva ecija', 'la union', 'quezon city', 'manila',
    'aurora', 'cagayan', 'isabela', 'quirino', 'batanes', 'ilocos norte',
    'ilocos sur', 'pangasinan', 'benguet', 'ifugao', 'kalinga', 'apayao',
    'abra', 'mountain province', 'bulacan', 'pampanga', 'tarlac', 'zambales',
    'bataan', 'cavite', 'laguna', 'batangas', 'rizal', 'quezon', 'marinduque',
    'romblon', 'palawan', 'occidental mindoro', 'oriental mindoro', 'albay',
    'camarines norte', 'camarines sur', 'catanduanes', 'masbate', 'sorsogon',
    'cebu', 'bohol', 'negros occidental', 'negros oriental', 'leyte',
    'southern leyte', 'samar', 'eastern samar', 'northern samar', 'biliran',
    'aklan', 'antique', 'capiz', 'iloilo', 'guimaras', 'siquijor'
]
