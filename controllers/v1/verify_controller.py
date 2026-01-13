import re
from datetime import datetime, timedelta
from sentence_transformers import SentenceTransformer
from sqlalchemy import select
from pgvector.sqlalchemy import Vector
from schemas.article_vector_schema import ArticleVector
from schemas.article_schema import Article
from models.article_vector_model import ArticleVectorModel
from models.article_model import ArticleModel
from constants.keywords import (
    ORGANIZATION_PATTERNS,
    PEOPLE_PATTERNS,
    LOCATION_PATTERNS,
    KEYWORD_PATTERNS,
    INC_SUBJECT_PATTERNS,
    NAME_SUFFIXES,
    MONTH_MAP,
    MONTH_PATTERNS,
    DATE_PATTERNS,
    DATE_RANGE_PATTERNS,
    TEMPORAL_RELATIONSHIP_PATTERNS,
    NUMBER_QUALIFIER_PATTERNS,
    TEMPORAL_NORMALIZATION_PATTERNS,
    STATEMENT_CLAIM_PATTERNS,
    NEGATION_PATTERNS,
    QUANTIFIER_PATTERNS,
    SPECIFIC_PLACES,
)


class VerifyController:
    def __init__(self):
        self.model = SentenceTransformer(
            "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        )
        # NLI model components will be lazy-loaded when first needed
        self._nli_tokenizer = None
        self._nli_classifier = None

    def embed_claim(self, claim: str) -> list[float]:
        return self.model.encode(claim)

    @staticmethod
    def get_articles_by_doc_ids(session, doc_ids: list[str]) -> list[Article]:
        """
        Retrieve articles by a list of doc_ids.
        """
        stmt = select(Article).where(Article.doc_id.in_(doc_ids))
        articles = session.execute(stmt).scalars().all()

        return articles

    @staticmethod
    def find_similar_embeddings(
        session, embedding, top_k=20
    ) -> list[tuple[ArticleVectorModel, float]]:
        """
        Search for top_k most similar embeddings in the database.
        Returns list of tuples: (ArticleVectorModel, similarity_score)
        where similarity_score is 1 - cosine_distance (higher = more similar)
        """
        from sqlalchemy import func
        
        distance_col = ArticleVector.embedding.cosine_distance(embedding)
        stmt = (
            select(ArticleVector, distance_col)
            .order_by(distance_col)
            .limit(top_k)
        )
        results = session.execute(stmt).all()
        
        # Convert distance to similarity score (1 - distance)
        # Cosine distance is in [0, 2], so similarity is in [0, 1]
        return [(article_vector, 1 - distance) for article_vector, distance in results]

    def get_relevant_articles(
        self, session, claim: str, limit: int = 20
    ) -> list[tuple[ArticleModel, float]]:
        """
        Get all relevant articles from claim with their similarity scores.
        Returns list of tuples: (ArticleModel, similarity_score)
        """
        embedding = self.embed_claim(claim)
        similar_embeddings = self.find_similar_embeddings(session, embedding, limit)
        
        # Build a map of doc_id to similarity score
        similarity_map = {vec.doc_id: score for vec, score in similar_embeddings}
        doc_ids = list(similarity_map.keys())
        
        articles = self.get_articles_by_doc_ids(session, doc_ids)
        
        # Return articles with their similarity scores, maintaining order
        return [(article, similarity_map[article.doc_id]) for article in articles]

    def extract_entities(self, text: str) -> dict:
        """
        Extract named entities (organizations, people, locations) from text.
        Returns dict with lists of extracted entities.
        """
        
        entities = {
            "organizations": [],
            "people": [],
            "locations": [],
            "keywords": []
        }
        
        # Extract organizations
        for pattern in ORGANIZATION_PATTERNS:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                org = match.group(0).strip()
                if org and org not in entities["organizations"]:
                    entities["organizations"].append(org)
        
        # Extract people
        for pattern in PEOPLE_PATTERNS:
            matches = re.finditer(pattern, text)
            for match in matches:
                person = match.group(1).strip()
                if person and len(person.split()) >= 2 and person not in entities["people"]:
                    entities["people"].append(person)
        
        # Extract locations
        for pattern in LOCATION_PATTERNS:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                location = match.group(0).strip()
                if location and location not in entities["locations"]:
                    entities["locations"].append(location)
        
        # Extract keywords
        for pattern in KEYWORD_PATTERNS:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                keyword = match.group(1).lower()
                if keyword and keyword not in entities["keywords"]:
                    entities["keywords"].append(keyword)
        
        return entities
    
    def calculate_entity_match_score(self, claim_entities: dict, article_text: str, article_title: str = "") -> float:
        """
        Calculate entity match score between claim entities and article text.
        Returns score from 0.0 to 1.0 indicating entity overlap.
        Higher score means key entities from claim are the MAIN ACTORS in the article.
        
        Stricter matching: Entity must be the subject/actor, not just mentioned.
        """
        
        article_text_lower = article_text.lower()
        article_title_lower = article_title.lower()
        
        matches = 0
        total_entities = 0
        
        # Check organizations (highest weight) - must be main actor
        for org in claim_entities["organizations"]:
            total_entities += 1
            org_lower = org.lower()
            
            # Special case: INC and Iglesia ni Cristo are equivalent
            if "iglesia ni cristo" in org_lower or org_lower == "inc":
                # Check if INC/Iglesia is the SUBJECT (actor), not just mentioned
                is_main_actor = False
                for pattern in INC_SUBJECT_PATTERNS:
                    if re.search(pattern, article_title_lower, re.IGNORECASE) or \
                       re.search(pattern, article_text_lower[:500], re.IGNORECASE):  # Check first 500 chars
                        is_main_actor = True
                        break
                
                if is_main_actor:
                    matches += 1
                elif "iglesia ni cristo" in article_text_lower or re.search(r'\binc\b', article_text_lower):
                    # Entity mentioned but not main actor - partial credit
                    matches += 0.3
            else:
                # Generic organization check
                if org_lower in article_title_lower:
                    matches += 1
                elif org_lower in article_text_lower[:300]:
                    matches += 0.5
        
        # Check people (medium weight, higher for subject-action claims)
        for person in claim_entities["people"]:
            # Higher weight if person is likely the subject (single name at start, or name + action verb)
            is_subject = len(claim_entities["people"]) == 1 and len(claim_entities["organizations"]) == 0
            weight = 1.0 if is_subject else 0.5
            total_entities += weight
            person_lower = person.lower()
            
            # Check full name match first
            if person_lower in article_text_lower:
                matches += weight
            else:
                # Try matching last name (more flexible)
                # Extract last name, handling suffixes like Jr., Sr., III, etc.
                name_parts = person.split()
                if len(name_parts) >= 2:
                    # Check if last part is a suffix
                    if name_parts[-1].lower().rstrip('.') in NAME_SUFFIXES and len(name_parts) >= 3:
                        # Use second-to-last as last name
                        last_name = name_parts[-2].lower()
                    else:
                        # Use last part as last name
                        last_name = name_parts[-1].lower()
                    
                    # Match last name as a standalone word (not part of another word)
                    if re.search(r'\b' + re.escape(last_name) + r'\b', article_text_lower):
                        matches += weight * 0.8  # Slightly lower score for partial match
                elif len(name_parts) == 1:
                    # Single name (e.g., "Trump") - try case-insensitive match
                    if re.search(r'\b' + re.escape(person_lower) + r'\b', article_text_lower):
                        matches += weight
        
        # Check locations (lower weight but strict for specific places)
        for location in claim_entities["locations"]:
            total_entities += 0.3
            location_lower = location.lower()
            
            # For specific provinces/cities/municipalities, require exact match
            is_specific_place = any(sp in location_lower for sp in SPECIFIC_PLACES)
            
            if is_specific_place:
                # Strict: Must appear in article for specific places
                if location_lower in article_text_lower or location_lower in article_title_lower:
                    matches += 0.3
                # No partial credit for specific places
            else:
                # Generic locations (NCR, Luzon, Visayas, etc.) - more lenient
                if location_lower in article_text_lower:
                    matches += 0.3
        
        if total_entities == 0:
            # No entities extracted - likely generic claim or extraction failed
            # Return 0.0 to filter out unrelated articles (was 1.0 which caused false matches)
            return 0.0
        
        return min(matches / total_entities, 1.0)

    def extract_claim_timeframe(self, claim: str) -> dict:
        """
        Extract temporal references from claim (e.g., 'late November 2025', 'early 2024', 'November 9 to November 12, 2025').
        Returns dict with year, month, and relative timing if found.
        """
        
        timeframe = {
            "years": [],
            "months": [],
            "relative_timing": None,  # "early", "late", "mid"
            "date_range": None,  # (start_date, end_date) tuple
            "temporal_relation": None  # "after", "before", "during"
        }
        
        # First, check for temporal relationships ("after X in November")
        for pattern in TEMPORAL_RELATIONSHIP_PATTERNS:
            match = re.search(pattern, claim, re.IGNORECASE)
            if match:
                month_name = match.group(1)
                month = MONTH_MAP[month_name.lower()]
                
                # Determine relationship type
                if 'after' in match.group(0).lower():
                    timeframe["temporal_relation"] = "after"
                elif 'following' in match.group(0).lower():
                    timeframe["temporal_relation"] = "after"
                elif 'before' in match.group(0).lower():
                    timeframe["temporal_relation"] = "before"
                
                # Extract year from claim
                years = re.findall(r'\b(202[0-9]|201[0-9])\b', claim)
                if years:
                    year = int(years[0])
                    timeframe["years"] = [year]
                    timeframe["months"] = [month]
                    
                    if timeframe["temporal_relation"] == "after":
                        # Events after X in November means late November onwards
                        start = datetime(year, month, 20)  # Late in the month
                        # Allow articles through end of next month
                        if month == 12:
                            end = datetime(year + 1, 2, 1) - timedelta(days=1)
                        else:
                            next_month = min(month + 2, 12)
                            if next_month == 12:
                                end = datetime(year, 12, 31)
                            else:
                                end = datetime(year, next_month, 1) - timedelta(days=1)
                        timeframe["date_range"] = (start, end)
                        return timeframe
                    elif timeframe["temporal_relation"] == "before":
                        # Events before X in November means up to early November
                        start = datetime(year, 1, 1)  # Start of year
                        end = datetime(year, month, 10)  # Early in the month
                        timeframe["date_range"] = (start, end)
                        return timeframe
        
        # Second, try to extract explicit date ranges (high priority)
        for pattern in DATE_RANGE_PATTERNS:
            match = re.search(pattern, claim, re.IGNORECASE)
            if match:
                groups = match.groups()
                
                if len(groups) == 5 and groups[0] == groups[2]:  # "November 9 to November 12, 2025"
                    month_name = groups[0]
                    start_day = int(groups[1])
                    end_day = int(groups[3])
                    year = int(groups[4])
                    month = MONTH_MAP[month_name.lower()]
                    
                    start = datetime(year, month, start_day)
                    end = datetime(year, month, end_day)
                    timeframe["date_range"] = (start, end)
                    timeframe["years"] = [year]
                    timeframe["months"] = [month]
                    return timeframe
                    
                elif len(groups) == 4:  # "November 9 to 12, 2025" or "November 9-12, 2025"
                    month_name = groups[0]
                    start_day = int(groups[1])
                    end_day = int(groups[2])
                    year = int(groups[3])
                    month = MONTH_MAP[month_name.lower()]
                    
                    start = datetime(year, month, start_day)
                    end = datetime(year, month, end_day)
                    timeframe["date_range"] = (start, end)
                    timeframe["years"] = [year]
                    timeframe["months"] = [month]
                    return timeframe
                    
                elif len(groups) == 5:  # "November 9 to December 12, 2025" (different months)
                    start_month_name = groups[0]
                    start_day = int(groups[1])
                    end_month_name = groups[2]
                    end_day = int(groups[3])
                    year = int(groups[4])
                    
                    start_month = MONTH_MAP[start_month_name.lower()]
                    end_month = MONTH_MAP[end_month_name.lower()]
                    
                    start = datetime(year, start_month, start_day)
                    end = datetime(year, end_month, end_day)
                    timeframe["date_range"] = (start, end)
                    timeframe["years"] = [year]
                    timeframe["months"] = [start_month, end_month]
                    return timeframe
        
        # If no explicit date range, fall back to year/month extraction
        # Extract years
        years = re.findall(r'\b(202[0-9]|201[0-9])\b', claim)
        timeframe["years"] = [int(y) for y in years]
        
        # Extract months
        for pattern, has_relative in MONTH_PATTERNS:
            matches = re.finditer(pattern, claim, re.IGNORECASE)
            for match in matches:
                if has_relative:
                    relative = match.group(1).lower()
                    month_name = match.group(2)
                    timeframe["relative_timing"] = relative
                else:
                    month_name = match.group(1)
                
                timeframe["months"].append(MONTH_MAP[month_name.lower()])
        
        # Build date range if we have enough info
        if timeframe["years"] and timeframe["months"]:
            year = timeframe["years"][0]
            month = timeframe["months"][0]
            
            if timeframe["relative_timing"] == "early":
                # Early month: days 1-10
                start = datetime(year, month, 1)
                end = datetime(year, month, min(10, 28))
            elif timeframe["relative_timing"] == "late":
                # Late month: days 20-end
                start = datetime(year, month, 20)
                # Get last day of month
                if month == 12:
                    end = datetime(year + 1, 1, 1) - timedelta(days=1)
                else:
                    end = datetime(year, month + 1, 1) - timedelta(days=1)
            elif timeframe["relative_timing"] == "mid":
                # Mid month: days 10-20
                start = datetime(year, month, 10)
                end = datetime(year, month, 20)
            else:
                # Just the month
                start = datetime(year, month, 1)
                if month == 12:
                    end = datetime(year + 1, 1, 1) - timedelta(days=1)
                else:
                    end = datetime(year, month + 1, 1) - timedelta(days=1)
            
            timeframe["date_range"] = (start, end)
        elif timeframe["years"]:
            # Just a year
            year = timeframe["years"][0]
            start = datetime(year, 1, 1)
            end = datetime(year, 12, 31)
            timeframe["date_range"] = (start, end)
        
        return timeframe

    def check_temporal_relevance(self, article_date, claim_timeframe: dict) -> bool:
        """
        Check if article's publish date falls within claim's timeframe.
        Returns True if relevant, False if outside timeframe.
        """
        
        if not claim_timeframe["date_range"] or not article_date:
            return True  # No timeframe restriction
        
        start_date, end_date = claim_timeframe["date_range"]
        
        # Allow 30-day window on either side for flexibility
        margin = timedelta(days=30)
        return (start_date - margin) <= article_date <= (end_date + margin)

    def extract_dates_and_numbers(self, text: str) -> dict:
        """
        Extract dates and numbers with qualifiers from text for fact-checking.
        Returns dict with dates, numbers, and their qualifiers.
        """

        result = {"dates": [], "numbers": [], "number_qualifiers": []}
        
        # Extract dates (various formats)
        for pattern in DATE_PATTERNS:
            matches = re.findall(pattern, text, re.IGNORECASE)
            result["dates"].extend(matches)
        
        # Extract numbers with qualifiers
        for pattern, qualifier_type in NUMBER_QUALIFIER_PATTERNS:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                if qualifier_type == 'exact':
                    num_str = match.group(1)
                else:
                    num_str = match.group(2)
                
                try:
                    clean_num = num_str.replace(',', '')
                    
                    # Skip 4-digit years (201X, 202X) - these are dates, not quantities
                    if len(clean_num) == 4 and clean_num.isdigit():
                        year_value = int(clean_num)
                        if 2010 <= year_value <= 2029:
                            continue
                    
                    # Check if 'k' appears right after the number
                    end_pos = match.end()
                    has_k = end_pos < len(text) and text[end_pos:end_pos+1].lower() == 'k'
                    
                    if has_k or 'k' in match.group(0).lower():
                        value = float(clean_num) * 1000
                    else:
                        value = float(clean_num)
                    
                    if value >= 100:  # Only include substantial numbers
                        result["numbers"].append(value)
                        result["number_qualifiers"].append(qualifier_type)
                except:
                    pass
        
        return result

    def normalize_temporal_language(self, text: str) -> str:
        """
        Normalize temporal language to reduce false mismatches from tense differences.
        Converts future/past tense to present tense for better semantic matching.
        """
        
        # Apply all normalization patterns
        for pattern, replacement in TEMPORAL_NORMALIZATION_PATTERNS:
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
        
        return text

    def extract_key_sentence(self, content: str, max_length: int = 200) -> str:
        """
        Extract the first substantial sentence from content as a proxy claim.
        """
        if not content:
            return ""
        
        # Split into sentences (simple approach)
        sentences = content.replace('\n', ' ').split('. ')
        
        # Find first sentence that's substantial (not too short)
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) > 50:  # Skip very short sentences
                # Truncate if too long
                if len(sentence) > max_length:
                    return sentence[:max_length] + "..."
                return sentence + ('' if sentence.endswith('.') else '.')
        
        # Fallback to first sentence
        return sentences[0][:max_length] + "..." if sentences else ""

    def check_claim_relation(self, user_claim: str, article_claim: str, full_content: str = None, similarity_score: float = None) -> dict:
        """
        Determine if article_claim supports, refutes, or is neutral to user_claim.
        
        Uses semantic similarity as primary signal (more reliable than NLI for this task).
        Also performs date/number extraction to detect mismatches.
        
        Returns a dict with:
        - relation: "supports", "refutes", or "neutral"
        - confidence: float score for the predicted relation
        - all_scores: dict with scores for all three categories
        - warnings: list of detected mismatches
        """
        
        # Detect if this is a statement/attribution claim (X said/claimed/announced Y)
        is_statement_claim = any(re.search(pattern, user_claim.lower()) for pattern in STATEMENT_CLAIM_PATTERNS)
        
        # Detect negation in user claim (e.g., "did not", "never", "no")
        is_negated_claim = any(re.search(pattern, user_claim.lower()) for pattern in NEGATION_PATTERNS)
        
        # Detect quantifiers/completeness in claim vs article
        claim_has_full = any(re.search(pattern, user_claim.lower()) for pattern in QUANTIFIER_PATTERNS['full'])
        article_has_partial = any(re.search(pattern, (article_claim + " " + (full_content or "")).lower()) for pattern in QUANTIFIER_PATTERNS['partial'])
        article_in_progress = any(re.search(pattern, (article_claim + " " + (full_content or "")).lower()) for pattern in QUANTIFIER_PATTERNS['in_progress'])
        
        # Extract dates and numbers for fact-checking
        user_data = self.extract_dates_and_numbers(user_claim)
        article_data = self.extract_dates_and_numbers(article_claim + " " + (full_content or ""))
        
        # Check for mismatches
        warnings = []
        predicted_relation = "neutral"
        confidence = 0.5
        
        # Quantifier/completeness mismatch detection
        if claim_has_full and (article_has_partial or article_in_progress):
            # Claim says "fully restored" but article says "60% restored" or "ongoing"
            warnings.append(f"Completeness mismatch: Claim indicates full completion, but article indicates partial or ongoing work")
        
        # Date mismatch detection
        if user_data["dates"] and article_data["dates"]:
            user_dates_str = " ".join(user_data["dates"]).lower()
            article_dates_str = " ".join(article_data["dates"]).lower()
            
            # Check if dates are completely different
            date_overlap = any(ud.lower() in article_dates_str for ud in user_data["dates"])
            if not date_overlap and len(user_data["dates"]) > 0:
                warnings.append(f"Date mismatch: User mentions {user_data['dates'][0]}, article mentions {article_data['dates'][0] if article_data['dates'] else 'different date'}")
        
        # Number mismatch detection (for crowd sizes, etc.) with qualifier awareness
        if user_data["numbers"] and article_data["numbers"]:
            user_num = user_data["numbers"][0]
            article_num = max(article_data["numbers"])  # Use max from article
            user_qualifier = user_data["number_qualifiers"][0] if user_data["number_qualifiers"] else 'exact'
            
            # Apply logic based on qualifier
            is_mismatch = False
            
            if user_qualifier == 'over':
                # "over 90,000" means >= 90,000, so 130,000 is valid
                if article_num < user_num * 0.9:  # Allow 10% margin for rounding
                    is_mismatch = True
                    warnings.append(f"Number contradiction: User claims over {int(user_num):,}, but article reports {int(article_num):,}")
            
            elif user_qualifier == 'at_least':
                # "at least X" means >= X
                if article_num < user_num * 0.9:
                    is_mismatch = True
                    warnings.append(f"Number contradiction: User claims at least {int(user_num):,}, but article reports {int(article_num):,}")
            
            elif user_qualifier == 'under':
                # "under X" means < X
                if article_num >= user_num:
                    is_mismatch = True
                    warnings.append(f"Number contradiction: User claims under {int(user_num):,}, but article reports {int(article_num):,}")
            
            elif user_qualifier == 'approximately':
                # "approximately X" allows wider margin (±30%)
                if abs(user_num - article_num) / max(user_num, article_num) > 0.3:
                    is_mismatch = True
                    warnings.append(f"Number mismatch: User mentions approximately {int(user_num):,}, article mentions {int(article_num):,}")
            
            else:  # 'exact' or no qualifier
                # Exact number with narrower margin (±20%)
                if abs(user_num - article_num) / max(user_num, article_num) > 0.2:
                    warnings.append(f"Number mismatch: User mentions {int(user_num):,}, article mentions {int(article_num):,}")
        
        # Use semantic similarity as primary signal
        # High similarity + no critical mismatches = SUPPORTS
        # Low similarity = NEUTRAL
        # Mismatches = REFUTES
        
        # Adjust thresholds for statement claims (lower bar for indirect reporting)
        if is_statement_claim:
            # For "X said Y" claims, lower thresholds since articles reporting the statement
            # may not use identical wording but still confirm the statement was made
            high_threshold = 0.60      # was 0.70
            moderate_threshold = 0.50  # was 0.60
            low_threshold = 0.40       # was 0.45
        else:
            # Standard thresholds for factual claims
            high_threshold = 0.70
            moderate_threshold = 0.60
            low_threshold = 0.45
        
        if similarity_score is not None:
            if similarity_score >= high_threshold:
                # Very high similarity - article clearly supports claim
                predicted_relation = "supports"
                if is_statement_claim:
                    confidence = min(0.85 + (similarity_score - high_threshold) * 2, 0.99)  # 0.85-0.99
                else:
                    confidence = min(0.9 + (similarity_score - high_threshold) * 2, 0.99)  # 0.90-0.99
            elif similarity_score >= moderate_threshold:
                # High similarity - likely supports
                predicted_relation = "supports"
                if is_statement_claim:
                    confidence = min(0.70 + (similarity_score - moderate_threshold) * 1.5, 0.90)  # 0.70-0.85
                else:
                    confidence = min(0.75 + (similarity_score - moderate_threshold), 0.95)  # 0.75-0.95
            elif similarity_score >= low_threshold:
                # Moderate similarity - neutral (unless statement claim with decent match)
                if is_statement_claim and similarity_score >= 0.45:
                    # Statement claims: 0.45-0.50 range can still be supportive
                    predicted_relation = "supports"
                    confidence = 0.65 + (similarity_score - 0.45) * 0.5  # 0.65-0.70
                else:
                    predicted_relation = "neutral"
                    confidence = 0.60 + (similarity_score - low_threshold) * 0.3  # 0.60-0.65
            else:
                # Low similarity - neutral or refutes
                predicted_relation = "neutral" if not warnings else "refutes"
                confidence = 0.50
        else:
            # Fallback if similarity not provided
            predicted_relation = "neutral"
            confidence = 0.5
        
        # Adjust prediction if there are critical mismatches
        if warnings and predicted_relation == "supports":
            # Downgrade to neutral if there are fact mismatches
            predicted_relation = "neutral"
            confidence = max(confidence * 0.7, 0.5)  # Reduce confidence
            warnings.append("Relation adjusted to neutral due to factual mismatches")
        
        # Handle negation: invert relation if claim is negated
        if is_negated_claim and similarity_score is not None and similarity_score >= 0.55:
            # High similarity to negated claim means article contradicts the negation
            # Example: Claim "X did NOT happen" + Article "X happened" = REFUTES
            if predicted_relation == "supports":
                predicted_relation = "refutes"
                warnings = warnings if warnings else []
                warnings.append("Claim contains negation: relation inverted from supports to refutes")
            elif predicted_relation == "refutes":
                # Article already contradicts claim, which aligns with negated claim
                predicted_relation = "supports"
                warnings = warnings if warnings else []
                warnings.append("Claim contains negation: relation inverted from refutes to supports")
            # Keep confidence but adjust scores for inverted relation
        
        # Convert to NLI-style all_scores format
        scores = {
            "refutes": 0.01,
            "neutral": 0.01,
            "supports": 0.01
        }
        scores[predicted_relation] = confidence
        # Distribute remaining probability
        remaining = 1.0 - confidence
        other_relations = [r for r in scores.keys() if r != predicted_relation]
        for relation in other_relations:
            scores[relation] = remaining / len(other_relations)
        
        return {
            "relation": predicted_relation,
            "confidence": confidence,
            "all_scores": scores,
            "warnings": warnings if warnings else None
        }

    def verify_claim_with_articles(
        self, session, claim: str, limit: int = 20, use_fallback: bool = True,
        relevance_threshold: float = 0.3, entity_threshold: float = 0.4
    ) -> list[dict]:
        """
        Get relevant articles and check their relationship with the input claim.
        
        Returns a list of articles with NLI analysis. Uses fallback strategies:
        1. Use explicit claim if available
        2. Use article title as proxy claim
        3. Extract key sentence from content
        
        Filters articles by:
        - Semantic similarity (relevance_threshold)
        - Entity matching (entity_threshold) - ensures key actors/orgs in claim appear in article
        
        Args:
            session: Database session
            claim: User's claim to verify
            limit: Maximum number of articles to retrieve
            use_fallback: Whether to use titles/content when no explicit claim exists
            relevance_threshold: Minimum similarity score (0-1) for NLI analysis
                               Default 0.3 filters out weakly related articles
            entity_threshold: Minimum entity match score (0-1) for article inclusion
                            Default 0.4 ensures key entities from claim appear in article
        """
        # Extract entities from user claim for matching
        claim_entities = self.extract_entities(claim)
        # Extract temporal references from claim for date filtering
        claim_timeframe = self.extract_claim_timeframe(claim)
        
        articles_with_scores = self.get_relevant_articles(session, claim, limit)
        
        results = []
        skipped_articles = []
        
        for article, similarity_score in articles_with_scores:
            # Filter 1: Check temporal relevance (date must match claim's timeframe)
            if not self.check_temporal_relevance(article.publish_date, claim_timeframe):
                article_dict = {
                    "doc_id": article.doc_id,
                    "title": article.title,
                    "content": article.content[:500] + "..." if len(article.content) > 500 else article.content,
                    "claim": article.claim,
                    "verdict": article.verdict,
                    "publish_date": article.publish_date.isoformat() if article.publish_date else None,
                    "url": article.url,
                    "similarity_score": round(similarity_score, 4),
                    "entity_match_score": None,
                    "combined_relevance_score": None,
                    "nli_result": None,
                    "skip_reason": ["Published outside claim's timeframe"]
                }
                skipped_articles.append(article_dict)
                continue
            
            # Calculate entity match score (stricter - checks if entity is main actor)
            article_full_text = f"{article.title} {article.content}"
            entity_match_score = self.calculate_entity_match_score(
                claim_entities, article_full_text, article.title
            )
            
            # Calculate combined relevance score (weighted: 60% semantic, 40% entity)
            combined_score = (similarity_score * 0.6) + (entity_match_score * 0.4)
            
            article_dict = {
                "doc_id": article.doc_id,
                "title": article.title,
                "content": article.content[:500] + "..." if len(article.content) > 500 else article.content,
                "claim": article.claim,
                "verdict": article.verdict,
                "publish_date": article.publish_date.isoformat() if article.publish_date else None,
                "url": article.url,
                "similarity_score": round(similarity_score, 4),
                "entity_match_score": round(entity_match_score, 4),
                "combined_relevance_score": round(combined_score, 4),
                "nli_result": None
            }
            
            # Filter: Only perform NLI on articles meeting both thresholds
            if similarity_score >= relevance_threshold and entity_match_score >= entity_threshold:
                # Determine what to use for NLI analysis
                claim_to_check = None
                claim_source = None
                
                if article.claim and article.claim.strip():
                    # Priority 1: Use explicit claim from fact-checking article
                    claim_to_check = article.claim
                    claim_source = "explicit_claim"
                elif use_fallback:
                    # Priority 2: Use title as proxy claim (clear and direct)
                    if article.title and article.title.strip():
                        claim_to_check = article.title
                        claim_source = "title"
                    # Priority 3: Extract key sentence from content only if no title
                    elif article.content:
                        extracted = self.extract_key_sentence(article.content)
                        if extracted:
                            claim_to_check = extracted
                            claim_source = "content_extraction"
                
                # Perform NLI analysis if we have something to check
                if claim_to_check:
                    nli_result = self.check_claim_relation(claim, claim_to_check, article.content, similarity_score)
                    nli_result["claim_source"] = claim_source
                    nli_result["analyzed_text"] = claim_to_check[:200] + "..." if len(claim_to_check) > 200 else claim_to_check
                    article_dict["nli_result"] = nli_result
                    results.append(article_dict)
            else:
                # Article didn't meet thresholds - add to skipped list
                article_dict["skip_reason"] = []
                if similarity_score < relevance_threshold:
                    article_dict["skip_reason"].append(f"Low semantic similarity ({similarity_score:.3f} < {relevance_threshold})")
                if entity_match_score < entity_threshold:
                    article_dict["skip_reason"].append(f"Key entities not found ({entity_match_score:.3f} < {entity_threshold})")
                if not article_dict["skip_reason"]:
                    article_dict["skip_reason"].append("Did not meet filtering criteria")
                skipped_articles.append(article_dict)
        
        # Sort results: prioritize by combined relevance score
        results.sort(key=lambda x: (
            # First: Has NLI analysis
            0 if x["nli_result"] else 1,
            # Second: Explicit claims over fallbacks
            0 if x["nli_result"] and x["nli_result"]["claim_source"] == "explicit_claim" else 1,
            # Third: Combined relevance score (descending)
            -x["combined_relevance_score"]
        ))
        
        # Add skipped articles at the end for transparency
        results.extend(skipped_articles)
        
        return results

    @staticmethod
    def calculate_summary_statistics(results: list[dict]) -> dict:
        """
        Calculate summary statistics from verification results.
        
        Args:
            results: List of article results from verify_claim_with_articles
        
        Returns:
            Dictionary with summary statistics including counts, averages, and breakdowns
        """
        # Calculate counts
        analyzed_count = sum(1 for r in results if r.get("nli_result"))
        skipped_count = sum(1 for r in results if r.get("skip_reason"))
        supports_count = sum(1 for r in results if r.get("nli_result") and r["nli_result"]["relation"] == "supports")
        refutes_count = sum(1 for r in results if r.get("nli_result") and r["nli_result"]["relation"] == "refutes")
        neutral_count = sum(1 for r in results if r.get("nli_result") and r["nli_result"]["relation"] == "neutral")
        
        # Breakdown by claim source
        source_breakdown = {}
        for r in results:
            if r.get("nli_result"):
                source = r["nli_result"].get("claim_source", "unknown")
                source_breakdown[source] = source_breakdown.get(source, 0) + 1
        
        # Calculate average scores for analyzed vs skipped articles
        analyzed_results = [r for r in results if r.get("nli_result")]
        skipped_results = [r for r in results if r.get("skip_reason")]
        
        analyzed_avg_sim = sum(r["similarity_score"] for r in analyzed_results) / len(analyzed_results) if analyzed_results else 0
        analyzed_avg_entity = sum(r["entity_match_score"] for r in analyzed_results) / len(analyzed_results) if analyzed_results else 0
        analyzed_avg_combined = sum(r["combined_relevance_score"] for r in analyzed_results) / len(analyzed_results) if analyzed_results else 0
        
        skipped_avg_sim = sum(r["similarity_score"] for r in skipped_results) / len(skipped_results) if skipped_results else 0
        skipped_avg_entity = sum(r["entity_match_score"] for r in skipped_results if r["entity_match_score"] is not None) / len([r for r in skipped_results if r["entity_match_score"] is not None]) if any(r["entity_match_score"] is not None for r in skipped_results) else 0
        skipped_avg_combined = sum(r["combined_relevance_score"] for r in skipped_results if r["combined_relevance_score"] is not None) / len([r for r in skipped_results if r["combined_relevance_score"] is not None]) if any(r["combined_relevance_score"] is not None for r in skipped_results) else 0
        
        # Count skip reasons
        skip_reasons_count = {}
        for r in skipped_results:
            for reason in r.get("skip_reason", []):
                reason_key = "low_similarity" if "similarity" in reason else "missing_entities"
                skip_reasons_count[reason_key] = skip_reasons_count.get(reason_key, 0) + 1
        
        return {
            "total_articles": len(results),
            "articles_analyzed": analyzed_count,
            "articles_skipped": skipped_count,
            "avg_similarity_analyzed": round(analyzed_avg_sim, 4),
            "avg_similarity_skipped": round(skipped_avg_sim, 4),
            "avg_entity_match_analyzed": round(analyzed_avg_entity, 4),
            "avg_entity_match_skipped": round(skipped_avg_entity, 4),
            "avg_combined_relevance_analyzed": round(analyzed_avg_combined, 4),
            "avg_combined_relevance_skipped": round(skipped_avg_combined, 4),
            "skip_reasons": skip_reasons_count,
            "supports": supports_count,
            "refutes": refutes_count,
            "neutral": neutral_count,
            "claim_sources": source_breakdown
        }
