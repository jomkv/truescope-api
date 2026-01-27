from schemas.article_schema import Article
from schemas.claim_schema import Claim
from constants.weights import (
    VERDICT_WEIGHT_MAP,
    SOURCE_BIAS_WEIGHT_MAP,
    NLI_LABEL_WEIGHT_MAP,
)
from constants.enums import Verdict, NLILabel, SourceBias
from dateparser.search import search_dates
from services import EmbeddingService, EntityExtractionService, NLIService
from sqlalchemy import select
from sqlalchemy.orm import Session
import unicodedata
import re
from datetime import datetime


class VerifyController:
    def __init__(self):
        self.embedding_service = EmbeddingService()
        self.entity_extraction_service = EntityExtractionService()
        self.nli_service = NLIService()

        # Fixed thresholds for article filtering
        self.RELEVANCE_THRESHOLD = 0.3
        self.ENTITY_THRESHOLD = 0.4

        # Weights for combined relevance score (60% semantic, 40% entity)
        self.SEMANTIC_WEIGHT = 0.6
        self.ENTITY_WEIGHT = 0.4

    @staticmethod
    def normalize_text(text: str) -> str:
        """
        Normalize input text for consistent downstream processing.

        Steps:
        - Unicode normalization (NFKC)
        - Lowercasing
        - Removal of extra whitespace
        - Stripping leading/trailing whitespace

        Args:
            text (str): The input text to normalize.

        Returns:
            str: The normalized text.
        """
        if not isinstance(text, str):
            return ""
        # Unicode normalization
        text = unicodedata.normalize("NFKC", text)
        # Lowercase
        text = text.lower()
        # Remove extra whitespace
        text = re.sub(r"\s+", " ", text)
        # Strip leading/trailing whitespace
        return text.strip()

    @staticmethod
    def find_claims_with_articles(
        session: Session, embedding: list[float], top_k: int = 20
    ) -> list[tuple[Claim, Article, float]]:
        """
        Retrieves the top_k most similar claims from the database using HNSW vector search,
        skipping claims with an 'UNKNOWN' verdict, and pairs each claim with its corresponding article.

        Args:
            session (Session): SQLAlchemy session for database access.
            embedding (list[float]): Embedding vector for the user claim.
            top_k (int, optional): Number of top similar claims to retrieve. Defaults to 20.

        Returns:
            list[tuple[Claim, Article, float]]: List of tuples containing the claim, its article, and the similarity score.
        """

        # Use HNSW index for efficient vector search
        distance_col = Claim.embedding.cosine_distance(embedding)  # type: ignore
        claim_stmt = (
            select(Claim, distance_col)  # type: ignore
            .where(Claim.verdict != "UNKNOWN")  # Skip claims with "UNKNOWN" verdicts
            .order_by(distance_col)
            .limit(top_k)
        )
        claim_results = session.execute(claim_stmt).all()
        claim_doc_ids: list[str] = [claim.doc_id for claim, _ in claim_results]

        # Get equivalent articles for all found vectors
        articles_stmt = select(Article).where(Article.doc_id.in_(claim_doc_ids))  # type: ignore
        article_results = session.execute(articles_stmt).scalars().all()
        article_map = {article.doc_id: article for article in article_results}

        # Combine results into a tuple
        # (claim, article, similarity_score)
        combined_results: list[tuple[Claim, Article, float]] = []
        for vector, distance in claim_results:
            article = article_map.get(vector.doc_id)
            if article:
                combined_results.append((vector, article, 1 - distance))

        return combined_results

    def extract_entities(self, text: str) -> list[str]:
        """
        Extracts named entities from the given text using the entity extraction service.

        Args:
            text (str): The input text from which to extract entities.

        Returns:
            list[str]: List of extracted entity names.
        """
        entities_with_label = self.entity_extraction_service.extract_entities(text)

        # Extract entity names only, exclude label
        return [entity[0] for entity in entities_with_label]

    def calculate_entity_match_score(
        self, claim_entities: list[str], text: str, article_title: str = ""
    ) -> float:
        """
        Calculates the entity match score between a list of claim entities and the provided text/article title.
        The score represents the proportion of claim entities found as whole words in the text or title.

        Args:
            claim_entities (list[str]): List of entities extracted from the claim.
            text (str): The text (e.g., article content) to search for entities.
            article_title (str, optional): The article title to also search for entities. Defaults to "".

        Returns:
            float: Entity match score between 0.0 and 1.0.
        """
        if not claim_entities:
            return 0.0

        text_norm = self.normalize_text(text)
        article_title_norm = self.normalize_text(article_title)

        matches = 0
        total = len(claim_entities)

        for entity in claim_entities:
            entity_lower = entity.lower()
            # Match as whole word in text or title
            if re.search(
                r"\b" + re.escape(entity_lower) + r"\b", text_norm
            ) or re.search(r"\b" + re.escape(entity_lower) + r"\b", article_title_norm):
                matches += 1

        return matches / total if total > 0 else 0.0

    def extract_claim_timeframe(self, user_claim: str) -> list[tuple[str, datetime]]:
        """
        Extracts temporal references (e.g., dates, periods) from a claim using dateparser.

        Args:
            claim (str): The user-provided claim to analyze for time expressions.

        Returns:
            list[tuple[str, datetime]]: List of tuples containing the matched time string and its parsed datetime.
        """

        # TODO: Still unsure on how to use extracted timeframe on the current logic

        return search_dates(user_claim, settings={"RETURN_TIME_SPAN": True})

    @staticmethod
    def compute_final_score(
        verdict: Verdict, source_bias: SourceBias, nli_label: NLILabel, nli_score: float
    ) -> float:
        """
        Computes a final score for a claim-article pair based on the verdict, source bias, NLI label, and NLI confidence.

        The output is a fuzzified score in the range [-1, 1]:
            - A score close to 1 means strong support for the user claim (completely true).
            - A score close to -1 means strong refutation of the user claim (completely false).
            - A score near 0 means the evidence is neutral or inconclusive.
        The magnitude reflects the strength of the evidence, and the sign reflects the direction (support or refute).

        Args:
            verdict (Verdict): The verdict of the found claim (as an enum).
            source_bias (SourceBias): The bias of the article's source (as an enum).
            nli_label (NLILabel): The NLI relationship label between user and found claim.
            nli_score (float): The NLI model's confidence score for the label (0.0 to 1.0).

        Returns:
            float: The computed final score (signed and fuzzified, with magnitude reflecting strength).
        """

        verdict_weight = VERDICT_WEIGHT_MAP.get(verdict, 0.5)
        bias_weight = SOURCE_BIAS_WEIGHT_MAP.get(source_bias, 0.7)
        nli_label_weight = NLI_LABEL_WEIGHT_MAP.get(nli_label, 0.5)

        return round(nli_score * bias_weight * verdict_weight * nli_label_weight, 2)

    def verify_claim(
        self, session: Session, user_claim: str, use_fallback: bool = True
    ):
        """
        Main entry point for verifying a user claim.

        Steps:
        - Embeds the user claim and searches for similar claims in the database.
        - Extracts entities from the user claim and calculates entity match scores.
        - For each found claim-article pair, runs NLI to determine the relationship and computes a final score.
        - Filters and sorts results based on relevance and entity match.

        Args:
            session (Session): SQLAlchemy session for database access.
            user_claim (str): The user-provided claim to verify.
            use_fallback (bool, optional): Whether to use fallback logic if no strong matches are found. Defaults to True.

        Returns:
            list[dict]: List of result dictionaries, each containing article and claim details, NLI results, scores, and skip reasons.
        """

        user_claim_norm = self.normalize_text(user_claim)
        claim_embedding = self.embedding_service.embed_text(user_claim_norm)
        search_results = self.find_claims_with_articles(session, claim_embedding)
        claim_entities = self.extract_entities(user_claim_norm)

        results = []

        for claim, article, similarity_score in search_results:
            entity_match_score = self.calculate_entity_match_score(
                claim_entities, claim.claim_text, article.title
            )
            combined_relevance_score = (similarity_score * self.SEMANTIC_WEIGHT) + (
                entity_match_score * self.ENTITY_WEIGHT
            )

            article_dict = {
                "doc_id": article.doc_id,
                "title": article.title,
                "content": (
                    article.content[:500] + "..."
                    if len(article.content) > 500
                    else article.content
                ),
                "found_claim": claim.claim_text,
                "found_verdict": claim.verdict,
                "publish_date": (
                    article.publish_date.isoformat() if article.publish_date else None
                ),
                "url": article.url,
                "similarity_score": round(similarity_score, 4),
                "entity_match_score": round(entity_match_score, 4),
                "combined_relevance_score": round(combined_relevance_score, 4),
                "nli_result": None,
                "verdict": None,
                "skip_reason": [],
            }

            if (
                similarity_score >= self.RELEVANCE_THRESHOLD
                and entity_match_score >= self.ENTITY_THRESHOLD
            ):
                nli_label, nli_score, nli_avg = self.nli_service.classify_nli(
                    user_claim_norm, claim.claim_text
                )
                nli_result_payload = {
                    "relationship": nli_label,
                    "relationship_confidence": nli_score,
                    "relationship_avg": nli_avg,
                    "claim_source": article.source,
                    "analyzed_text": claim.claim_text,
                }
                article_dict["nli_result"] = nli_result_payload
                article_dict["verdict"] = self.compute_final_score(
                    Verdict(claim.verdict),
                    SourceBias(article.source_bias),
                    nli_label,
                    nli_score,
                )
            else:
                if similarity_score < self.RELEVANCE_THRESHOLD:
                    article_dict["skip_reason"].append(
                        f"Low semantic similarity ({similarity_score:.3f} < {self.RELEVANCE_THRESHOLD})"
                    )
                if entity_match_score < self.ENTITY_THRESHOLD:
                    article_dict["skip_reason"].append(
                        f"Key entities not found ({entity_match_score:.3f} < {self.ENTITY_THRESHOLD})"
                    )
                if not article_dict["skip_reason"]:
                    article_dict["skip_reason"].append(
                        "Did not meet filtering criteria"
                    )

            results.append(article_dict)

        # Sort: NLI results first, explicit claims first, then by combined score
        results.sort(
            key=lambda x: (
                0 if x["nli_result"] else 1,
                (
                    0
                    if x["nli_result"]
                    and x["nli_result"]["claim_source"] == "explicit_claim"
                    else 1
                ),
                -x["combined_relevance_score"],
            )
        )

        return results
