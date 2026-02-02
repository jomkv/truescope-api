from schemas.article_schema import Article
from schemas.claim_schema import Claim
from schemas.article_chunk_schema import ArticleChunk
from constants.weights import (
    VERDICT_WEIGHT_MAP,
    SOURCE_BIAS_WEIGHT_MAP,
    NLI_LABEL_WEIGHT_MAP,
)
from constants.enums import Verdict, NLILabel, SourceBias
from dateparser.search import search_dates
from services import (
    EmbeddingService,
    EntityExtractionService,
    NLIService,
    RemarksGenerationService,
)
from sqlalchemy import select
from sqlalchemy.orm import Session
import unicodedata
import re
from datetime import datetime
import asyncio
from concurrent.futures import ThreadPoolExecutor


class VerifyController:
    def __init__(self):
        self.embedding_service = EmbeddingService()
        self.entity_extraction_service = EntityExtractionService()
        self.nli_service = NLIService()
        self.remarks_generation_service = RemarksGenerationService()
        # Fixed thresholds for article filtering
        self.RELEVANCE_THRESHOLD = 0.3
        self.ENTITY_THRESHOLD = 0.4

        # Weights for combined relevance score (60% semantic, 40% entity)
        self.SEMANTIC_WEIGHT = 0.6
        self.ENTITY_WEIGHT = 0.4

        # Thread pool for CPU-intensive ML operations (2 workers optimal for PyTorch models)
        self.executor = ThreadPoolExecutor(max_workers=2)

        # Limit deep analysis to top results to avoid excessive NLI + remarks tasks
        self.MAX_DEEP_ANALYSIS = 5

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

    def find_claims_with_articles(
        self, session: Session, embedding: list[float], user_claim: str, top_k: int = 20
    ) -> list[tuple[Claim, Article, float, str]]:
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

        # Batch-load all chunks at once (N+1 query fix)
        all_chunks = (
            session.query(ArticleChunk)
            .filter(ArticleChunk.doc_id.in_(claim_doc_ids))
            .all()
        )
        chunk_map = {}
        for chunk in all_chunks:
            chunk_map.setdefault(chunk.doc_id, []).append(chunk)

        # Combine results into a tuple
        # (claim, article, similarity_score, chunk_texts)
        combined_results: list[tuple[Claim, Article, float, str]] = []
        for vector, distance in claim_results:
            article = article_map.get(vector.doc_id)
            if article:
                # Get chunks from batch-loaded map
                chunks = chunk_map.get(article.doc_id, [])
                chunk_texts = self._build_remarks_context(chunks)
                combined_results.append((vector, article, 1 - distance, chunk_texts))

        return combined_results

    def find_news_articles(
        self, session: Session, embedding: list[float], user_claim: str, top_k: int = 20
    ) -> list[tuple[Article, float, str]]:
        """
        Retrieves the top_k most similar articles from the database using HNSW vector search.

        Args:
            session (Session): SQLAlchemy session for database access.
            embedding (list[float]): Embedding vector for the user claim.
            top_k (int, optional): Number of top similar articles to retrieve. Defaults to 20.

        Returns:
            list[tuple[Article, float]]: List of tuples containing the article and the similarity score.
        """
        distance_col = ArticleChunk.embedding.cosine_distance(embedding)  # type: ignore

        chunk_stmt = (
            select(ArticleChunk, distance_col)  # type: ignore
            .order_by(distance_col)
            .limit(top_k)
        )
        chunk_results = session.execute(chunk_stmt).all()

        article_map: dict[str, Article] = {}
        doc_ids_seen: set[str] = set()
        results: list[tuple[Article, float, str]] = []

        # First pass: collect unique doc_ids
        for chunk, distance in chunk_results:
            doc_id = chunk.doc_id
            if doc_id not in doc_ids_seen:
                doc_ids_seen.add(doc_id)

        # Batch-load all articles and chunks at once (N+1 query fix)
        articles = session.query(Article).filter(Article.doc_id.in_(doc_ids_seen)).all()
        article_map = {article.doc_id: article for article in articles}

        all_chunks = (
            session.query(ArticleChunk)
            .filter(ArticleChunk.doc_id.in_(doc_ids_seen))
            .all()
        )
        chunk_map = {}
        for chunk in all_chunks:
            chunk_map.setdefault(chunk.doc_id, []).append(chunk)

        # Second pass: build results using cached data
        for chunk, distance in chunk_results:
            similarity_score = 1 - distance
            doc_id = chunk.doc_id

            if doc_id in article_map and doc_id not in {r[0].doc_id for r in results}:
                article = article_map[doc_id]
                # Get chunks from batch-loaded map
                chunks = chunk_map.get(doc_id, [])
                chunk_texts = self._build_remarks_context(chunks)
                results.append((article, similarity_score, chunk_texts))

        results.sort(key=lambda x: x[1], reverse=True)

        return results[:top_k]

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

    @staticmethod
    def _build_remarks_context(
        chunks: list[ArticleChunk],
        max_chars: int = 800,
        max_chunks: int = 3,
    ) -> str:
        """
        Builds a short context string for remarks generation.
        Keeps a few chunk excerpts to avoid sending full article content.
        """
        if not chunks:
            return ""

        parts: list[str] = []
        used_chars = 0

        for chunk in chunks:
            if not chunk.chunk_content:
                continue

            if len(parts) >= max_chunks or used_chars >= max_chars:
                break

            remaining = max_chars - used_chars
            content = chunk.chunk_content.strip()
            if not content:
                continue

            snippet = content[:remaining].rsplit(" ", 1)[0].strip()
            if not snippet:
                continue

            parts.append(snippet)
            used_chars += len(snippet) + 1

        return " ".join(parts).strip()

    @staticmethod
    def _truncate_at_sentence(text: str, max_chars: int = 200) -> str:
        if not text:
            return ""

        cleaned = re.sub(r"\s+", " ", text.strip())
        if len(cleaned) <= max_chars:
            return cleaned

        truncated = cleaned[:max_chars]
        sentence_matches = list(re.finditer(r"[.!?](?=\s+[A-Z0-9]|$)", truncated))

        if sentence_matches:
            last_end = sentence_matches[-1].end()
            return cleaned[:last_end].strip()

        return truncated.rsplit(" ", 1)[0].rstrip() + "..."

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
        verdict: Verdict | None,
        source_bias: SourceBias | None,
        nli_label: NLILabel,
        nli_score: float,
        is_factcheck: bool = True,
        similarity_score: float = 0.0,
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
            is_factcheck (bool): Whether the claim is from a fact-checking source.
            similarity_score: float = 0.0,

        Returns:
            float: The computed final score (signed and fuzzified, with magnitude reflecting strength).
        """
        if not is_factcheck:

            if nli_label == NLILabel.SUPPORT:
                base_score = 0.75
            elif nli_label == NLILabel.REFUTE:
                base_score = -0.75
            else:
                base_score = similarity_score  # some news articles may be neutral but relevant, so instead of relying only on NLI, use similarity score directly

            confidence_multiplier = 0.5 + (nli_score * 0.5)

            bias_weight = (
                SOURCE_BIAS_WEIGHT_MAP.get(source_bias, 0.7) if source_bias else 0.7
            )

            return round(base_score * confidence_multiplier * bias_weight, 2)

        if verdict is None or source_bias is None:
            return 0.0

        verdict_weight = VERDICT_WEIGHT_MAP.get(verdict, 0.5)
        bias_weight = SOURCE_BIAS_WEIGHT_MAP.get(source_bias, 0.7)
        nli_label_weight = NLI_LABEL_WEIGHT_MAP.get(nli_label, 0.5)

        return round(nli_score * bias_weight * verdict_weight * nli_label_weight, 2)

    async def verify_claim(
        self, session: Session, user_claim: str, use_fallback: bool = True
    ):
        """
        Main entry point for verifying a user claim (async version).

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

        # Run embedding and entity extraction in parallel
        loop = asyncio.get_event_loop()
        claim_embedding, claim_entities = await asyncio.gather(
            loop.run_in_executor(
                self.executor, self.embedding_service.embed_text, user_claim_norm
            ),
            loop.run_in_executor(self.executor, self.extract_entities, user_claim_norm),
        )

        # search for both FC and news articles (DB queries can't be parallelized with same session)
        factcheck_results = self.find_claims_with_articles(
            session, claim_embedding, user_claim
        )
        news_results = self.find_news_articles(session, claim_embedding, user_claim)

        # Limit deep analysis to top results to avoid excessive parallelization (FIX 1)
        factcheck_results = factcheck_results[: self.MAX_DEEP_ANALYSIS]
        news_results = news_results[: self.MAX_DEEP_ANALYSIS]

        # Process all results in parallel
        tasks = []

        # Process FC results
        for claim, article, similarity_score, chunk_texts in factcheck_results:
            tasks.append(
                self._process_result_async(
                    user_claim_norm=user_claim_norm,
                    claim_entities=claim_entities,
                    similarity_score=similarity_score,
                    article=article,
                    claim_text=claim.claim_text,
                    claim_verdict=claim.verdict,
                    source_bias=article.source_bias,
                    is_factcheck=True,
                    chunk_texts=chunk_texts,
                )
            )

        # Process news results
        for article, similarity_score, chunk_texts in news_results:
            if (article.type or "").strip().lower() == "fact-check":
                continue
            tasks.append(
                self._process_result_async(
                    user_claim_norm=user_claim_norm,
                    claim_entities=claim_entities,
                    similarity_score=similarity_score,
                    article=article,
                    claim_text=None,
                    claim_verdict=None,
                    source_bias=article.source_bias,
                    is_factcheck=False,
                    chunk_texts=chunk_texts,
                )
            )

        # Wait for all processing to complete
        results = await asyncio.gather(*tasks)

        results.sort(
            key=lambda x: (
                0 if x["nli_result"] else 1,
                0 if x.get("found_claim") else 1,
                -x["combined_relevance_score"],
            )
        )

        return results

    async def _process_result_async(
        self,
        user_claim_norm: str,
        claim_entities: list[str],
        similarity_score: float,
        article: Article,
        claim_text: str | None,
        claim_verdict: str | None,
        source_bias: str | None,
        is_factcheck: bool,
        chunk_texts: str | None,
    ) -> dict:

        # For entity matching: use claim text for FC, or article content+title for news
        if is_factcheck and claim_text:
            entity_comparison_text = claim_text
            entity_comparison_title = ""  # Claim text is self-contained
        else:
            # For news articles, combine content and title
            entity_comparison_text = article.content or ""
            entity_comparison_title = article.title

        entity_match_score = self.calculate_entity_match_score(
            claim_entities, entity_comparison_text, entity_comparison_title
        )
        combined_relevance_score = (similarity_score * self.SEMANTIC_WEIGHT) + (
            entity_match_score * self.ENTITY_WEIGHT
        )

        # Truncate content ONLY for display
        display_content = (
            article.content[:500] + "..."
            if article.content and len(article.content) > 500
            else (article.content or "")
        )

        result = {
            "doc_id": article.doc_id,
            "title": article.title,
            "content": display_content,
            "found_claim": claim_text,
            "found_verdict": claim_verdict,
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
            "source_type": "fact_check" if is_factcheck else "news_article",
            "remarks": None,
        }

        if (
            similarity_score >= self.RELEVANCE_THRESHOLD
            and entity_match_score >= self.ENTITY_THRESHOLD
        ):
            # For NLI: use claim for FC, article content for news
            nli_text = (
                claim_text
                if is_factcheck and claim_text
                else (article.content or article.title)
            )

            # Run NLI classification in thread pool
            loop = asyncio.get_event_loop()
            nli_label, nli_score, nli_avg = await loop.run_in_executor(
                self.executor,
                self.nli_service.classify_nli,
                user_claim_norm,
                nli_text,
            )

            # For remarks: always use article content for better evidence
            remarks_text = article.content or article.title or ""

            result["nli_result"] = {
                "relationship": nli_label,
                "relationship_confidence": nli_score,
                "relationship_avg": nli_avg,
                "claim_source": article.source,
                "analyzed_text": self._truncate_at_sentence(remarks_text, 200),
            }

            # verdict for FC
            if is_factcheck and claim_verdict and source_bias:
                verdict_score = self.compute_final_score(
                    verdict=Verdict(claim_verdict),
                    source_bias=SourceBias(source_bias) if source_bias else None,
                    nli_label=nli_label,
                    nli_score=nli_score,
                    is_factcheck=is_factcheck,
                    similarity_score=similarity_score,
                )
                result["verdict"] = verdict_score
            # verdict for news articles
            else:
                verdict_score = self.compute_final_score(
                    verdict=None,
                    source_bias=SourceBias(source_bias) if source_bias else None,
                    nli_label=nli_label,
                    nli_score=nli_score,
                    is_factcheck=is_factcheck,
                    similarity_score=similarity_score,
                )
                result["verdict"] = verdict_score

            # Generate remarks after verdict computation
            # Only generate remarks for high-relevance items (combined_relevance_score >= 0.6)
            # Use chunk_texts preferentially (already optimized excerpts)
            if combined_relevance_score >= 0.6:
                remarks_input = chunk_texts or article.title or ""

                if remarks_input:
                    # Run remarks generation in thread pool
                    result["remarks"] = await loop.run_in_executor(
                        self.executor,
                        self.remarks_generation_service.generate_remarks,
                        remarks_input,
                        verdict_score,
                        result["nli_result"].get("relationship", "neutral"),
                        not is_factcheck,
                    )
        else:
            if similarity_score < self.RELEVANCE_THRESHOLD:
                result["skip_reason"].append(
                    f"Low semantic similarity ({similarity_score:.3f} < {self.RELEVANCE_THRESHOLD})"
                )
            if entity_match_score < self.ENTITY_THRESHOLD:
                result["skip_reason"].append(
                    f"Key entities not found ({entity_match_score:.3f} < {self.ENTITY_THRESHOLD})"
                )
            if not result["skip_reason"]:
                result["skip_reason"].append("Did not meet filtering criteria")

        return result
