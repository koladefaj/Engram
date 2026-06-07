import uuid
import logging
from sqlalchemy import select
from sqlalchemy.orm import Session
from app.infrastructure.db.models import DocumentEmbedding
from app.infrastructure.config import settings
from llama_index.core import Settings

logger = logging.getLogger(__name__)

# Cosine similarity below this threshold triggers the CRAG fallback path.
CRAG_CONFIDENCE_THRESHOLD = 0.30

# Initial retrieval pool before re-ranking (fetch more, keep the best).
RETRIEVAL_POOL_SIZE = 20


class _Reranker:
    """
    Lazy singleton for the BGE cross-encoder.
    The model is ~270MB and downloads once to ~/.cache/huggingface.
    It is loaded on first use rather than at startup to keep boot time fast.
    """
    _model = None
    MODEL_NAME = "BAAI/bge-reranker-base"

    @classmethod
    def get(cls):
        if cls._model is None:
            from sentence_transformers import CrossEncoder
            logger.info(f"Re-ranker: loading {cls.MODEL_NAME} (first-use download if not cached)")
            cls._model = CrossEncoder(cls.MODEL_NAME)
        return cls._model

    @classmethod
    def rerank(cls, query: str, passages: list[str], top_k: int = 5) -> list[tuple[int, float]]:
        """
        Scores (query, passage) pairs with a cross-encoder.
        Returns up to top_k (original_index, score) tuples sorted by score descending.
        Cross-encoders are more accurate than bi-encoders for re-ranking because
        they see the query and passage together, not as separate embeddings.
        """
        model = cls.get()
        pairs = [(query, p) for p in passages]
        scores = model.predict(pairs).tolist()
        ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
        return ranked[:top_k]


class RAGService:
    """
    Retrieval-Augmented Generation service.

    Retrieval pipeline (two-stage):
      1. Vector search (pgvector HNSW) — high recall, retrieves RETRIEVAL_POOL_SIZE candidates
      2. BGE cross-encoder re-ranking — high precision, keeps top-K

    CRAG fallback:
      If the best retrieval similarity score is below CRAG_CONFIDENCE_THRESHOLD
      the answer is prefixed with an explicit low-confidence warning so the user
      knows not to rely on it — rather than silently hallucinating.
    """

    def __init__(self, ollama_client=None, gemini_client=None):
        self.provider = settings.ai_provider.lower()
        self.ollama_client = ollama_client
        self.gemini_client = gemini_client
        self.ollama_model = settings.ollama_model
        self.gemini_model = settings.gemini_model

    # ------------------------------------------------------------------ #
    # INDEXING                                                            #
    # ------------------------------------------------------------------ #

    def index_nodes(self, session: Session, document_id: str, nodes: list) -> int:
        """
        Batch-embeds LlamaIndex nodes and stores them in pgvector.
        Deletes existing embeddings first so re-indexing is idempotent.
        """
        try:
            doc_uuid = uuid.UUID(document_id) if isinstance(document_id, str) else document_id

            session.query(DocumentEmbedding).filter(
                DocumentEmbedding.document_id == doc_uuid
            ).delete()

            texts = [node.get_content() for node in nodes]
            logger.info(f"RAG: batch-embedding {len(texts)} chunks")
            embeddings = Settings.embed_model.get_text_embedding_batch(texts)

            for node, embedding in zip(nodes, embeddings):
                session.add(DocumentEmbedding(
                    document_id=doc_uuid,
                    text=node.get_content(),
                    embedding=embedding,
                    meta=node.metadata,
                ))

            session.commit()
            logger.info(f"RAG: indexed {len(nodes)} chunks for document {document_id}")
            return len(nodes)

        except Exception as e:
            session.rollback()
            logger.error(f"RAG indexing error: {e}")
            raise

    # ------------------------------------------------------------------ #
    # RETRIEVAL HELPERS                                                   #
    # ------------------------------------------------------------------ #

    def _vector_search(
        self,
        session: Session,
        document_id: str,
        query_embedding: list[float],
        limit: int,
    ) -> tuple[list[DocumentEmbedding], list[float]]:
        """
        Runs cosine similarity search in pgvector and returns
        (embeddings, similarity_scores) — similarity = 1 − cosine_distance.
        """
        doc_uuid = uuid.UUID(document_id) if isinstance(document_id, str) else document_id
        distance_col = DocumentEmbedding.embedding.cosine_distance(query_embedding).label("distance")

        stmt = (
            select(DocumentEmbedding, distance_col)
            .filter(DocumentEmbedding.document_id == doc_uuid)
            .order_by(distance_col)
            .limit(limit)
        )
        rows = session.execute(stmt).all()

        embeddings = [row[0] for row in rows]
        similarities = [round(1.0 - float(row[1]), 4) for row in rows]
        return embeddings, similarities

    def _rerank(
        self,
        query: str,
        candidates: list[DocumentEmbedding],
        top_k: int,
    ) -> list[tuple[DocumentEmbedding, float]]:
        """
        Re-ranks candidates with the BGE cross-encoder.
        Returns a list of (embedding_obj, reranker_score) sorted best-first.
        """
        passages = [c.text for c in candidates]
        ranked = _Reranker.rerank(query, passages, top_k=top_k)
        return [(candidates[idx], score) for idx, score in ranked]

    def _assess_confidence(self, max_similarity: float) -> str:
        if max_similarity >= 0.60:
            return "high"
        elif max_similarity >= CRAG_CONFIDENCE_THRESHOLD:
            return "medium"
        return "low"

    # ------------------------------------------------------------------ #
    # CONTEXT BUILDING                                                    #
    # ------------------------------------------------------------------ #

    def _prepare_rag_context(
        self,
        session: Session,
        document_id: str,
        query_text: str,
        chat_history: list | None = None,
        final_limit: int = 5,
    ) -> tuple[list[tuple[DocumentEmbedding, float]], str, str]:
        """
        Full retrieval pipeline:
          vector search (pool) → re-rank → CRAG confidence assessment → prompt

        Returns (ranked_results, prompt, confidence_level).
        ranked_results is a list of (DocumentEmbedding, reranker_score).
        """
        query_embedding = Settings.embed_model.get_text_embedding(query_text)

        # Stage 1: vector search for recall
        candidates, similarities = self._vector_search(
            session, document_id, query_embedding, limit=RETRIEVAL_POOL_SIZE
        )

        if not candidates:
            return [], "", "low"

        max_similarity = max(similarities) if similarities else 0.0
        confidence = self._assess_confidence(max_similarity)

        # Stage 2: BGE cross-encoder re-ranking for precision
        try:
            ranked = self._rerank(query_text, candidates, top_k=final_limit)
        except Exception as e:
            logger.warning(f"Re-ranker unavailable ({e}), falling back to vector order")
            ranked = [(c, s) for c, s in zip(candidates[:final_limit], similarities[:final_limit])]

        # Build context from re-ranked chunks
        context_text = "\n\n---\n\n".join(r.text for r, _ in ranked)

        history_text = ""
        if chat_history:
            history_text = "RECENT CONVERSATION:\n" + "\n".join(
                f"{m['role'].upper()}: {m['content']}" for m in chat_history
            )

        # CRAG prefix injected when retrieval confidence is low
        crag_prefix = ""
        if confidence == "low":
            crag_prefix = (
                "[SYSTEM NOTE — LOW RETRIEVAL CONFIDENCE: The retrieved context may not "
                "directly address this question. Answer cautiously and acknowledge uncertainty.]\n\n"
            )

        prompt = f"""{crag_prefix}You are Aegis, a professional document analyst.

CORE GUIDELINES:
1. For small talk or acknowledgements, respond naturally — do not force a document reference.
2. For document questions, only use the provided DOCUMENT CONTEXT. Admit clearly if the information is not present.
3. Be concise and precise.

{history_text}

DOCUMENT CONTEXT:
{context_text}

USER MESSAGE:
{query_text}

AEGIS RESPONSE:"""

        return ranked, prompt, confidence

    # ------------------------------------------------------------------ #
    # QUERY (non-streaming)                                               #
    # ------------------------------------------------------------------ #

    def query(
        self,
        session: Session,
        document_id: str,
        query_text: str,
        chat_history: list | None = None,
        limit: int = 5,
    ) -> dict:
        ranked, prompt, confidence = self._prepare_rag_context(
            session, document_id, query_text, chat_history, limit
        )

        if not ranked:
            return {
                "answer": "I couldn't find any relevant information in the document to answer your question.",
                "sources": [],
                "retrieval_confidence": "low",
            }

        if self.provider == "ollama":
            if not self.ollama_client:
                raise ValueError("Ollama client not initialised in RAGService")
            response = self.ollama_client.chat(
                model=self.ollama_model,
                messages=[{"role": "user", "content": prompt}],
                options={"temperature": 0.1},
            )
            answer = response["message"]["content"]
        else:
            if not self.gemini_client:
                raise ValueError("Gemini client not initialised in RAGService")
            response = self.gemini_client.models.generate_content(
                model=self.gemini_model,
                contents=[prompt],
            )
            answer = response.text

        sources = [
            {"text": emb.text, "score": round(score, 4), "metadata": emb.meta}
            for emb, score in ranked
        ]

        return {
            "answer": answer,
            "sources": sources,
            "retrieval_confidence": confidence,
        }

    # ------------------------------------------------------------------ #
    # STREAM QUERY                                                        #
    # ------------------------------------------------------------------ #

    def stream_query(
        self,
        session: Session,
        document_id: str,
        query_text: str,
        chat_history: list | None = None,
        limit: int = 5,
    ):
        """Synchronous generator that yields tokens one at a time."""
        ranked, prompt, confidence = self._prepare_rag_context(
            session, document_id, query_text, chat_history, limit
        )

        if not ranked:
            yield "I couldn't find any relevant information in the document to answer your question."
            return

        if confidence == "low":
            yield "[Low retrieval confidence — answer may be incomplete]\n\n"

        if self.provider == "ollama":
            stream = self.ollama_client.chat(
                model=self.ollama_model,
                messages=[{"role": "user", "content": prompt}],
                stream=True,
                options={"temperature": 0.1},
            )
            for chunk in stream:
                content = chunk.get("message", {}).get("content", "")
                if content:
                    yield content
        else:
            stream = self.gemini_client.models.generate_content_stream(
                model=self.gemini_model,
                contents=[prompt],
            )
            for chunk in stream:
                if chunk.text:
                    yield chunk.text

    # ------------------------------------------------------------------ #
    # CHUNKING STRATEGY COMPARISON (called by the worker, stored in DB)  #
    # ------------------------------------------------------------------ #

    @staticmethod
    def compare_chunking_strategies(text: str) -> tuple[dict, list]:
        """
        Runs both fixed-size (SentenceSplitter) and semantic chunking
        (SemanticSplitterNodeParser) on the same text, logs the comparison,
        and returns (comparison_dict, fixed_nodes).

        Fixed nodes are returned for actual indexing — semantic chunking is
        an extra embedding pass that is expensive; we log it for observability
        but index with fixed-size chunks in production.
        """
        from llama_index.core import Document as LlamaDoc
        from llama_index.core.node_parser import SentenceSplitter

        doc = LlamaDoc(text=text)

        fixed_parser = SentenceSplitter(chunk_size=512, chunk_overlap=50)
        fixed_nodes = fixed_parser.get_nodes_from_documents([doc])
        fixed_lens = [len(n.get_content()) for n in fixed_nodes]

        semantic_count = len(fixed_nodes)
        avg_semantic_chars = sum(fixed_lens) // max(len(fixed_lens), 1)

        try:
            from llama_index.core.node_parser import SemanticSplitterNodeParser
            semantic_parser = SemanticSplitterNodeParser(
                embed_model=Settings.embed_model,
                breakpoint_percentile_threshold=95,
            )
            semantic_nodes = semantic_parser.get_nodes_from_documents([doc])
            semantic_count = len(semantic_nodes)
            semantic_lens = [len(n.get_content()) for n in semantic_nodes]
            avg_semantic_chars = sum(semantic_lens) // max(len(semantic_lens), 1)
        except Exception as e:
            logger.warning(f"Semantic chunking failed (non-fatal): {e}")

        avg_fixed_chars = sum(fixed_lens) // max(len(fixed_lens), 1)
        recommended = "semantic" if len(text) > 3_000 else "fixed"

        comparison = {
            "fixed_chunk_count": len(fixed_nodes),
            "semantic_chunk_count": semantic_count,
            "avg_fixed_chars": avg_fixed_chars,
            "avg_semantic_chars": avg_semantic_chars,
            "recommended_strategy": recommended,
            "reason": (
                "Semantic chunking preserves topic boundaries for long documents."
                if recommended == "semantic"
                else "Fixed-size chunking is faster and sufficient for short documents."
            ),
        }

        logger.info(
            f"Chunking comparison — fixed: {comparison['fixed_chunk_count']} chunks "
            f"(avg {avg_fixed_chars} chars), semantic: {semantic_count} chunks "
            f"(avg {avg_semantic_chars} chars)"
        )
        return comparison, fixed_nodes
