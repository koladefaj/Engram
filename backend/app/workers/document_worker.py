import json
import redis
import os
import logging
from app.infrastructure.queue.celery_app import celery_app
from app.dependencies import get_document_processor, get_storage_service
from app.infrastructure.db.session_sync import db_session_scope
from app.infrastructure.db.models import Document
from app.infrastructure.config import settings
from app.infrastructure.logging import request_id_var
from asgiref.sync import async_to_sync

logger = logging.getLogger(__name__)

redis_client = redis.from_url(settings.redis_url)


def _publish(channel: str, task_id: str, status: str, **extra):
    redis_client.publish(channel, json.dumps({"task_id": task_id, "status": status, **extra}))


@celery_app.task(bind=True, name="process_document_task", max_retries=5)
def process_document_task(self, document_id: str, request_id: str = "worker-gen"):
    """Core background task: extract → summarise → embed → index."""
    token = request_id_var.set(request_id)
    task_id = self.request.id
    channel = f"notifications_{task_id}"

    processor, storage_service = get_document_processor(), get_storage_service()

    try:
        with db_session_scope() as db:
            doc = db.query(Document).filter(Document.id == document_id).first()
            if not doc:
                logger.error(f"Task {task_id}: document {document_id} not found")
                return {"error": "Document not found"}

            doc.status = "PROCESSING"
            db.commit()
            logger.info(f"Processing document {document_id} (task {task_id})")

            # ---------------------------------------------------------- #
            # DE-DUPLICATION                                              #
            # ---------------------------------------------------------- #
            if doc.content_hash:
                existing = db.query(Document).filter(
                    Document.content_hash == doc.content_hash,
                    Document.status == "COMPLETED",
                    Document.id != doc.id,
                ).first()

                if existing:
                    logger.info(f"Duplicate found — cloning results from {existing.id}")
                    doc.raw_text = existing.raw_text
                    doc.analysis = existing.analysis
                    doc.status = "COMPLETED"

                    from sqlalchemy import text as sa_text
                    db.execute(sa_text("""
                        INSERT INTO data_document_embeddings (id, document_id, text, embedding, meta)
                        SELECT gen_random_uuid(), :new_id, text, embedding, meta
                        FROM data_document_embeddings
                        WHERE document_id = :old_id
                    """), {"new_id": doc.id, "old_id": existing.id})

                    db.commit()

                    # BUG FIX: notify WebSocket so the frontend doesn't hang
                    _publish(channel, task_id, "COMPLETED", document_id=document_id)
                    return {"document_id": document_id, "status": "CLONED", "source": str(existing.id)}

            # ---------------------------------------------------------- #
            # PHASE 1: TEXT EXTRACTION                                   #
            # ---------------------------------------------------------- #
            doc.status = "EXTRACTING_TEXT"
            db.commit()
            _publish(channel, task_id, "EXTRACTING_TEXT")

            path_to_process = async_to_sync(storage_service.get_file_path)(str(doc.id))

            if not os.path.exists(path_to_process):
                doc.status = "FAILED"
                raise Exception(f"NON_RETRYABLE: File not found at {path_to_process}")

            # Call extraction directly so the status update is accurate
            # (the old process_sync did both extraction + summary invisibly)
            raw_text = processor.extract_text(path_to_process, mime_type=doc.content)

            if not raw_text.strip():
                raise Exception("NON_RETRYABLE: No text could be extracted from the document")

            # ---------------------------------------------------------- #
            # PHASE 2: AI SUMMARY (STREAMING)                            #
            # ---------------------------------------------------------- #
            doc.status = "GENERATING_SUMMARY"
            db.commit()
            _publish(channel, task_id, "GENERATING_SUMMARY")

            def on_summary_chunk(chunk: str):
                _publish(channel, task_id, "SUMMARY_CHUNK", chunk=chunk)

            summary = processor.generate_summary_sync(raw_text, on_chunk=on_summary_chunk)

            # Instructor-structured metadata (key_points, document_type, flags, etc.)
            structured_metadata = processor.get_structured_analysis(raw_text)

            doc.raw_text = raw_text
            doc.analysis = {
                "summary": summary,
                "word_count": len(raw_text.split()),
                "estimated_tokens": len(raw_text) // 4,
                "contains_email": "@" in raw_text,
                "contains_money": any(s in raw_text for s in ["$", "USD", "NGN", "€"]),
                "ai_provider": settings.ai_provider,
                **structured_metadata,
            }

            # ---------------------------------------------------------- #
            # PHASE 3: CHUNKING COMPARISON + EMBEDDING                   #
            # ---------------------------------------------------------- #
            doc.status = "GENERATING_EMBEDDINGS"
            db.commit()
            _publish(channel, task_id, "GENERATING_EMBEDDINGS")

            from app.dependencies import get_rag_service
            from llama_index.core import Document as LlamaDocument

            rag_service = get_rag_service()

            # Compare fixed vs semantic chunking — stores comparison stats in analysis.
            # Indexing uses fixed-size nodes (fast + predictable); comparison is for observability.
            chunking_comparison, nodes = RAGService.compare_chunking_strategies(raw_text)
            doc.analysis["chunking_comparison"] = chunking_comparison

            logger.info(f"Chunking: {len(nodes)} fixed-size chunks")

            # ---------------------------------------------------------- #
            # PHASE 4: VECTOR INDEXING                                   #
            # ---------------------------------------------------------- #
            doc.status = "INDEXING"
            db.commit()
            _publish(channel, task_id, "INDEXING")

            rag_service.index_nodes(db, str(doc.id), nodes)

            # ---------------------------------------------------------- #
            # COMPLETION                                                  #
            # ---------------------------------------------------------- #
            doc.status = "COMPLETED"

            tokens_used = doc.analysis.get("estimated_tokens", 0)
            from app.infrastructure.db.models import User
            db.query(User).filter(User.id == doc.owner_id).update(
                {User.total_tokens: User.total_tokens + tokens_used}
            )

            db.commit()
            logger.info(f"Document {document_id} completed. Tokens ≈ {tokens_used}")

            _publish(channel, task_id, "COMPLETED", analysis=doc.analysis)
            return {"document_id": document_id, "status": "COMPLETED"}

    except Exception as e:
        error_msg = str(e)
        is_permanent = (
            "NON_RETRYABLE" in error_msg
            or "too short" in error_msg
            or "not found" in error_msg
        )
        is_transient = any(
            kw in error_msg for kw in ["Rate Limit", "429", "timeout", "connection", "AI Engine failed"]
        )

        if is_permanent or (not is_transient) or self.request.retries >= self.max_retries:
            logger.critical(f"Task {task_id} permanently failed: {error_msg}")
            with db_session_scope() as db:
                doc = db.query(Document).filter(Document.id == document_id).first()
                if doc:
                    doc.status = "FAILED"
                    doc.error_message = error_msg
            _publish(channel, task_id, "FAILED", error=error_msg)
            return {"error": error_msg}

        logger.warning(f"Task {task_id} transient error — retrying. Error: {error_msg}")
        with db_session_scope() as db:
            doc = db.query(Document).filter(Document.id == document_id).first()
            if doc:
                doc.status = "FAILED"
                doc.error_message = f"Transient (retrying): {error_msg}"

        _publish(channel, task_id, "RETRYING", message="Transient error, retrying...")
        countdown = min(60 * (2 ** self.request.retries), 3600)
        raise self.retry(exc=e, countdown=countdown)

    finally:
        request_id_var.reset(token)


# Deferred import to avoid circular dependency at module load time
from app.domain.services.rag_service import RAGService  # noqa: E402
