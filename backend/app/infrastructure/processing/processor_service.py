import os
import logging
import asyncio
import pandas as pd
import pytesseract
from ollama import Client
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_message
from pypdf import PdfReader
from google import genai
from docx import Document as DocxReader
from pdf2image import convert_from_path
from typing import Any, Callable, Optional

from app.infrastructure.config import settings
from app.domain.exceptions import ProcessingError
from app.domain.services.document_processor import DocumentProcessorInterface

logger = logging.getLogger(__name__)


class DocumentProcessor(DocumentProcessorInterface):
    def __init__(self, provider: str, ollama_client: Client = None, gemini_client: genai.Client = None):
        self.provider = provider.lower()
        self.ollama_client = ollama_client
        self.gemini_client = gemini_client
        self.ollama_model = settings.ollama_model
        self.gemini_model = settings.gemini_model
        self.api_key = settings.gemini_api.strip('"') if settings.gemini_api else ""
        # Instructor client is created lazily to avoid import-time errors
        self._instructor_client = None

    # ------------------------------------------------------------------ #
    # TEXT SANITIZATION                                                    #
    # ------------------------------------------------------------------ #

    def _sanitize_text(self, text: str) -> str:
        if not text:
            return ""
        text = text.replace("\x00", "")
        text = "".join(c for c in text if c.isprintable() or c in "\n\r\t")
        return text.strip()

    # ------------------------------------------------------------------ #
    # TEXT EXTRACTION  (now public so the worker can call it directly)    #
    # ------------------------------------------------------------------ #

    def extract_text(self, file_path: str, mime_type: str | None = None) -> str:
        """
        Extracts plain text from PDF, DOCX, Excel, CSV, and TXT files.
        Falls back to Tesseract OCR for scanned PDFs.
        """
        text = ""
        try:
            ext = os.path.splitext(file_path)[1].lower()

            is_pdf = ext == ".pdf" or mime_type == "application/pdf"
            is_docx = ext in [".docx", ".doc"] or mime_type in [
                "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                "application/msword",
            ]
            is_excel = ext in [".xls", ".xlsx"] or mime_type in [
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                "application/vnd.ms-excel",
            ]
            is_csv = ext == ".csv" or mime_type == "text/csv"
            is_txt = ext == ".txt" or mime_type == "text/plain"

            if is_pdf:
                reader = PdfReader(file_path)
                for i, page in enumerate(reader.pages):
                    page_text = self._sanitize_text(page.extract_text() or "")
                    text += page_text
                    logger.debug(f"PDF page {i + 1}: {len(page_text)} chars")
                logger.info(f"PDF extraction complete: {len(text)} characters")

                if not text.strip():
                    logger.warning("PDF appears scanned — falling back to OCR...")
                    try:
                        pages = convert_from_path(file_path)
                        ocr_text = ""
                        for i, page_image in enumerate(pages[:20]):
                            ocr_text += self._sanitize_text(pytesseract.image_to_string(page_image))
                            logger.debug(f"OCR page {i + 1}")
                        text = ocr_text if ocr_text.strip() else "[Scanned PDF — OCR found no text]"
                        logger.info(f"OCR extraction complete: {len(text)} characters")
                    except Exception as e:
                        raise ProcessingError(f"OCR failed: {e}")
                return self._sanitize_text(text)

            elif is_docx:
                doc = DocxReader(file_path)
                text = "\n".join(p.text for p in doc.paragraphs)
            elif is_excel:
                df = pd.read_excel(file_path, nrows=500)
                text = df.to_string(index=False)
            elif is_csv:
                df = pd.read_csv(file_path, nrows=500)
                text = df.to_string(index=False)
            elif is_txt:
                with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                    text = f.read()
            else:
                logger.warning(f"Unsupported file type: {mime_type or ext}")

        except Exception as e:
            logger.error("Text extraction failed", exc_info=True)
            raise ProcessingError(f"Text extraction error: {e}")

        return self._sanitize_text(text)

    # ------------------------------------------------------------------ #
    # INSTRUCTOR — STRUCTURED METADATA VIA PYDANTIC                      #
    # ------------------------------------------------------------------ #

    def _get_instructor_client(self):
        """
        Returns a lazily-created instructor-patched client.
        instructor enforces that LLM output deserialises into a Pydantic model —
        it retries with corrective feedback on validation failure.
        """
        if self._instructor_client is not None:
            return self._instructor_client

        import instructor

        if self.provider == "ollama":
            self._instructor_client = instructor.from_ollama(
                Client(host=settings.ollama_base_url),
                mode=instructor.Mode.JSON,
            )
        else:
            self._instructor_client = instructor.from_gemini(
                client=self.gemini_client,
                mode=instructor.Mode.GEMINI_JSON,
            )
        return self._instructor_client

    def get_structured_analysis(self, raw_text: str) -> dict:
        """
        Uses instructor to extract typed metadata from the document text.
        Returns a plain dict so it can be merged into analysis JSON.
        Falls back to sensible defaults on any error — never blocks processing.
        """
        from app.domain.schemas.analysis import DocumentMetadata

        prompt = (
            "Extract structured metadata from the following document.\n\n"
            f"Document (first 4000 chars):\n{raw_text[:4000]}"
        )

        try:
            client = self._get_instructor_client()
            model_kwarg = (
                {"model": self.ollama_model}
                if self.provider == "ollama"
                else {"model": self.gemini_model}
            )
            metadata: DocumentMetadata = client.chat.completions.create(
                **model_kwarg,
                messages=[{"role": "user", "content": prompt}],
                response_model=DocumentMetadata,
            )
            return metadata.model_dump()

        except Exception as e:
            logger.warning(f"Structured analysis via instructor failed (non-fatal): {e}")
            word_count = len(raw_text.split())
            return {
                "key_points": [],
                "document_type": "unknown",
                "contains_financial_data": any(s in raw_text for s in ["$", "USD", "NGN", "€", "revenue"]),
                "contains_personal_data": "@" in raw_text,
                "language": "English",
                "estimated_reading_time_minutes": max(1, word_count // 200),
            }

    # ------------------------------------------------------------------ #
    # SUMMARY GENERATION — STREAMING (SYNC, CELERY-SAFE)                 #
    # ------------------------------------------------------------------ #

    def generate_summary_sync(self, raw_text: str, on_chunk: Optional[Callable[[str], None]] = None) -> str:
        """
        Generates a streaming narrative summary.
        Calls on_chunk(token) for each token so the worker can forward
        chunks to Redis for real-time UI updates.
        """
        summary_limit = 10_000
        prompt = (
            "Summarize the document content below.\n"
            "STRICT RULES:\n"
            "- START IMMEDIATELY with the summary content.\n"
            "- Do NOT say 'Here is a summary' or any introduction.\n"
            "- Provide 3-5 concise, professional sentences.\n\n"
            f"Document Content:\n{raw_text[:summary_limit]}"
        )

        full_summary: list[str] = []
        try:
            if self.provider == "ollama":
                response = self.ollama_client.chat(
                    model=self.ollama_model,
                    messages=[{"role": "user", "content": prompt}],
                    stream=True,
                )
                for chunk in response:
                    content = chunk.get("message", {}).get("content", "")
                    if content:
                        full_summary.append(content)
                        if on_chunk:
                            on_chunk(content)
            else:
                response = self.gemini_client.models.generate_content(
                    model=self.gemini_model,
                    contents=[prompt],
                    config={"stream": True},
                )
                for chunk in response:
                    if chunk.text:
                        full_summary.append(chunk.text)
                        if on_chunk:
                            on_chunk(chunk.text)
        except Exception as e:
            logger.warning(f"Summary generation failed (non-fatal): {e}")

        return self._sanitize_text("".join(full_summary)) or "Summary unavailable."

    # ------------------------------------------------------------------ #
    # GEMINI ASYNC (used by FastAPI routes)                               #
    # ------------------------------------------------------------------ #

    @retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=2, min=10, max=60),
        retry=retry_if_exception_message(match=".*Rate Limit.*|.*429.*"),
    )
    async def _get_gemini_summary(self, file_path: str, mime_type: str) -> str:
        try:
            uploaded_file = self.gemini_client.files.upload(
                file=file_path,
                config={"mime_type": mime_type},
            )
            await asyncio.sleep(2)
            response = self.gemini_client.models.generate_content(
                model=self.gemini_model,
                contents=[
                    "Analyze the document and provide exactly 4 bullet-point key insights.\n"
                    "Rules: no intro, no conclusion, output ONLY bullet points.\n\nDocument:",
                    uploaded_file,
                ],
            )
            return response.text.strip()
        except Exception as e:
            if "429" in str(e):
                raise Exception("Gemini Rate Limit")
            raise Exception(f"Gemini error: {e}")

    # ------------------------------------------------------------------ #
    # HIGH-LEVEL ENTRY POINTS                                             #
    # ------------------------------------------------------------------ #

    async def process(self, file_path: str, mime_type: str | None = None) -> dict:
        """Async path used by direct FastAPI routes (non-Celery)."""
        logger.info(f"Processing document with {self.provider}: {file_path}")
        raw_text = self.extract_text(file_path, mime_type)

        if self.provider == "ollama":
            loop = asyncio.get_running_loop()
            summary = await loop.run_in_executor(
                None, self.generate_summary_sync, raw_text, None
            )
        else:
            summary = await self._get_gemini_summary(file_path, mime_type)

        metadata = self.get_structured_analysis(raw_text)
        return self._format_results(raw_text, summary, metadata)

    def process_sync(self, file_path: str, mime_type: str | None = None, on_chunk: Optional[Any] = None) -> dict:
        """
        Sync path for Celery workers.
        Kept for backward compatibility — prefer calling extract_text() and
        generate_summary_sync() separately so status updates are accurate.
        """
        logger.info(f"process_sync called for: {file_path}")
        raw_text = self.extract_text(file_path, mime_type)
        if not raw_text.strip():
            raise ProcessingError("No text could be extracted from the document.")
        summary = self.generate_summary_sync(raw_text, on_chunk)
        metadata = self.get_structured_analysis(raw_text)
        return self._format_results(raw_text, summary, metadata)

    def _format_results(self, raw_text: str, summary: str, metadata: dict | None = None) -> dict:
        word_count = len(raw_text.split())
        base = {
            "summary": summary,
            "word_count": word_count,
            "estimated_tokens": len(raw_text) // 4,
            "contains_email": "@" in raw_text,
            "contains_money": any(s in raw_text for s in ["$", "USD", "NGN", "€"]),
            "ai_provider": self.provider,
        }
        if metadata:
            base.update(metadata)
        return {
            "raw_text": raw_text,
            "analysis": base,
        }
