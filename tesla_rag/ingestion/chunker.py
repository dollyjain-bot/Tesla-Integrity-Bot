"""
Tesla Integrity Bot - Text Chunking Module
Implements recursive character splitting with overlap to preserve clause boundaries.
"""

from typing import List, Dict, Any, Optional


class TextChunker:
    """Recursively splits text documents into smaller contextual chunks."""

    def __init__(self, chunk_size: int = 500, overlap: int = 100):
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.separators = ["\n\n", "\n", ". ", "? ", "! ", " ", ""]

    def chunk_text(
        self,
        text: str,
        chunk_size: Optional[int] = None,
        overlap: Optional[int] = None,
    ) -> List[str]:
        """Recursively splits a raw text string into chunks."""
        size = chunk_size if chunk_size is not None else self.chunk_size
        lap = overlap if overlap is not None else self.overlap

        if not text or not text.strip():
            return []

        def _recursive_split(content: str, sep_index: int) -> List[str]:
            if not content.strip():
                return []

            if len(content) <= size:
                return [content.strip()]

            if sep_index >= len(self.separators):
                hard_chunks = []
                start = 0
                step = size - lap if (size - lap) > 0 else size
                while start < len(content):
                    chunk = content[start : start + size].strip()
                    if chunk:
                        hard_chunks.append(chunk)
                    start += step
                return hard_chunks

            separator = self.separators[sep_index]
            if separator != "":
                splits = content.split(separator)
            else:
                splits = list(content)

            chunks = []
            current_pieces = []
            current_len = 0

            for piece in splits:
                piece_text = (
                    piece
                    if separator == ""
                    else (piece + separator if piece != splits[-1] else piece)
                )

                if len(piece_text) > size:
                    if current_pieces:
                        accumulated = "".join(current_pieces).strip()
                        if accumulated:
                            chunks.append(accumulated)
                        current_pieces = []
                        current_len = 0

                    sub_chunks = _recursive_split(piece_text, sep_index + 1)
                    chunks.extend(sub_chunks)
                    continue

                if current_len + len(piece_text) > size:
                    accumulated = "".join(current_pieces).strip()
                    if accumulated:
                        chunks.append(accumulated)

                    overlap_pieces = []
                    overlap_len = 0
                    for prev_piece in reversed(current_pieces):
                        overlap_pieces.insert(0, prev_piece)
                        overlap_len += len(prev_piece)
                        if overlap_len >= lap:
                            break

                    current_pieces = overlap_pieces + [piece_text]
                    current_len = sum(len(p) for p in current_pieces)
                else:
                    current_pieces.append(piece_text)
                    current_len += len(piece_text)

            if current_pieces:
                accumulated = "".join(current_pieces).strip()
                if accumulated:
                    chunks.append(accumulated)

            return chunks

        raw_chunks = _recursive_split(text, 0)
        return [c.strip() for c in raw_chunks if c and c.strip()]

    def chunk_documents(
        self,
        documents: List[Dict[str, Any]],
        chunk_size: Optional[int] = None,
        overlap: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Splits a list of document objects into structured chunk dictionaries."""
        all_chunk_objects = []

        for doc in documents:
            raw_text = doc.get("text", "")
            source = doc.get("source", "unknown_source")
            doc_id = doc.get("doc_id") or "UNKNOWN_DOC_ID"
            classification = doc.get("classification") or "CONFIDENTIAL"
            section = doc.get("section_info") or "Full Document"
            effective_date = doc.get("effective_date") or "N/A"

            text_chunks = self.chunk_text(
                raw_text, chunk_size=chunk_size, overlap=overlap
            )

            for idx, chunk_str in enumerate(text_chunks):
                chunk_id = f"{doc_id}_chunk_{idx + 1}"

                chunk_obj = {
                    "chunk_id": chunk_id,
                    "source": source,
                    "document_id": doc_id,
                    "section": section,
                    "classification": classification,
                    "effective_date": effective_date,
                    "chunk_text": chunk_str,
                }
                all_chunk_objects.append(chunk_obj)

        print(
            f"Successfully processed {len(documents)} document(s) into {len(all_chunk_objects)} chunk(s)."
        )
        return all_chunk_objects
