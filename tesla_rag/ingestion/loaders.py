"""
Tesla Integrity Bot - Document Loaders Module
Reads TXT and PDF documents from disk and extracts structured header metadata.
"""

import re
from pathlib import Path
from typing import List, Dict, Any, Optional
import pypdf


class DocumentLoader:
    """Loads plain text and PDF compliance documents from a directory."""

    def __init__(self, data_dir: Path = Path("data")):
        self.data_dir = Path(data_dir)

    def _extract_header_metadata(self, text: str) -> Dict[str, Optional[str]]:
        """Extracts Document ID, Classification, Title, and Effective Date from document headers."""
        metadata = {
            "doc_id": None,
            "classification": None,
            "policy_title": None,
            "effective_date": None,
        }

        doc_id_match = re.search(r"Document ID:\s*([^\n\r]+)", text, re.IGNORECASE)
        if doc_id_match:
            metadata["doc_id"] = doc_id_match.group(1).strip()

        class_match = re.search(r"Classification:\s*([^\n\r]+)", text, re.IGNORECASE)
        if class_match:
            metadata["classification"] = class_match.group(1).strip()

        title_match = re.search(r"Policy Title:\s*([^\n\r]+)", text, re.IGNORECASE)
        if title_match:
            metadata["policy_title"] = title_match.group(1).strip()

        date_match = re.search(r"Effective Date:\s*([^\n\r]+)", text, re.IGNORECASE)
        if date_match:
            metadata["effective_date"] = date_match.group(1).strip()

        return metadata

    def load_txt_files(self) -> List[Dict[str, Any]]:
        """Loads all .txt policy files from data_dir."""
        documents = []
        if not self.data_dir.exists():
            print(f"Warning: Directory '{self.data_dir}' does not exist.")
            return documents

        txt_files = sorted(list(self.data_dir.glob("*.txt")))
        print(f"Found {len(txt_files)} text files in '{self.data_dir}'.")

        for file_path in txt_files:
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()

                parsed_meta = self._extract_header_metadata(content)

                doc_obj = {
                    "text": content,
                    "source": file_path.name,
                    "page_number": None,
                    "doc_id": parsed_meta["doc_id"] or "TSLA-ETH-DOC",
                    "classification": parsed_meta["classification"] or "CONFIDENTIAL",
                    "policy_title": parsed_meta["policy_title"],
                    "effective_date": parsed_meta["effective_date"],
                    "section_info": "Full Document",
                }
                documents.append(doc_obj)
                print(f"Loaded TXT: '{file_path.name}' | Doc ID: {doc_obj['doc_id']}")

            except Exception as e:
                print(f"Error reading TXT file '{file_path.name}': {e}")

        return documents

    def load_pdf_files(self) -> List[Dict[str, Any]]:
        """Loads and extracts text page-by-page from all .pdf files."""
        documents = []
        if not self.data_dir.exists():
            return documents

        pdf_files = sorted(list(self.data_dir.glob("*.pdf")))
        print(f"Found {len(pdf_files)} PDF files in '{self.data_dir}'.")

        for file_path in pdf_files:
            try:
                reader = pypdf.PdfReader(str(file_path))
                total_pages = len(reader.pages)

                first_page_text = (
                    reader.pages[0].extract_text() if total_pages > 0 else ""
                )
                parsed_meta = self._extract_header_metadata(first_page_text)

                for page_idx, page in enumerate(reader.pages):
                    page_text = page.extract_text()
                    page_num = page_idx + 1

                    if page_text and page_text.strip():
                        doc_obj = {
                            "text": page_text,
                            "source": file_path.name,
                            "page_number": page_num,
                            "doc_id": parsed_meta["doc_id"] or "TSLA-PDF-DOC",
                            "classification": parsed_meta["classification"] or "CONFIDENTIAL",
                            "policy_title": parsed_meta["policy_title"],
                            "effective_date": parsed_meta["effective_date"],
                            "section_info": f"Page {page_num} of {total_pages}",
                        }
                        documents.append(doc_obj)

                print(f"Loaded PDF: '{file_path.name}' | {total_pages} total pages")

            except Exception as e:
                print(f"Error reading PDF file '{file_path.name}': {e}")

        return documents

    def load_all_documents(self) -> List[Dict[str, Any]]:
        """Loads both TXT and PDF documents from the configured directory."""
        txt_docs = self.load_txt_files()
        pdf_docs = self.load_pdf_files()
        all_docs = txt_docs + pdf_docs
        print(f"Total documents loaded: {len(all_docs)}")
        return all_docs
