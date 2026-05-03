"""
PDF Chunking Processor for GraphRAG
Adapted from chunking.ipynb
"""
import os
import json
import hashlib
import unicodedata
import re
import logging
import pdfplumber
from typing import List, Dict, Any
from langchain_text_splitters import RecursiveCharacterTextSplitter

logger = logging.getLogger(__name__)

# Configuration (should match config.py)
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150


def clean_text(text: str) -> str:
    """
    Clean and normalize text from PDF extraction errors
    """
    if not text:
        return ""
    
    # Normalize Unicode
    text = unicodedata.normalize("NFC", text)
    
    # Fix common OCR errors specific to Vietnamese documents
    fix_map = {
        "SÖT": "SÚT",
        "HƯỚNG": "HƯỚNG",
        "Ơ": "Ư",
        "thƯ": "thư",
        "TRỊ": "TRỊ",
        "đối tƯỢng": "đối tượng",
        "tƯ": "tư",
        "Việt": "Việt",
        "chiến": "chiến"
    }
    for k, v in fix_map.items():
        text = text.replace(k, v)
    
    # Remove page numbers (isolated numbers on new lines)
    text = re.sub(r'\n\s*\d+\s*\n', '\n', text)
    
    # Fix hyphenated words at line breaks
    text = re.sub(r'(\w+)-\n\s*(\w+)', r'\1\2', text)
    
    # Replace multiple spaces/tabs with single space
    text = re.sub(r'[ \t]+', ' ', text)
    
    return text.strip()


def extract_tables_to_md(page) -> str:
    """
    Extract tables from page and convert to Markdown format
    """
    tables = page.extract_tables()
    table_md = ""
    if tables:
        for table in tables:
            table_md += "\n\n[DỮ LIỆU BẢNG TRA CỨU]:\n"
            for row in table:
                if any(row):
                    # Clean each cell and replace empty cells with "-"
                    r = [clean_text(str(cell)) if cell else "-" for cell in row]
                    table_md += "| " + " | ".join(r) + " |\n"
                    # Add separator row after header if it looks like header
                    if table.index(row) == 0:
                        table_md += "| " + " | ".join(["---"] * len(r)) + " |\n"
    return table_md


def get_section_title(text: str, current_title: str) -> str:
    """
    Try to detect section titles from text patterns
    """
    section_patterns = [
        r'(Bài\s+\d+[:\.]?.*)',
        r'(Chương\s+\d+[:\.]?.*)',
        r'(Phần\s+[IVX]+[:\.]?.*)',
        r'(Phụ lục\s+\d+[:\.]?.*)',
        r'(\d+\.\d+\.?\s+[A-ZĐ].*)',
        r'(Mục\s+\d+[:\.]?.*)'
    ]
    
    for pat in section_patterns:
        found = re.search(pat, text)
        if found:
            new_title = found.group(1).strip()
            # Only update if reasonably short
            if len(new_title) < 100:
                return new_title
    
    return current_title


def process_pdf(
    filepath: str,
    doc_type: str,
    risk_priority: str,
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP
) -> List[Dict[str, Any]]:
    """
    Process a single PDF file and return chunks
    
    Args:
        filepath: Path to PDF file
        doc_type: Type of document (medical_guideline, first_aid, school_counseling)
        risk_priority: Priority level (high, medium, low)
        chunk_size: Maximum chunk size
        chunk_overlap: Overlap between chunks
        
    Returns:
        List of chunk dictionaries with page_content and metadata
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")
    
    logger.info(f"Processing PDF: {filepath}")
    
    # Initialize text splitter
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=[
            "\n\n[MỤC:",
            "\n\n",
            "\n",
            ". ",
            " ",
            ""
        ]
    )
    
    final_data = []
    seen_hashes = set()
    current_section = "Thông tin chung"
    
    with pdfplumber.open(filepath) as pdf:
        for page_num, page in enumerate(pdf.pages, 1):
            raw_text = page.extract_text() or ""
            
            # Skip table of contents or index pages
            if "MỤC LỤC" in raw_text.upper():
                continue
            if len(re.findall(r'\.{10,}', raw_text)) > 5:
                continue
            
            cleaned_text = clean_text(raw_text)
            if len(cleaned_text) < 50:  # Skip very short pages
                continue
            
            # Update section title
            current_section = get_section_title(cleaned_text, current_section)
            
            # Extract tables
            table_content = extract_tables_to_md(page)
            
            # Combine text and tables with metadata
            full_content = f"[NGUỒN: {os.path.basename(filepath)}] - [MỤC: {current_section}]\n{cleaned_text}\n{table_content}"
            
            # Split into chunks
            chunks = splitter.split_text(full_content)
            
            for chunk in chunks:
                # Skip very short chunks
                if len(chunk.strip()) < 120:
                    continue
                
                # Deduplicate using hash
                text_hash = hashlib.md5(chunk.encode()).hexdigest()
                if text_hash in seen_hashes:
                    continue
                seen_hashes.add(text_hash)
                
                # Create chunk entry
                chunk_data = {
                    "page_content": chunk.strip(),
                    "metadata": {
                        "source": filepath,
                        "source_filename": os.path.basename(filepath),
                        "section": current_section,
                        "risk_priority": risk_priority,
                        "doc_type": doc_type,
                        "page_no": page_num,
                        "chunk_id": f"{os.path.basename(filepath)}_p{page_num}_{text_hash[:8]}"
                    }
                }
                final_data.append(chunk_data)
    
    logger.info(f"Generated {len(final_data)} chunks from {filepath}")
    return final_data


def process_all_documents(docs_config: List[Dict[str, str]]) -> List[Dict[str, Any]]:
    """
    Process multiple PDF documents according to configuration
    
    Args:
        docs_config: List of dicts with keys: path, type, risk
        
    Returns:
        Combined list of all chunks from all documents
    """
    all_chunks = []
    
    for doc in docs_config:
        try:
            chunks = process_pdf(
                filepath=doc["path"],
                doc_type=doc["type"],
                risk_priority=doc["risk"]
            )
            all_chunks.extend(chunks)
            logger.info(f"✓ Processed {doc['path']}: {len(chunks)} chunks")
        except Exception as e:
            logger.error(f"✗ Failed to process {doc['path']}: {e}")
            continue
    
    logger.info(f"Total chunks from all documents: {len(all_chunks)}")
    return all_chunks


def save_chunks_to_jsonl(chunks: List[Dict[str, Any]], output_path: str):
    """
    Save chunks to JSONL format (one JSON object per line)
    """
    with open(output_path, 'w', encoding='utf-8') as f:
        for chunk in chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + '\n')
    logger.info(f"Saved {len(chunks)} chunks to {output_path}")


def load_chunks_from_jsonl(input_path: str) -> List[Dict[str, Any]]:
    """
    Load chunks from JSONL file
    """
    chunks = []
    with open(input_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                chunks.append(json.loads(line))
    logger.info(f"Loaded {len(chunks)} chunks from {input_path}")
    return chunks


# Default configuration matching the design document
DEFAULT_DOCS_CONFIG = [
    {
        "path": "resources/pdtt.pdf",
        "type": "medical_guideline",
        "risk": "high"
    },
    {
        "path": "resources/sctl.pdf",
        "type": "first_aid",
        "risk": "medium"
    },
    {
        "path": "resources/tvtl.pdf",
        "type": "school_counseling",
        "risk": "medium"
    }
]


if __name__ == "__main__":
    import logging
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Process all documents
    chunks = process_all_documents(DEFAULT_DOCS_CONFIG)
    
    # Save to file
    output_file = "data/processed_chunks.jsonl"
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    save_chunks_to_jsonl(chunks, output_file)
    
    print(f"\n✅ Chunking complete. Total chunks: {len(chunks)}")
    print(f"Output saved to: {output_file}")