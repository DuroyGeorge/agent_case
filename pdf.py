from pathlib import Path
import logging
import PyPDF2
from typing import Dict, Any, List
import re

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)


def extract_text_from_pdf(pdf_path: Path) -> List[str]:
    """Extract text content from PDF file"""
    logger.debug(f"Extracting text from PDF: {pdf_path}")
    try:
        text = []
        with open(pdf_path, "rb") as file:
            reader = PyPDF2.PdfReader(file)
            logger.debug(f"Extracted {len(reader.pages)} pages from PDF")
            for page_num in range(len(reader.pages)):
                page = reader.pages[page_num]
                text.append(page.extract_text())
        logger.debug(f"Successfully extracted {len(text)} pages from {pdf_path}")
        return text
    except Exception as e:
        logger.error(f"Error extracting text from {pdf_path}: {str(e)}", exc_info=True)
        return []


def process_bookmarks(reader: PyPDF2.PdfReader) -> List[Dict[str, Any]]:
    """
    Process PDF top-level bookmarks and convert to an easy-to-handle format
    """
    result = []
    bookmarks = reader.outline
    if not bookmarks:
        return result
    page_id_to_index = {}
    for i, page in enumerate(reader.pages):
        if hasattr(page, "indirect_reference") and hasattr(
            page.indirect_reference, "idnum"
        ):
            page_id_to_index[page.indirect_reference.idnum] = i + 1

    for item in bookmarks:
        # Only process dictionary-type bookmark entries
        if isinstance(item, dict) and "/Title" in item:
            page_ref = item.get("/Page", "未知")
            object_id = "未知"
            actual_page = "未知"

            if hasattr(page_ref, "idnum"):
                object_id = page_ref.idnum
                # Find the corresponding actual page number
                actual_page = page_id_to_index.get(object_id, "未找到")
            bookmark = {
                "title": item["/Title"],
                "page_num": actual_page,
            }
            result.append(bookmark)
    logger.debug(f"Extracted {result} from PDF")
    return result


def extract_structure_from_pdf(pdf_path: Path) -> Dict[str, Any]:
    """
    Extract structure information from PDF file, including table of contents and text content
    Returns a dictionary containing document structure and text content
    """
    logger.debug(f"Extracting PDF structure information: {pdf_path}")
    result = {
        "title": pdf_path.stem,
        "bookmarks": [],
        "pages": [],
    }

    try:
        with open(pdf_path, "rb") as file:
            reader = PyPDF2.PdfReader(file)

            if hasattr(reader, "outline") and reader.outline:
                result["bookmarks"] = process_bookmarks(reader)
                logger.debug(
                    f"Extracted {len(result['bookmarks'])} bookmark items from PDF"
                )
                logger.debug(f"Extracted bookmark structure: {result['bookmarks']}")
            else:
                logger.debug("No built-in bookmarks found in PDF")
            result["pages"] = extract_text_from_pdf(pdf_path)

        return result
    except Exception as e:
        logger.error(f"Error extracting PDF structure: {str(e)}", exc_info=True)
        return {
            "title": pdf_path.stem,
            "bookmarks": [],
            "pages": extract_text_from_pdf(pdf_path),
        }


def preprocess_text(structure: Dict[str, Any]) -> Dict[str, Any]:
    """
    Split text content into different sections according to bookmarks
    """
    result = {}
    pages_text = structure["pages"]
    bookmarks = structure["bookmarks"]

    # If no bookmarks or text, return directly
    if not bookmarks or not pages_text:
        result["full_text"] = "\n".join(pages_text)
        return result

    # Traverse all bookmarks, extract corresponding section text
    for i, bookmark in enumerate(bookmarks):
        # Get current bookmark page number
        start_page = bookmark["page_num"]

        # Get next bookmark page number as end page
        if i < len(bookmarks) - 1:
            end_page = bookmarks[i + 1]["page_num"]
        else:
            end_page = len(pages_text)

        # Get all page text for this section and merge
        chapter_text = ""
        for page_idx in range(start_page - 1, end_page):
            # Clean page text
            page_content = pages_text[page_idx]
            # Remove extra spaces and line breaks
            page_content = re.sub(r"\s+", " ", page_content)
            # Remove reference marks like [1], [2]
            page_content = re.sub(r"\[\d+\]", "", page_content)
            # Remove header/footer common formats
            page_content = re.sub(r"^\d+\s*$", "", page_content)
            # Remove other possible interference characters
            page_content = re.sub(r"[•◦]", "", page_content)
            chapter_text += page_content + "\n"

        # Save to result
        result[bookmark["title"]] = chapter_text.strip()
    result["full_text"] = "\n".join(
        [f"**{key}**\n{value}" for key, value in result.items()]
    )
    return result


def main() -> List[Dict[str, Any]]:
    pdfs = Path("./papers").glob("*.pdf")
    result = []
    for pdf in pdfs:
        structure = extract_structure_from_pdf(pdf)
        if structure["pages"]:
            preprocessed_text = preprocess_text(structure)
            result.append(
                {
                    "title": structure["title"],
                    "bookmarks": structure["bookmarks"],
                    "text": preprocessed_text,
                }
            )
    return result


if __name__ == "__main__":
    main()
