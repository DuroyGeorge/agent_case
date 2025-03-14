from typing import List, Dict, Any
import numpy as np
from sentence_transformers import SentenceTransformer
import time
import logging
import re
import torch

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)


def embedding_section(section_text: str) -> np.ndarray:
    """
    Encode text into vector representation

    Parameters:
        section_text: A single string of text
        batch_size: Size of batch processing
        normalize: Whether to normalize the vector

    Returns:
        Text embedding vectors in numpy array format
    """
    # Split text by periods, question marks, and exclamation marks
    sentences = re.split(r"[.!?]", section_text)
    # Filter out empty sentences
    sentences = [s.strip() for s in sentences if s.strip()]

    # Record time
    start_time = time.time()

    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    # Use SentenceTransformers API
    embeddings = (
        SentenceTransformer("all-mpnet-base-v2")
        .to(device)
        .encode(
            sentences,
            batch_size=32,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
    )

    elapsed = time.time() - start_time
    logger.debug(
        f"Encoding completed, processed {len(sentences)} sentences, time elapsed: {elapsed:.2f}s"
    )

    return embeddings


def embedding(papers: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    for paper in papers:
        paper["embedding_text"] = {
            key: embedding_section(value)
            for key, value in paper["summary_text"].items()
        }
    return papers


def similarity_by_sentence(sentenceA: np.ndarray, sectionB: np.ndarray) -> float:
    return np.max([np.dot(sentenceA, b) for b in sectionB])


def similarity_by_section(sectionA: np.ndarray, sectionB: np.ndarray) -> float:
    return np.mean(
        [similarity_by_sentence(a, sectionB) for a in sectionA]
        + [similarity_by_sentence(b, sectionA) for b in sectionB]
    )


def generate_similarity_matrix(
    papers: List[Dict[str, Any]], sectionA_title: str, sectionB_title: str
) -> np.ndarray:
    num_papers = len(papers)
    matrix = np.zeros((num_papers, num_papers))
    for i in range(num_papers):
        for j in range(i, num_papers):
            matrix[i, j] = similarity_by_section(
                papers[i]["embedding_text"][sectionA_title],
                papers[j]["embedding_text"][sectionB_title],
            )
    return matrix
