from typing import List, Dict, Any, Tuple
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
    papers: List[Dict[str, Any]], section_title: str
) -> np.ndarray:
    num_papers = len(papers)
    matrix = np.zeros((num_papers, num_papers))
    for i in range(num_papers):
        for j in range(i, num_papers):
            matrix[i, j] = similarity_by_section(
                papers[i]["embedding_text"][section_title],
                papers[j]["embedding_text"][section_title],
            )
    return matrix


def get_top_pairs(
    papers: List[Dict[str, Any]], num_pairs: int, section_title: str
) -> List[Tuple[int, int]]:
    matrix = generate_similarity_matrix(papers, section_title)

    # 复制上三角矩阵到下三角，得到完整的对称矩阵
    full_matrix = matrix + matrix.T - np.diag(np.diag(matrix))

    # 设置对角线为-1以排除自身与自身的比较
    np.fill_diagonal(full_matrix, -1)

    # 找到最大的num_pairs个元素的索引
    # 将矩阵展平，找到最大值的索引
    flat_indices = np.argsort(full_matrix.flat)[-num_pairs:]

    # 将展平的索引转换回二维索引
    pairs = []
    for idx in flat_indices:
        # 计算对应的行列号
        i, j = np.unravel_index(idx, full_matrix.shape)
        pairs.append((i, j))

    # 按相似度降序排列
    pairs.sort(key=lambda p: full_matrix[p[0], p[1]], reverse=True)

    return pairs
