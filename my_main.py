from pathlib import Path
import arxiv
import aiohttp
import asyncio
from typing import List, Dict, Any, Optional, Tuple
import os
from dotenv import load_dotenv
import logging
import re

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

# Load environment variables from .env file
load_dotenv()

# =======================
# Configuration Constants
# =======================
PAPER_STORAGE_DIR = os.getenv("PAPER_STORAGE_DIR", "./papers")

LLM_API_KEY = os.getenv("LLM_API_KEY")
SERPAPI_API_KEY = os.getenv("SERPAPI_API_KEY")
JINA_API_KEY = os.getenv("JINA_API_KEY")

# Endpoints
LLM_BASE_URL = os.getenv("LLM_BASE_URL")
SERPAPI_URL = os.getenv("SERPAPI_URL")
JINA_BASE_URL = os.getenv("JINA_BASE_URL")

# Default LLM model (can be changed if desired)
DEFAULT_MODEL = os.getenv("DEFAULT_MODEL")


async def call_llm(session, messages, model=DEFAULT_MODEL):
    """
    Asynchronously call the LLM chat completion API with the provided messages.
    Returns the content of the assistant’s reply.
    """
    headers = {
        "Authorization": f"Bearer {LLM_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {"model": model, "messages": messages}
    try:
        async with session.post(LLM_BASE_URL, headers=headers, json=payload) as resp:
            if resp.status == 200:
                result = await resp.json()
                try:
                    return result["choices"][0]["message"]["content"]
                except (KeyError, IndexError) as e:
                    logger.error("Unexpected LLM response structure:", result)
                    return None
            else:
                text = await resp.text()
                logger.error(f"LLM API error: {resp.status} - {text}")
                return None
    except Exception as e:
        logger.error("Error calling LLM:", e)
        return None


# =========================================================
# Get pdf papers from arxiv(maybe some other sources later)
# =========================================================


async def search_paper(topic: str, max_results: int = 10) -> List[Dict[str, Any]]:
    try:
        search = arxiv.Search(query=topic, max_results=max_results)
        papers = []
        for result in search.results():
            paper = {
                "title": result.title,
                "pdf_url": result.pdf_url,
            }
            papers.append(paper)
        logger.info(f"Found {len(papers)} papers matching '{topic}'")
        return papers
    except Exception as e:
        logger.error(
            f"Error retrieving papers for topic '{topic}': {str(e)}", exc_info=True
        )
        return []


async def download_paper(
    session: aiohttp.ClientSession,
    paper: Dict[str, Any],
    save_dir: Path = PAPER_STORAGE_DIR,
) -> Optional[Path]:
    title = paper["title"]
    logger.debug(f"Attempting to download paper: '{title}'")
    pdf_url = paper["pdf_url"]

    if not pdf_url:
        logger.warning(f"No PDF URL found for paper: '{title}'")
        return None

    filename = save_dir / f"{title}.pdf"
    try:
        logger.debug(f"Connecting to {pdf_url}")
        async with session.get(pdf_url) as resp:
            if resp.status == 200:
                stream = await aiohttp.StreamReader.read(resp.content)
                with open(filename, "wb") as f:
                    f.write(stream)
                logger.info(f"Successfully downloaded paper to: {filename}")
                return filename
            else:
                logger.error(f"Failed to download paper, status code: {resp.status}")
                return None
    except Exception as e:
        logger.error(f"Error downloading paper '{title}': {str(e)}", exc_info=True)
        return None


async def paper_main(topic: str, max_paper: int) -> List[Dict[str, Any]]:
    logger.info(f"Starting paper search and download process for '{topic}'")
    save_dir = Path(PAPER_STORAGE_DIR)
    if not save_dir.exists():
        logger.info(f"Creating directory: {save_dir}")
        save_dir.mkdir(parents=True, exist_ok=True)
    else:
        logger.info(f"Cleaning existing PDF files in: {save_dir}")
        for pdf_file in save_dir.glob("*.pdf"):
            logger.debug(f"Removing existing file: {pdf_file}")
            pdf_file.unlink()

    papers = await search_paper(topic, max_paper)
    if not papers:
        logger.warning(f"No papers found for '{topic}', process complete")
        return

    results = [download_paper(paper, save_dir) for paper in papers]
    results = await asyncio.gather(*results)

    logger.info(f"Process complete. Downloaded {len(results)}/{len(papers)} papers")
    return papers


# ===================================
# PDF Processing
# ===================================
import PyPDF2


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
    page_id_to_index = {}
    for i, page in enumerate(reader.pages):
        if hasattr(page, "indirect_reference") and hasattr(
            page.indirect_reference, "idnum"
        ):
            page_id_to_index[page.indirect_reference.idnum] = i + 1

    # FIXME 如果没有abstract怎么办?
    result.append({"title": "abstract", "page_num": 1})
    for item in bookmarks:
        # Only process dictionary-type bookmark entries
        if isinstance(item, dict) and "/Title" in item:
            page_ref = item.get("/Page", "unknown")
            object_id = "unknown"
            actual_page = "unknown"

            if hasattr(page_ref, "idnum"):
                object_id = page_ref.idnum
                # Find the corresponding actual page number
                actual_page = page_id_to_index.get(object_id, "unknown")
            if item["/Title"].lower() == "abstract":
                continue
            bookmark = {
                "title": item["/Title"].lower(),
                "page_num": actual_page,
            }
            result.append(bookmark)
    logger.debug(f"Extracted {result} from PDF")
    return result


def preprocess_text(paper: Dict[str, Any]):
    """
    Split text content into different sections according to bookmarks
    """
    paper["section_title_to_text"] = {}
    bookmarks = paper["bookmarks"]
    pages_text = paper["pages"]

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

        # Save to paper
        paper["section_title_to_text"][bookmark["title"]] = chapter_text.strip()
    paper["full_text"] = "\n".join(
        [f"**{key}**\n{value}" for key, value in paper["section_title_to_text"].items()]
    )
    logger.debug(f"Preprocessed paper: {paper["title"]}")


def merge_section_text(paper: Dict[str, Any]) -> Dict[str, Any]:
    section_title_to_text = paper["section_title_to_text"]
    """
    classify section into these categories:
    - Abstract
    - Introduction
    - Body
    - Discussion
    - Conclusion
    """
    # 创建标准类别
    standard_categories = {
        "abstract": "",
        "introduction": "",
        "body": "",
        "discussion": "",
        "conclusion": "",
    }

    # 关键词映射到标准类别
    category_keywords = {
        "abstract": ["abstract", "summary", "overview"],
        "introduction": [
            "introduction",
            "background",
            "preliminaries",
            "motivation",
            "related work",
        ],
        "body": [
            "method",
            "model",
            "approach",
            "algorithm",
            "implementation",
            "experiment",
            "architecture",
            "system",
            "design",
            "proposed",
        ],
        "discussion": [
            "discussion",
            "analysis",
            "evaluation",
            "result",
            "performance",
            "comparison",
            "ablation",
            "finding",
            "experiments",
        ],
        "conclusion": ["conclusion", "future work", "limitation", "final"],
    }

    # 检测章节序号的正则模式
    section_number_pattern = re.compile(r"^\d+\.?\s*|^[ivxlcdm]+\.?\s*", re.IGNORECASE)

    # 分配每个章节到标准类别
    logger.debug(f"Merging sections for paper: {paper.get('title', 'Unknown')}")
    for section_title, text in section_title_to_text.items():
        # 预处理章节标题：移除数字前缀，转为小写
        cleaned_title = section_number_pattern.sub("", section_title.lower())

        # 匹配到的类别
        matched_category = None

        # 根据关键词匹配类别
        for category, keywords in category_keywords.items():
            for keyword in keywords:
                if keyword in cleaned_title:
                    matched_category = category
                    logger.debug(
                        f"Section '{section_title}' matched to category '{category}' by keyword '{keyword}'"
                    )
                    break
            if matched_category:
                break

        # 使用启发式规则处理未匹配的情况
        if not matched_category:
            # 第一部分通常是摘要或引言
            if section_title == list(section_title_to_text.keys())[0]:
                if len(text.split()) < 300:  # 短文本可能是摘要
                    matched_category = "abstract"
                else:
                    matched_category = "introduction"
                logger.debug(
                    f"First section '{section_title}' assigned to '{matched_category}' by position"
                )
            # 最后部分通常是结论
            elif section_title == list(section_title_to_text.keys())[-1]:
                matched_category = "conclusion"
                logger.debug(
                    f"Last section '{section_title}' assigned to 'conclusion' by position"
                )
            # 其他情况默认为主体部分
            else:
                matched_category = "body"
                logger.debug(f"Section '{section_title}' defaulted to 'body' category")

        # 将文本添加到对应类别
        if standard_categories[matched_category]:
            standard_categories[matched_category] += f"\n\n"
        standard_categories[matched_category] += text

    # 保存合并后的类别文本
    paper["section_title_to_text"] = standard_categories
    logger.debug(
        f"Successfully merged sections into standard categories for {paper.get('title', 'Unknown')}"
    )


def extract_structure_from_pdf(pdf_path: Path, paper: Dict[str, Any]):
    """
    Extract structure information from PDF file, including table of contents and text content
    """
    logger.debug(f"Extracting PDF structure information: {pdf_path}")
    try:
        with open(pdf_path, "rb") as file:
            reader = PyPDF2.PdfReader(file)

            if hasattr(reader, "outline") and reader.outline:
                paper["bookmarks"] = process_bookmarks(reader)
                logger.debug(
                    f"Extracted {len(paper['bookmarks'])} bookmark items from PDF"
                )
                logger.debug(f"Extracted bookmark structure: {paper['bookmarks']}")
            else:
                logger.debug("No built-in bookmarks found in PDF")
                # If no bookmarks, return directly
                return
            paper["pages"] = extract_text_from_pdf(pdf_path)

    except Exception as e:
        logger.error(f"Error extracting PDF structure: {str(e)}", exc_info=True)


def pdf_process_main(papers: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    pdfs = Path(PAPER_STORAGE_DIR).glob("*.pdf")
    for pdf in pdfs:
        for paper in papers:
            if pdf.stem == paper["title"]:
                extract_structure_from_pdf(pdf, paper)
                if paper["pages"]:
                    preprocess_text(paper)
    processed_papers = []
    # TODO 可以把以后用不到的属性del掉，减少传输数据的开销
    for paper in papers:
        if "pages" in paper:
            processed_papers.append(paper)
    for paper in processed_papers:
        merge_section_text(paper)
    return processed_papers


# ============================
# Summary text for per section
# ============================


async def summary_section(session, title: str, section_title: str, text: str) -> str:
    sys_prompt = """You are an expert in summarizing academic papers."""
    prompt = f"""Title: {title}
Section Title: {section_title}
Text: {text}
Please provide a concise summary of the section of the paper.
Don't leave out any important details."""
    messages = [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": prompt},
    ]
    response = await call_llm(session, messages)
    return response


async def summary(session, papers: List[Dict[str, Any]]):
    for paper in papers:
        paper["section_title_to_summary"] = {}
        paper_summary = ""
        res = [
            (
                section_title,
                summary_section(session, paper["title"], section_title, section_text),
            )
            for section_title, section_text in paper["section_title_to_text"].items()
        ]
        summaries = await asyncio.gather(*res)
        for summary in summaries:
            paper_summary += f"{summary}\n\n"
            paper["section_title_to_summary"][summary[0]] = summary[1]
        paper["full_summary"] = paper_summary


# ======================
# Embedding
# =======================
import numpy as np
from sentence_transformers import SentenceTransformer
import time
import torch


def embedding_section(section_text: str) -> np.ndarray:
    """
    Encode text into vector representation

    Args:
        section_text: A single string of text

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


def embedding(papers: List[Dict[str, Any]]):
    for paper in papers:
        paper["section_title_to_embedding"] = {
            key: embedding_section(value)
            for key, value in paper["section_title_to_summary"].items()
        }


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
                papers[i]["section_title_to_embedding"][section_title],
                papers[j]["section_title_to_embedding"][section_title],
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


# ===================
# Report Generation
# ===================


def help_show_summary(papers: List[Dict[str, Any]]) -> str:
    res = ""
    for paper in papers:
        res += f"paper title:{paper['title']}\nsummary:{paper['summary_text']}\n"
    return res


def help_show_pair(papers: List[Dict[str, Any]], pairs: List[Tuple[int, int]]) -> str:
    res = ""
    for i, (a, b) in enumerate(pairs):
        res += f"Pair {i+1}:\n{papers[a]["title"]}:\n{papers[a]["text"]}\n{papers[b]["title"]}:\n{papers[b]["text"]}\n"
    return res


async def report_abstract(
    session: aiohttp.ClientSession,
    topic: str,
    papers: List[Dict[str, Any]],
    pairs: List[Tuple[int, int]],
) -> str:
    sys_prompt = f"""You are a senior researcher specializing in {topic} with 15+ years of experience, commissioned to write a comprehensive review on {topic}. Focus solely on crafting the Abstract section.
    """
    prompt = f"""Key findings from seminal papers:

{help_show_summary(papers)}
Besides the above content, the following paper pairs have similar points, I will show the original text to you, please pay special attention to them:

{help_show_pair(papers, pairs)}
Writing Requirements
Develop a 500-word abstract following these structural components:

Context (2 sentences):
1st sentence: Highlight field significance using [Term 1] and [Term 2]
2nd sentence: Articulate current bottlenecks (reference conflicting evidence from [Paper C])
Review Value (1 sentence):
Emphasize unique contributions (e.g., "First synthesis of [Approach X] and [Theory Y] perspectives")
Methodology (3 pillars):
Systematic review of [X] milestone papers (past [3/5] years)
Analytical framework incorporating [Dimension 1], [Dimension 2], [Dimension 3]
Cross-comparison using [evaluation metric]
Key Insights:
Demonstrate [Technique A]'s superiority in [Scenario 1] (cite [Paper B])
Expose limitations of [Method C] (reference [Paper D]'s empirical data)
Reveal [Phenomenon E]-[Factor F]** correlation (supported by [Papers A,C])
Future Directions:
Short-term: Address [Challenge 1] (extend [Paper A]'s unfinished work)
Mid-term: Develop [Framework/Tool] (align with [Paper C]'s recommendations)
Long-term: Overcome [Theoretical Barrier] (building on [Paper D]'s projections)
Style Guidelines

Formal academic tone with precise terminology
Avoid subjective language
Italicize non-English terms at first mention (e.g., "Term")
Emulate writing standards in Nature Reviews series
Execute stepwise:

Outline logical structure
Populate content blocks
Refine for academic rigor

Please use markdown format to output
    """
    messages = [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": prompt},
    ]
    response = await call_llm(session, messages)
    return response


async def report_introduction(
    session: aiohttp.ClientSession,
    topic: str,
    papers: List[Dict[str, Any]],
    pairs: List[Tuple[int, int]],
) -> str:
    sys_prompt = f"""As the lead author of a high-impact review in [Journal Name] (IF>20), you are tasked with composing an authoritative Introduction section that contextualizes {topic} within broader academic discourse.
"""

    prompt = f"""Curated knowledge base includes:

{help_show_summary(papers)}
Besides the above content, the following paper pairs have similar points, I will show the original text to you, please pay special attention to them:

{help_show_pair(papers, pairs)}

Structural Framework
Develop 800-1000 words organized as:

Contextual Layer (3 paragraphs)
Historical Arc: Trace evolution from [Year1]'s [Theory1] to [Year2]'s [Breakthrough2]
Disciplinary Matrix: Map relationships between [Subfield1], [Subfield2], and [Subfield3]
Societal Impact: Quantify real-world effects using [Metric1] (↑[X]% since 2015)
Critical Analysis (2 paragraphs)
Convergence Points: Synthesize consensus on [Principle1] (supported by [5+ Papers])
Scholarly Divides: Contrast [School of Thought A] vs [School of Thought B] positions
Motivational Framework
Imperative Statement: "The urgent need to reconcile [Conflict1] with [Conflict2] demands..."
Limitations Synthesis: Cluster 3 types of methodological constraints from [Appendix Table1]
Novelty Declaration: "This review uniquely integrates [Perspective1] and [Dataset Type2] approaches to..."
Architectural Preview
Methodological Signature: Highlight [Analytical Framework Name]'s distinct phases
Roadmap Statement: "Section II deciphers...while Section V pioneers..."
Citation Protocol

Prioritize papers with 500+ citations for foundational claims
Use recent (2023-2024) preprints for cutting-edge assertions
Balance disciplinary sources (min. 30% from [Adjacent Field])
Tone Management

Begin with accessible analogies (e.g., "Much like [Common Phenomenon]...")
Gradually intensify technicality using [Field-Specific Jargon]
Conclude with 3 nested rhetorical questions aligning with [Review's Core Thesis]
Revision Checklist
□ Verify temporal progression logic
□ Cross-validate disciplinary impact claims
□ Ensure gap statements directly feed into contribution claims
    """

    messages = [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": prompt},
    ]

    response = await call_llm(session, messages)
    return response


async def report_body(
    session: aiohttp.ClientSession,
    topic: str,
    papers: List[Dict[str, Any]],
    pairs: List[Tuple[int, int]],
) -> str:
    sys_prompt = f"""You are a {topic} scholarly architect tasked with generating structured review introductions by synthesizing core research elements. Implement this modular construction method:
    """
    prompt = f"""Knowledge Integration
Synthesize data from:

{help_show_summary(papers)}
Besides the above content, the following paper pairs have similar points, I will show the original text to you, please pay special attention to them:

{help_show_pair(papers, pairs)}

Structural Architecture
Develop 2000-2500 words through 5 interlocked modules:

1. Content Focus:
   - Extract and integrate core concepts, research methods, and key findings from the provided literature.
   - Highlight agreements, controversies, and differing perspectives among the studies.
   - Analyze the strengths, weaknesses, and limitations of the various methodologies and findings.
   - Critically assess the current state of research, identifying gaps or areas needing further exploration.

2. Structural Guidelines:
   - Organize the main discussion section in a clear, logical structure.
   - Break down the content into subsections if needed (e.g., theoretical foundations, methodological approaches, data analysis, and emerging trends/future directions).
   - Ensure each paragraph or subsection centers on a specific idea or theme, supporting it with detailed explanations and citing relevant studies where applicable (e.g., [Author, Year]).

3. Academic Tone and Style:
   - Utilize formal, precise, and scholarly language.
   - Emphasize clarity and coherence, ensuring that each point is well-supported with data or examples from the literature.
   - Provide critical analysis and synthesis rather than merely summarizing individual studies.

4. Analytical and Critical Perspective:
   - Compare and contrast the various viewpoints and research findings.
   - Offer an objective critique of the methodologies and conclusions discussed.
   - Discuss the implications of the findings for the overall field and suggest potential directions for future research.

Based on these instructions, please produce a detailed and well-structured main discussion section for the literature review on {topic} that thoroughly reflects current research debates and insights.
    """

    messages = [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": prompt},
    ]

    response = await call_llm(session, messages)
    return response


async def report_disscussion(
    session: aiohttp.ClientSession,
    topic: str,
    papers: List[Dict[str, Any]],
    pairs: List[Tuple[int, int]],
) -> str:
    sys_prompt = f"""You are an academic writing expert with extensive experience in critically synthesizing research literature. 
    """
    prompt = f"""You have already produced a comprehensive body section for a literature review on {topic}. Your next task is to generate the discussion section. This section should integrate and critically assess the key findings presented in the main body, exploring their broader implications, limitations, and potential future directions.
Curated knowledge base includes:

{help_show_summary(papers)}
Besides the above content, the following paper pairs have similar points, I will show the original text to you, please pay special attention to them:

{help_show_pair(papers, pairs)}
    
Requirements:

1. Synthesis of Key Findings:
   - Summarize the main insights and overarching themes from the body section.
   - Highlight how these findings interrelate and address the core issues of {topic}.

2. Critical Evaluation:
   - Analyze the strengths and weaknesses of the reviewed studies.
   - Identify any controversies or conflicting viewpoints present in the literature.
   - Discuss the limitations of current methodologies and research approaches.

3. Implications and Future Directions:
   - Reflect on the implications of the findings for the field of {topic}.
   - Identify and discuss existing research gaps or unanswered questions.
   - Propose potential avenues for future research that could address these gaps or further refine the current understanding.

4. Academic Tone and Style:
   - Use formal, precise, and scholarly language.
   - Ensure that the discussion is coherent, logically structured, and flows smoothly between ideas.
   - Provide evidence-based arguments and, where applicable, reference key studies (e.g., [Author, Year]).

Based on these guidelines, please craft a detailed, well-organized, and critically reflective discussion section that not only summarizes and critiques the current state of the literature on {topic} but also offers clear recommendations for future research.
    """

    messages = [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": prompt},
    ]

    response = await call_llm(session, messages)
    return response


async def report_conclusion(
    session: aiohttp.ClientSession,
    topic: str,
    papers: List[Dict[str, Any]],
    pairs: List[Tuple[int, int]],
) -> str:
    sys_prompt = f"""You are an academic writing expert with extensive experience in synthesizing and summarizing research literature.
    """
    prompt = f"""You have already developed the main body and discussion sections of a literature review on {topic}. Your next task is to produce a robust conclusion section that effectively integrates and summarizes the review.
Curated knowledge base includes:

{help_show_summary(papers)}
Besides the above content, the following paper pairs have similar points, I will show the original text to you, please pay special attention to them:

{help_show_pair(papers, pairs)}
Requirements:

1. Summary of Key Points:
   - Concisely recapitulate the central themes, significant findings, and critical insights presented in the main body and discussion sections.
   - Emphasize how the reviewed literature collectively advances the understanding of {topic}.

2. Reflection on Implications:
   - Discuss the broader implications of the key findings for the field.
   - Briefly mention any acknowledged limitations and how they influence the overall interpretations.
   - Showcase the overall contribution of the reviewed literature to the development of {topic}.

3. Recommendations for Future Research:
   - Provide clear, actionable recommendations or future research directions.
   - Highlight unresolved issues, gaps in the current literature, and areas that warrant further exploration.

4. Academic Tone and Structure:
   - Use formal, concise, and scholarly language.
   - Ensure that the conclusion is well-organized, logically coherent, and does not introduce new evidence beyond the summary of the review.
   - Conclude with a strong final statement that reinforces the significance of the literature review.

Based on the above criteria, please produce a comprehensive conclusion section for the literature review on {topic} that clearly encapsulates the essence of the review and provides insightful directions for future work.
    """

    messages = [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": prompt},
    ]

    response = await call_llm(session, messages)
    return response


async def report_title(
    session: aiohttp.ClientSession,
    topic: str,
    papers: List[Dict[str, Any]],
    pairs: List[Tuple[int, int]],
) -> str:
    sys_prompt = f"""You are an academic writing expert who has thoroughly authored a comprehensive literature review on {topic}. Having completed the entire document and deeply understood its content,
    """
    prompt = f"""
Curated knowledge base includes:

{help_show_summary(papers)}
Besides the above content, the following paper pairs have similar points, I will show the original text to you, please pay special attention to them:

{help_show_pair(papers, pairs)}

1. Concise and Impactful:
   - Create a title that is succinct yet powerful, capturing the essence of the review.
   - Ensure it is clear and informative, providing readers with an immediate sense of the main themes and contributions.

2. Reflective of Core Themes:
   - Incorporate key insights, findings, and the overall focus of the literature review.
   - Highlight the significant aspects or trends discussed in the paper without being overly technical.

3. Engaging to the Target Audience:
   - The title should intrigue potential readers while remaining academically appropriate.
   - Consider the balance between precision and creativity.

Based on these guidelines, please generate a single title (one sentence) that best encapsulates your work on [TOPIC].
    """

    messages = [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": prompt},
    ]

    response = await call_llm(session, messages)
    return response


# =========================
# Main Asynchronous Routine
# =========================


async def async_main():
    async with aiohttp.ClientSession() as session:
        papers = await paper_main("machine learning", 20)
        papers = pdf_process_main(papers)
        await summary(session, papers)
        embedding(papers)
        abstract_pairs = get_top_pairs(papers, 3, "abstract")
        introduction_pairs = get_top_pairs(papers, 3, "introduction")
        body_pairs = get_top_pairs(papers, 3, "body")
        discussion_pairs = get_top_pairs(papers, 3, "discussion")
        conclusion_pairs = get_top_pairs(papers, 3, "conclusion")
        title_pairs = get_top_pairs(papers, 3, "title")
        res = []


def main():
    asyncio.run(async_main())


if __name__ == "__main__":
    main()
