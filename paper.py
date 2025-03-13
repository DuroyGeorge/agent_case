from pathlib import Path
import arxiv
import aiohttp
import asyncio
from typing import List, Dict, Any, Optional

import logging

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)


def get_paper(keyword: str, max_results: int = 2) -> List[Dict[str, Any]]:
    logger.info(
        f"Searching for papers with keyword: '{keyword}', max results: {max_results}"
    )
    try:
        search = arxiv.Search(query=keyword, max_results=max_results)
        papers = []
        for result in search.results():
            paper = {
                "title": result.title,
                "authors": [author.name for author in result.authors],
                "summary": result.summary,
                "pdf_url": result.pdf_url,
            }
            papers.append(paper)
        logger.info(f"Found {len(papers)} papers matching '{keyword}'")
        return papers
    except Exception as e:
        logger.error(
            f"Error retrieving papers for keyword '{keyword}': {str(e)}", exc_info=True
        )
        return []


async def download_paper(paper: Dict[str, Any], save_dir: Path) -> Optional[Path]:
    title = paper.get("title", "Untitled")
    logger.debug(f"Attempting to download paper: '{title}'")
    pdf_url = paper.get("pdf_url")

    if not pdf_url:
        logger.warning(f"No PDF URL found for paper: '{title}'")
        return None

    filename = save_dir / f"{title}.pdf"
    try:
        async with aiohttp.ClientSession() as session:
            logger.debug(f"Connecting to {pdf_url}")
            async with session.get(pdf_url) as resp:
                if resp.status == 200:
                    stream = await aiohttp.StreamReader.read(resp.content)
                    with open(filename, "wb") as f:
                        f.write(stream)
                    logger.info(f"Successfully downloaded paper to: {filename}")
                    return filename
                else:
                    logger.error(
                        f"Failed to download paper, status code: {resp.status}"
                    )
                    return None
    except Exception as e:
        logger.error(f"Error downloading paper '{title}': {str(e)}", exc_info=True)
        return None


async def main(keyword: str, save_dir_str: str, max_paper: int):
    logger.info(f"Starting paper search and download process for '{keyword}'")
    save_dir = Path(save_dir_str)
    if not save_dir.exists():
        logger.info(f"Creating directory: {save_dir}")
        save_dir.mkdir(parents=True, exist_ok=True)

    papers = get_paper(keyword, max_paper)
    if not papers:
        logger.warning(f"No papers found for '{keyword}', process complete")
        return

    results = [download_paper(paper, save_dir) for paper in papers]
    results = await asyncio.gather(*results)

    logger.info(f"Process complete. Downloaded {len(results)}/{len(papers)} papers")


if __name__ == "__main__":
    asyncio.run(main("ai agent", "./papers", 20))
