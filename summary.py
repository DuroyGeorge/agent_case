from typing import List, Dict, Any
import logging
import asyncio

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)


async def call_llm(session, messages, model=DEFAULT_MODEL):
    """
    Asynchronously call the OpenRouter chat completion API with the provided messages.
    Returns the content of the assistant’s reply.
    """
    headers = {
        "Authorization": f"Bearer {LLM_BASE_URL}",
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


async def summary_section(session, title: str, section_title: str, text: str) -> str:
    sys_prompt = """
    You are an expert in summarizing academic papers.
    """
    prompt = f"""
    Title: {title}
    Section Title: {section_title}
    Text: {text}
    
    Please provide a concise summary of the section of the paper.
    Don't leave out any important details.
    """
    messages = [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": prompt},
    ]
    response = await call_llm(session, messages)
    return response


async def summary(session, papers: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    for paper in papers:
        paper["summary_text"] = {}
        paper_summary = ""
        res = [
            (
                section_title,
                summary_section(session, paper["title"], section_title, section_text),
            )
            for section_title, section_text in paper["text"].items()
        ]
        summaries = await asyncio.gather(*res)
        for summary in summaries:
            paper_summary += f"{summary}\n\n"
            paper["summary_text"][summary[0]] = summary[1]
        paper["summary_text"]["full_summary"] = paper_summary
    return papers
