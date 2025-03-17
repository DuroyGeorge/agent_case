from typing import List, Dict, Any, Tuple
import aiohttp
import asyncio


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


# async def report_future_direction(
#     session: aiohttp.ClientSession,
#     topic: str,
#     papers: List[Dict[str, Any]],
#     pairs: List[Tuple[int, int]],
# ) -> str:
#     sys_prompt = f"""You are the chief editor of {topic} coordinating 10 domain experts to construct the future direction of a state-of-the-art review on {topic}. Integrate perspectives from {topic} scholars.
#     """
#     prompt = f"""
#     Curated knowledge base includes:

#     {help_show_summary(papers)}
#     Besides the above content, the following paper pairs have similar points, I will show the original text to you, please pay special attention to them:

#     {help_show_pair(papers, pairs)}
#     """

#     messages = [
#         {"role": "system", "content": sys_prompt},
#         {"role": "user", "content": prompt},
#     ]

#     response = await call_llm(session, messages)
#     return response


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


async def report(
    session: aiohttp.ClientSession,
    topic: str,
    papers: List[Dict[str, Any]],
    pairs: List[Tuple[int, int]],
) -> str:
    pass
