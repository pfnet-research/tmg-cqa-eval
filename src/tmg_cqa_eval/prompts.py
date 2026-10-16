"""Evaluation prompts used by the original TM-Gov benchmark."""

RAG_NEED_CITATION_PROMPT = """You are an expert in evaluating text quality. You will receive a user's question regarding their uploaded document (due to the length of the document, it is not shown to you), an AI assistant's response based on the document, and a sentence from the response. Your task is to determine whether this sentence is a factual sentence made based on the information in the document that requires citation, rather than an introductory sentence, transition sentence, or a summary, reasoning, or inference based on the previous response.
Ensure that you do not use any other external information during your evaluation.
Please first provide your judgment (answer with [[Yes]] or [[No]]), then provide your analysis in the format "Need Citation: [[Yes/No]] Analysis: ...".

<question>
{question}
</question>

<response>
{answer}
</response>

<sentence>
{sentence}
</sentence>
"""

RAG_SUPPORT_PROMPT = """You are an expert in evaluating text quality. You will receive a user's question about an uploaded document, a factual sentence from an AI assistant's response based on that document, and a snippet from the document (since the document is too long to display in full). Your task is to carefully assess whether this sentence is supported by the snippet. Please use the following scale to generate your rating:
- [[Fully supported]] - Most information in the sentence is supported by or extracted from the snippet. This applies only to cases where the sentence and parts of the snippet are almost identical.
- [[Partially supported]] - More than half of the content in the sentence is supported by the snippet, but a small portion is either not mentioned or contradicts the snippet. For example, if the sentence has two key points and the snippet supports only one of them, it should be considered [Partially supported].
- [[No support]] - The sentence is largely unrelated to the snippet, or most key points in the sentence do not align with the content of the snippet.
Ensure that you do not use any information or knowledge outside of the snippet when evaluating.
Please provide the rating first, followed by the analysis, in the format "Rating: [[...]] Analysis: ...".

<question>
{question}
</question>

<sentence>
{sentence}
</sentence>

<snippet>
{snippet}
</snippet>

"""

RAG_RELEVANT_PROMPT = """You are an expert in evaluating text quality. You will receive a user's question about an uploaded document, a factual sentence from an AI assistant's response based on that document, and a snippet from the document (since the document is too long to display in full). Your task is to carefully assess whether the snippet contains some key information of the sentence. Please use the following grades to generate the rating:
- [[Relevant]] - Some key points of the sentence are supported by the snippet or extracted from it.
- [[Unrelevant]] - The sentence is almostly unrelated to the snippet, or all key points are inconsistent with the snippet content.
Ensure that you do not use any information or knowledge outside of the snippet when evaluating.
Please provide the rating first, followed by the analysis, in the format "Rating: [[...]] Analysis: ...".

<question>
{question}
</question>

<sentence>
{sentence}
</sentence>

<snippet>
{snippet}
</snippet>

"""
NEED_CITATION_PROMPT = """You are an expert in evaluating text quality. You will receive a user's question that an AI assistant answered by searching a document collection (the retrieved documents are not shown to you), the assistant's response, and a sentence from the response. Your task is to determine whether this sentence is a factual sentence made based on the information in the retrieved documents that requires citation, rather than an introductory sentence, transition sentence, or a summary, reasoning, or inference based on the previous response.
Ensure that you do not use any other external information during your evaluation.
Please first provide your judgment (answer with [[Yes]] or [[No]]), then provide your analysis in the format "Need Citation: [[Yes/No]] Analysis: ...".

<question>
{question}
</question>

<response>
{answer}
</response>

<sentence>
{sentence}
</sentence>
"""

SUPPORT_PROMPT = """You are an expert in evaluating text quality. You will receive a user's question that an AI assistant answered by searching a document collection, a factual sentence from the assistant's response, and the search observations the sentence cites (search hits, an opened page, or matched passages). Your task is to carefully assess whether this sentence is supported by the observations. Please use the following scale to generate your rating:
- [[Fully supported]] - Most information in the sentence is supported by or extracted from the observations. This applies only to cases where the sentence and parts of the observations are almost identical.
- [[Partially supported]] - More than half of the content in the sentence is supported by the observations, but a small portion is either not mentioned or contradicts them. For example, if the sentence has two key points and the observations support only one of them, it should be considered [Partially supported].
- [[No support]] - The sentence is largely unrelated to the observations, or most key points in the sentence do not align with their content.
Ensure that you do not use any information or knowledge outside of the observations when evaluating.
Please provide the rating first, followed by the analysis, in the format "Rating: [[...]] Analysis: ...".

<question>
{question}
</question>

<sentence>
{sentence}
</sentence>

<observations>
{snippet}
</observations>

"""

RELEVANT_PROMPT = """You are an expert in evaluating text quality. You will receive a user's question that an AI assistant answered by searching a document collection, a factual sentence from the assistant's response, and one search observation the sentence cites (search hits, an opened page, or matched passages). Your task is to carefully assess whether the observation contains some key information of the sentence. Please use the following grades to generate the rating:
- [[Relevant]] - Some key points of the sentence are supported by the observation or extracted from it.
- [[Unrelevant]] - The sentence is almostly unrelated to the observation, or all key points are inconsistent with its content.
Ensure that you do not use any information or knowledge outside of the observation when evaluating.
Please provide the rating first, followed by the analysis, in the format "Rating: [[...]] Analysis: ...".

<question>
{question}
</question>

<sentence>
{sentence}
</sentence>

<observation>
{snippet}
</observation>

"""
JUDGE_PROMPT = """You are evaluating a candidate answer against a reference answer for a question \
about a document (the document itself is not shown).  The reference answer was written earlier \
from the same document: it is generally reliable about facts, but it may contain typos and it \
may be less complete or less detailed than the candidate.

Rate ONLY factual consistency, using exactly one of these verdicts:
- "consistent": no factual claim in the candidate contradicts the reference.  The candidate may \
be more detailed, worded differently, or include extra information that the reference does not \
mention.
- "contradictory": at least one factual claim in the candidate directly conflicts with a factual \
claim in the reference (a different value, name, date, place, or a negation).
- "insufficient": the candidate does not actually answer the question (it is off-topic, answers \
a different question, or refuses/claims the information is missing even though the reference \
answers it).

Never penalize style, length, formatting, ordering, extra detail, or typos in the reference.  \
Judge in favor of "consistent" when a difference can be explained as paraphrase, rounding, \
partial listing, or added detail.

Question:
{question}

Reference answer:
{reference}

Candidate answer:
{candidate}

Respond with a JSON object only, no other text:
{{"verdict": "consistent|contradictory|insufficient", "conflicts": ["<short description of each contradiction, empty if none>"], "reason": "<one sentence>"}}"""
