"""Search-R1 continuation protocol: prompts, document formatting, trajectory representation.

A trajectory is ONE assistant message that the policy writes in chunks. After every chunk that ends in
`</search>`, the environment appends `\n\n<information>...</information>\n\n` and generation continues.
Policy tokens are the generated chunks; prompt and information blocks are masked out of the loss.
"""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field

_SEARCH_RE = re.compile(r"<search>(.*?)</search>", re.DOTALL)
_ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)

# Official Search-R1 inference prompt (verbatim, including the original "as your want" typo).
OFFICIAL_PROMPT = (
    "Answer the given question. "
    "You must conduct reasoning inside <think> and </think> first every time you get new information. "
    "After reasoning, if you find you lack some knowledge, you can call a search engine by <search> query </search> "
    "and it will return the top searched results between <information> and </information>. "
    "You can search as many times as your want. "
    "If you find no further external knowledge needed, you can directly provide the answer inside <answer> and </answer>, "
    "without detailed illustrations. For example, <answer> Beijing </answer>. Question: {question}\n"
)

# Neutral prompt shared by every compared method: the official prompt plus an explicit abstention option.
NEUTRAL_PROMPT = (
    "Answer the given question. "
    "You must conduct reasoning inside <think> and </think> first every time you get new information. "
    "After reasoning, if you find you lack some knowledge, you can call a search engine by <search> query </search> "
    "and it will return the top searched results between <information> and </information>. "
    "You can search as many times as your want. "
    "If you find no further external knowledge needed, you can directly provide the answer inside <answer> and </answer>, "
    "without detailed illustrations. For example, <answer> Beijing </answer>. "
    "If the retrieved information does not determine an answer, or the question rests on a false premise, "
    "answer <answer> ABSTAIN </answer>. Question: {question}\n"
)

# Prompt-only control: the neutral prompt plus an explicit warning about repeated copies.
DIAGNOSTIC_PROMPT = (
    "Answer the given question. "
    "You must conduct reasoning inside <think> and </think> first every time you get new information. "
    "After reasoning, if you find you lack some knowledge, you can call a search engine by <search> query </search> "
    "and it will return the top searched results between <information> and </information>. "
    "You can search as many times as your want. "
    "Retrieved passages may repeat the same claim copied from a single source: repeated copies are not independent "
    "evidence. Prefer passages that directly state the underlying facts, and check claims against them before answering. "
    "If you find no further external knowledge needed, you can directly provide the answer inside <answer> and </answer>, "
    "without detailed illustrations. For example, <answer> Beijing </answer>. "
    "If the retrieved information does not determine an answer, or the question rests on a false premise, "
    "answer <answer> ABSTAIN </answer>. Question: {question}\n"
)

# Forced verification prompt: instructs active multi-hop entity checking across multiple queries.
VERIFY_PROMPT = (
    "Answer the given question. "
    "You must conduct reasoning inside <think> and </think> first every time you get new information. "
    "After reasoning, if you find you lack some knowledge, you can call a search engine by <search> query </search> "
    "and it will return the top searched results between <information> and </information>. "
    "Retrieved passages may repeat unverified bare claims: do not stop at the first passage stating an answer. "
    "Issue multiple distinct searches targeting the underlying bridge entities and primary Wikipedia articles to verify claims before answering. "
    "If you find no further external knowledge needed after verification, directly provide the answer inside <answer> and </answer>, "
    "without detailed illustrations. For example, <answer> Beijing </answer>. "
    "If the retrieved information does not determine an answer, or the question rests on a false premise, "
    "answer <answer> ABSTAIN </answer>. Question: {question}\n"
)

PROMPTS = {"official": OFFICIAL_PROMPT, "neutral": NEUTRAL_PROMPT, "diagnostic": DIAGNOSTIC_PROMPT, "verify": VERIFY_PROMPT}


@dataclass
class AgentProtocol:
    """Multi-turn search continuation protocol (<think> / <search> / <information> / <answer>)."""
    name: str
    stops: tuple[str, ...]
    search_end: str
    info_open: str
    info_close: str
    doc_style: str = "searchr1"
    keep_special: bool = False

    def prompt(self, question: str, tokenizer, prompt_style: str = "neutral", hint: str | None = None) -> str:
        return build_prompt_text(question, prompt_style, tokenizer, hint=hint)

    def query(self, chunk: str) -> str | None:
        return last_search_query(chunk)

    def answer(self, chunk: str) -> str | None:
        return last_answer(chunk)

    def information(self, docs: list[dict], search_no: int = 1) -> str:
        if self.doc_style == "searchr1":
            body = format_information(docs).strip()
        elif self.doc_style == "plain":
            body = "\n".join(f"Doc {i + 1}: {' '.join(d['contents'].split())}" for i, d in enumerate(docs))
        else:
            body = "".join(f"({i + 1}){' '.join(d['contents'].split())}\n" for i, d in enumerate(docs))
        return f"{self.info_open}{body}{self.info_close}"


AGENT_PROTOCOLS = {
    "searchr1": AgentProtocol("searchr1", ("</search>", "</answer>"), "</search>", "\n\n<information>", "</information>\n\n"),
}


def make_contents(title: str, text: str) -> str:
    """wiki-18 / Search-R1 corpus format: quoted title on the first line, body on the following lines."""
    return f'"{title.strip()}"\n{" ".join(text.split())}'


def split_contents(contents: str) -> tuple[str, str]:
    lines = contents.split("\n")
    return lines[0], "\n".join(lines[1:])


def format_information(docs: list[dict]) -> str:
    """Search-R1 `_passages2string` format (title keeps its surrounding quotes, as in the official code)."""
    out = ""
    for i, d in enumerate(docs):
        title, text = split_contents(d["contents"])
        out += f"Doc {i + 1}(Title: {title}) {text}\n"
    return out


def build_prompt_text(question: str, prompt_style: str, tokenizer, hint: str | None = None) -> str:
    """hint: privileged guidance appended to the user turn (trajectory synthesis only; never shown at evaluation)."""
    q = question.strip()
    if q and q[-1] != "?":
        q += "?"
    user = PROMPTS[prompt_style].format(question=q)
    if hint:
        user = user.rstrip("\n") + "\n" + hint.strip() + "\n"
    if getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": user}], add_generation_prompt=True, tokenize=False
        )
    return user


def last_search_query(chunk: str) -> str | None:
    m = _SEARCH_RE.findall(chunk)
    return m[-1].strip() if m else None


def last_answer(chunk: str) -> str | None:
    m = _ANSWER_RE.findall(chunk)
    return m[-1].strip() if m else None


@dataclass
class Trajectory:
    instance_id: str
    prompt_text: str
    segments: list[tuple[str, bool]] = field(default_factory=list)  # (text, is_policy)
    queries: list[str] = field(default_factory=list)
    retrieved_ids: list[list[str]] = field(default_factory=list)
    retrieved_docs: list[str] = field(default_factory=list)  # contents of every passage shown, in order
    answer: str | None = None
    finish: str = "running"  # answer | max_turns | length | no_action | error
    gen_tokens: int = 0
    prompt_tokens: int = 0

    @property
    def n_searches(self) -> int:
        return len(self.queries)

    def full_text(self) -> str:
        return self.prompt_text + "".join(t for t, _ in self.segments)

    def to_record(self, keep_text: bool = False) -> dict:
        d = asdict(self)
        if not keep_text:
            d.pop("prompt_text")
            d.pop("retrieved_docs")
            d["segments"] = [[t if p else f"<info:{len(t)} chars>", p] for t, p in self.segments]
        d["n_searches"] = self.n_searches
        return d
