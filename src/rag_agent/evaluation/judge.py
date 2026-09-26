"""LLM judge (ТЗ S9, [37]): correctness against the reference answer, faithfulness
to the cited fragments (RAGAS-style), supported citations and relevance.

The judge sees one answer at a time, so position bias of pairwise comparison
does not apply; length and style are explicitly excluded from the criteria. Its
agreement with human labels must be checked on a manual sample before the
scores are trusted (calibration, ТЗ S9).
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from rag_agent.llm import BaseLLM, LLMError

JUDGE_PROMPT = """Ты — строгий и беспристрастный эксперт. Ты оцениваешь ответ ассистента, который отвечает на вопросы по личному архиву исследовательских файлов пользователя или на общие вопросы.

Тебе даны: вопрос, эталонный ответ, ответ ассистента и фрагменты файлов, на которые ассистент сослался номерами [n].

Оцени:
- correctness:
  "correct" — ответ по существу совпадает с эталоном: ключевые числа, имена, файлы совпадают; формулировка и лишние подробности не важны;
  "partial" — верна только часть ответа или пропущено существенное из эталона;
  "incorrect" — ответ неверный, выдуманный или ответа нет.
  Если эталон говорит, что в архиве нет ответа, правильно — честно сообщить, что информации нет; любой уверенный фактический ответ тогда incorrect.
- faithful: true, если каждое утверждение ответа о файлах, коде и результатах пользователя подтверждается приведёнными фрагментами. Общие пояснения и отказ не нарушают faithful. Если фрагментов нет, а ответ утверждает что-то о файлах пользователя, — false.
- supported_citations: номера ссылок [n], чьи фрагменты действительно подтверждают утверждение, к которому ссылка приписана.
- relevant: true, если ответ отвечает именно на заданный вопрос.
- rationale: одно короткое предложение с обоснованием.

Не учитывай длину, стиль и язык ответа. Верни только JSON."""

JUDGE_SCHEMA = {
    "type": "object",
    "properties": {
        "correctness": {"type": "string", "enum": ["correct", "partial", "incorrect"]},
        "faithful": {"type": "boolean"},
        "supported_citations": {"type": "array", "items": {"type": "integer"}},
        "relevant": {"type": "boolean"},
        "rationale": {"type": "string"},
    },
    "required": ["correctness", "faithful", "supported_citations", "relevant", "rationale"],
    "additionalProperties": False,
}

SCORES = {"correct": 1.0, "partial": 0.5, "incorrect": 0.0}
NO_ANSWER_REFERENCE = "В архиве пользователя нет ответа на этот вопрос; ожидается корректный отказ."


class Verdict(BaseModel):
    correctness: str
    score: float
    faithful: bool
    supported_citations: list[int] = Field(default_factory=list)
    relevant: bool
    rationale: str = ""
    error: str | None = None


def judge_answer(
    llm: BaseLLM,
    question: str,
    reference: str | None,
    answer: str,
    general: str,
    cited_sources: list[tuple[int, str, str]],  # (n, label, text)
    max_source_chars: int = 8000,  # at least what the generator saw: a shorter cut marks true answers as made up
) -> Verdict:
    sources = "\n\n".join(
        f'<source id="{n}" ref="{label}">\n{text[:max_source_chars]}\n</source>' for n, label, text in cited_sources
    ) or "(ассистент не сослался ни на один фрагмент)"
    reply = answer + (f"\n\n[Дополнение из общих знаний]\n{general}" if general else "")
    user = (
        f"Вопрос:\n{question}\n\nЭталонный ответ:\n{reference or NO_ANSWER_REFERENCE}\n\n"
        f"Ответ ассистента:\n{reply}\n\nФрагменты, на которые сослался ассистент:\n{sources}"
    )
    try:
        resp = llm.chat(
            [{"role": "system", "content": JUDGE_PROMPT}, {"role": "user", "content": user}],
            json_schema=JUDGE_SCHEMA,
            temperature=0.0,
            max_tokens=600,
            purpose="judge",
        )
        data = resp.json()
        correctness = data.get("correctness", "incorrect")
        valid = {n for n, _, _ in cited_sources}
        return Verdict(
            correctness=correctness,
            score=SCORES.get(correctness, 0.0),
            faithful=bool(data.get("faithful")),
            supported_citations=sorted({int(n) for n in data.get("supported_citations", []) if int(n) in valid}),
            relevant=bool(data.get("relevant")),
            rationale=str(data.get("rationale", ""))[:500],
        )
    except (LLMError, ValueError, TypeError) as exc:
        return Verdict(correctness="error", score=0.0, faithful=False, relevant=False, error=str(exc)[:300])
