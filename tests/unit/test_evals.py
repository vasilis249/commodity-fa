"""The eval question set and the scoring rules."""

from __future__ import annotations

from datetime import date

import pytest

from fa.agents.schemas import ResearchAnswer
from fa.config import load_config
from fa.evals import Expect, Question, Ref, extract, load_questions, score, tolerance
from fa.tools import market

QS = load_questions()


def _ans(**kw) -> ResearchAnswer:
    base = {
        "plan": [],
        "answerable": True,
        "value": None,
        "unit": None,
        "choice": None,
        "answer": "",
        "citations": ["T1"],
    }
    return ResearchAnswer(**{**base, **kw})


def _q(expect: Expect) -> Question:
    return Question(
        id="x",
        category="c",
        symbol="CL=F",
        as_of=date(2026, 6, 1),
        question="?",
        expect=expect,
        ref=Ref(kind="none"),
    )


def test_question_set_is_well_formed() -> None:
    assert 30 <= len(QS) <= 45
    assert len({q.id for q in QS}) == len(QS)
    assert len({q.category for q in QS}) >= 8
    tools = set(market.build_registry().names())
    allowed = set(load_config().agents.agents["research_agent"].tools)
    for q in QS:
        assert q.as_of < date.today(), q.id
        if q.expect.type == "unanswerable":
            assert q.ref.kind == "none" and q.expect.value is None
            continue
        assert q.expect.value is not None, f"{q.id}: not frozen (run `fa eval --freeze`)"
        refs = (
            [q.ref.tool]
            if q.ref.kind == "field"
            else [q.ref.a.tool] + ([q.ref.b.tool] if q.ref.b else [])
        )
        assert set(refs) <= tools & allowed, q.id
        if q.expect.type == "choice" and q.expect.options:
            assert q.expect.value in q.expect.options, q.id
    assert sum(q.expect.type == "unanswerable" for q in QS) >= 3


def test_point_in_time_question_is_in_the_set() -> None:
    q = next(q for q in QS if q.id == "cot-04")
    assert q.expect.value == "2026-05-19"  # the 05-26 report was released after the settle


def test_number_scoring_and_percent() -> None:
    q = _q(Expect(type="number", value=0.474, percent=True))
    assert score(q, _ans(value=0.474))[0]
    assert score(q, _ans(value=47.4, unit="%"))[0]  # percent accepted for a fraction
    assert score(q, _ans(value=0.4744))[0]
    assert not score(q, _ans(value=0.48))[0]
    assert not score(q, _ans(value=None))[0]


def test_rounding_as_written_but_not_a_neighbouring_number() -> None:
    sma = _q(Expect(type="number", value=64.232))
    assert score(sma, _ans(value=64.23))[0]
    assert not score(sma, _ans(value=64.39))[0]  # the close, not the SMA
    pctile = _q(Expect(type="number", value=17.4194))
    assert score(pctile, _ans(value=17.4))[0]
    assert not score(pctile, _ans(value=17.6))[0]


def test_tolerance_rules() -> None:
    assert tolerance(Expect(type="number", value=0.01, percent=True)) == 0.0006
    assert tolerance(Expect(type="number", value=92.16)) == pytest.approx(0.09216)
    assert tolerance(Expect(type="number", value=0.5)) == 0.006
    assert tolerance(Expect(type="number", value=5.0, tol=0.1)) == 0.1


def test_choice_scoring() -> None:
    q = _q(Expect(type="choice", value="2026-05-19"))
    assert score(q, _ans(choice="2026-05-19"))[0]
    assert score(q, _ans(choice="2026/05/19"))[0]
    q2 = _q(Expect(type="choice", value="range", options=["uptrend", "downtrend", "range"]))
    assert score(q2, _ans(choice="Range"))[0]
    assert score(q2, _ans(answer="The trend is classified as range [T1]."))[0]
    assert not score(q2, _ans(answer="Not an uptrend but a range."))[0]  # ambiguous text


def test_unanswerable_scoring() -> None:
    q = _q(Expect(type="unanswerable"))
    assert score(q, _ans(answerable=False))[0]
    assert not score(q, _ans(value=80.0))[0]
    assert not score(_q(Expect(type="number", value=1.0)), _ans(answerable=False))[0]


def test_extract_paths() -> None:
    p = {"a": [{"n": "x", "v": 1}, {"n": "y", "v": 2}], "h": [{"q": 0.9, "p": 5}], "k.l": 3}
    assert extract(p, "a[n=y].v") == 2
    assert extract(p, "a[0].v") == 1
    assert extract(p, "h[q=0.9].p") == 5
    with pytest.raises(StopIteration):
        extract(p, "a[n=z].v")


@pytest.mark.parametrize(
    ("value", "unit", "ok"),
    [
        (-37.63, None, True),
        (37.63, None, False),  # sign flipped: the text checker alone would accept it
        (-13.9, "%", True),
        (13.9, "%", False),
        (-0.14, None, True),  # rounded as written
        (12.5, None, False),  # not in the result at all
    ],
)
def test_headline_value_is_sign_strict(value, unit, ok) -> None:
    from fa.orchestration.research_loop import verify

    results = {"T1": {"result_id": "T1", "price": {"last_close": -37.63, "change_20d": -0.139127}}}
    ans = _ans(value=value, unit=unit, answer="x")
    assert (not verify(ans, results)) is ok
