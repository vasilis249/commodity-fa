"""Render a Report as Markdown and HTML, and save report files."""

from __future__ import annotations

import html
import re
from pathlib import Path

from fa.reports.schema import Report


def _pct(x: float | None, digits: int = 1) -> str:
    return "n/a" if x is None else f"{x * 100:+.{digits}f}%"


def _num(x: float | str | None) -> str:
    if x is None:
        return "n/a"
    if isinstance(x, str):
        return x
    return f"{x:,.4g}"


def to_markdown(r: Report) -> str:
    d = r.decision
    out = [
        f"# {r.symbol}: {r.name}",
        f"*As of {r.as_of} settlement · generated {r.generated_at:%Y-%m-%d %H:%M} UTC"
        f" · run `{r.run_id}`*",
        "",
        f"## Summary: **{r.rating}** (conviction {r.conviction:.2f})",
        "",
        r.thesis,
        "",
        "| component | weight | signal | contribution | note |",
        "|---|---:|---:|---:|---|",
        *[
            f"| {c.name} | {c.weight:.2f} | {c.signal:+.2f} | {c.contribution:+.3f} | {c.note} |"
            for c in d.components
        ],
        f"| **score** | | | **{d.score:+.3f}** | computed in code |",
        "",
        f"## Price forecast ({r.unit})",
        "",
        "| horizon | model | P10 | P50 | P90 | P(up) |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for h in r.forecast:
        b = {x.quantile: x for x in h.bands}
        cells = " | ".join(
            f"{_num(b[q].price)} ({_pct(b[q].log_return)})" if q in b else "n/a"
            for q in (0.1, 0.5, 0.9)
        )
        pup = "n/a" if h.p_up is None else f"{h.p_up:.0%}"
        out.append(f"| {h.horizon}d | {h.model} | {cells} | {pup} |")
    out += ["", f"*Model skill ({r.forecast_eval}):*", ""]
    out += [f"- **{h.horizon}d:** {h.skill.verdict}" for h in r.forecast]
    for s in r.sections:
        out += ["", f"## {s.title}"]
        if s.view:
            out.append(f"*{s.agent}: {s.view.stance}, confidence {s.view.confidence:.2f}*")
        out += ["", s.text]
        if s.key_numbers:
            out += [
                "",
                "Key numbers (code): "
                + "; ".join(f"{k} = {_num(v)}" for k, v in s.key_numbers.items()),
            ]
        if s.view and s.view.key_points:
            out += [""] + [f"- {p}" for p in s.view.key_points]
    out += [
        "",
        "## Bull case vs bear case",
        "",
        f"**Bull.** {r.bull_summary}",
        "",
        f"**Bear.** {r.bear_summary}",
    ]
    out += ["", "## Risks and suggested max exposure", "", r.risk_summary, ""]
    out += [f"- {x}" for x in r.risks]
    out += ["", f"**Suggested max exposure: {r.max_exposure:.0%} of equity.** {r.exposure_note}"]
    v = r.validation
    out += [
        "",
        "## Data, validation and cost",
        "",
        f"- Numeric claims fixed by agents after the code check: {v.numeric_violations_fixed}; "
        f"sentences removed as unverifiable: {len(v.sentences_removed)}",
        f"- Validator issues raised: {len(v.validator_issues)}; "
        f"unresolved: {len(v.unresolved_validator_issues)}",
        *[f"  - unresolved: {x}" for x in v.unresolved_validator_issues],
        f"- Run cost: ${r.cost.usd:.3f} of ${r.cost.budget_usd:.2f} budget, "
        f"{r.cost.tokens:,} tokens, {r.cost.calls} calls",
        f"- Scratchpad (every tool call and result behind each [T#]): `{r.scratchpad}`",
        "- Data sources: " + ", ".join(r.data_sources),
        *[f"- Note: {n}" for n in r.data_notes],
        "- Models: " + "; ".join(f"{a}: {m}" for a, m in r.models.items()),
        "",
        f"*{r.disclaimer}*",
    ]
    return "\n".join(out) + "\n"


def to_html(r: Report) -> str:
    """Minimal, dependency-free HTML from the Markdown (headings, tables, lists, emphasis)."""
    lines = to_markdown(r).splitlines()
    body: list[str] = []
    in_table = in_list = False
    for line in lines:
        esc = html.escape(line)
        esc = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", esc)
        esc = re.sub(r"\*(.+?)\*", r"<em>\1</em>", esc)
        esc = re.sub(r"`(.+?)`", r"<code>\1</code>", esc)
        if line.startswith("|"):
            if re.match(r"^\|[-:| ]+\|$", line):
                continue
            cells = [c.strip() for c in esc.strip("|").split("|")]
            if not in_table:
                body.append("<table>")
                in_table = True
                body.append("<tr>" + "".join(f"<th>{c}</th>" for c in cells) + "</tr>")
            else:
                body.append("<tr>" + "".join(f"<td>{c}</td>" for c in cells) + "</tr>")
            continue
        if in_table:
            body.append("</table>")
            in_table = False
        if line.startswith("- ") or line.startswith("  - "):
            if not in_list:
                body.append("<ul>")
                in_list = True
            body.append(f"<li>{esc.lstrip(' -')}</li>")
            continue
        if in_list:
            body.append("</ul>")
            in_list = False
        if line.startswith("## "):
            body.append(f"<h2>{esc[3:]}</h2>")
        elif line.startswith("# "):
            body.append(f"<h1>{esc[2:]}</h1>")
        elif line.strip():
            body.append(f"<p>{esc}</p>")
    if in_table:
        body.append("</table>")
    if in_list:
        body.append("</ul>")
    style = (
        "body{font-family:system-ui,sans-serif;max-width:900px;margin:2rem auto;padding:0 1rem;"
        "line-height:1.5;color:#1b1b1f;background:#fff}"
        "table{border-collapse:collapse;margin:1rem 0}"
        "td,th{border:1px solid #ccd;padding:.3rem .6rem;text-align:left}th{background:#f2f3f7}"
        "code{background:#f2f3f7;padding:0 .2rem}"
        "@media (prefers-color-scheme:dark){body{background:#15161a;color:#e6e6ea}"
        "th,code{background:#23252c}td,th{border-color:#3a3d46}}"
    )
    title = html.escape(f"{r.symbol} report {r.as_of}")
    return (
        f"<!doctype html><html lang=en><head><meta charset=utf-8>"
        f"<meta name=viewport content='width=device-width,initial-scale=1'><title>{title}</title>"
        f"<style>{style}</style></head><body>{''.join(body)}</body></html>\n"
    )


def save(r: Report, reports_dir: Path) -> dict[str, Path]:
    reports_dir.mkdir(parents=True, exist_ok=True)
    base = reports_dir / f"{r.symbol.replace('=', '_')}_{r.as_of}"
    paths = {
        "json": Path(f"{base}.json"),
        "md": Path(f"{base}.md"),
        "html": Path(f"{base}.html"),
    }
    paths["json"].write_text(r.model_dump_json(indent=2), encoding="utf-8")
    paths["md"].write_text(to_markdown(r), encoding="utf-8")
    paths["html"].write_text(to_html(r), encoding="utf-8")
    return paths
