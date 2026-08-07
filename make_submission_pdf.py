"""Generates the one-page submission PDF."""

from __future__ import annotations

import sys

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    HRFlowable, ListFlowable, ListItem, PageBreak, Paragraph,
    SimpleDocTemplate, Spacer, Table, TableStyle,
)

from xml.sax.saxutils import escape as _esc

# Angle brackets in a placeholder are parsed as markup by ReportLab and the
# text vanishes, so placeholders avoid them and all URLs are escaped.
APP_URL = sys.argv[1] if len(sys.argv) > 1 else "PASTE-STREAMLIT-URL-HERE"
VIDEO_URL = sys.argv[2] if len(sys.argv) > 2 else "PASTE-VIDEO-URL-HERE"
OUT = sys.argv[3] if len(sys.argv) > 3 else "Gameweek_Desk_Submission.pdf"

INK = colors.HexColor("#14181f")
MUTED = colors.HexColor("#5b6572")
ACCENT = colors.HexColor("#0f7b57")
RULE = colors.HexColor("#d6dbe1")

ss = getSampleStyleSheet()
S = {
    "title": ParagraphStyle("title", parent=ss["Title"], fontName="Helvetica-Bold",
                            fontSize=21, leading=25, textColor=INK, alignment=TA_LEFT,
                            spaceAfter=2),
    "sub": ParagraphStyle("sub", parent=ss["Normal"], fontName="Helvetica",
                          fontSize=10.5, leading=14.5, textColor=MUTED, spaceAfter=10),
    "h": ParagraphStyle("h", parent=ss["Heading2"], fontName="Helvetica-Bold",
                        fontSize=11.5, leading=14, textColor=ACCENT,
                        spaceBefore=11, spaceAfter=5),
    "b": ParagraphStyle("b", parent=ss["Normal"], fontName="Helvetica",
                        fontSize=9.4, leading=13.2, textColor=INK, spaceAfter=5),
    "li": ParagraphStyle("li", parent=ss["Normal"], fontName="Helvetica",
                         fontSize=9.3, leading=12.8, textColor=INK),
    "cell": ParagraphStyle("cell", parent=ss["Normal"], fontName="Helvetica",
                           fontSize=8.6, leading=11.4, textColor=INK),
    "cellb": ParagraphStyle("cellb", parent=ss["Normal"], fontName="Helvetica-Bold",
                            fontSize=8.6, leading=11.4, textColor=INK),
    "mono": ParagraphStyle("mono", parent=ss["Normal"], fontName="Courier",
                           fontSize=8.4, leading=11.6, textColor=INK),
    "foot": ParagraphStyle("foot", parent=ss["Normal"], fontName="Helvetica-Oblique",
                           fontSize=8.2, leading=11, textColor=MUTED),
}


def rule(space_before: float = 3, space_after: float = 6):
    return HRFlowable(width="100%", thickness=0.7, color=RULE,
                      spaceBefore=space_before, spaceAfter=space_after)


def bullets(items: list[str]):
    return ListFlowable(
        [ListItem(Paragraph(t, S["li"]), leftIndent=11) for t in items],
        bulletType="bullet", bulletFontSize=6, bulletOffsetY=1.5,
        leftIndent=11, spaceAfter=5,
    )


def table(rows: list[list[str]], widths: list[float], header: bool = True):
    data = [[Paragraph(c, S["cellb"] if (header and i == 0) else S["cell"]) for c in row]
            for i, row in enumerate(rows)]
    t = Table(data, colWidths=widths, hAlign="LEFT")
    style = [
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ("LINEBELOW", (0, 0), (-1, -2), 0.4, RULE),
        ("BOX", (0, 0), (-1, -1), 0.6, RULE),
    ]
    if header:
        style += [("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#eef2f5"))]
    t.setStyle(TableStyle(style))
    return t


story = []
W = A4[0] - 34 * mm

# ---------------------------------------------------------------- page 1
story += [
    Paragraph("Gameweek Desk", S["title"]),
    Paragraph(
        "An availability-verification and editorial review system for a small "
        "content operations team. &nbsp;Amaan Javed", S["sub"]),
    rule(0, 8),
]

story += [
    table([
        ["Live prototype",
         f'<link href="{_esc(APP_URL)}" color="#0f7b57">{_esc(APP_URL)}</link>'],
        ["Video walkthrough",
         f'<link href="{_esc(VIDEO_URL)}" color="#0f7b57">{_esc(VIDEO_URL)}</link>'],
    ], [34 * mm, W - 34 * mm], header=False),
    Spacer(1, 4),
    Paragraph(
        "The prototype runs with no API keys configured, on recorded agent verdicts. The "
        "orchestration, aggregation and escalation logic are the same code paths used "
        "against live APIs; only the two external calls are replayed.", S["foot"]),
]

story += [Paragraph("The problem", S["h"])]
story += [Paragraph(
    "A six-person FPL advice service with 40,000 subscribers must research injury, "
    "suspension and rotation news across roughly 600 players before every gameweek "
    "deadline, then decide what to publish. Done manually that is about "
    "<b>25 reviewer-hours per week</b> of repetitive lookup — most of one person's time — "
    "and it caps the product: covering more players or more subscriber segments requires "
    "hiring.", S["b"])]
story += [Paragraph(
    "This is a stand-in for a shape of problem rather than a claim about any company. The "
    "same structure — a deterministic ranking, external research requiring judgement, and "
    "a publish decision with real consequences — describes a trust &amp; safety review "
    "queue, a refunds desk, or a grant-eligibility screen. Only the vocabulary changes.",
    S["b"])]

story += [Paragraph("Architecture", S["h"])]
story += [Paragraph(
    "FPL data &rarr; xP model (ridge, cross-validated) &rarr; transfer optimiser "
    "(shared budget, legal squads) &rarr; availability agent (search + LLM, "
    "self-consistency) &rarr; review queue &rarr; <b>human publish gate</b>", S["mono"])]
story += [Spacer(1, 3), Paragraph(
    "A <b>confirmed</b> availability problem — unanimous across samples, cited, and fresh — "
    "removes that player from the candidate pool and the optimiser re-runs. The agent does "
    "not merely annotate a recommendation it disagrees with. Anything less certain is "
    "escalated to a human with a stated reason.", S["b"])]

story += [Paragraph("The human / AI boundary", S["h"])]
story += [table([
    ["The AI owns", "The human owns"],
    ["Ingesting and cleaning data; ranking the player pool; searching live sources; "
     "reading unstructured news and forming a verdict; drafting the subscriber note; "
     "removing confirmed-unavailable players and re-running the optimiser.",
     "Every decision that reaches a subscriber; anything flagged uncertain; the "
     "confidence and freshness thresholds; whether a &minus;4 point hit is worth taking."],
], [W / 2, W / 2])]
story += [Spacer(1, 4), Paragraph(
    "The system has no write access to any external system. Its only outputs are a draft "
    "and a queue. The one action it takes alone — excluding a confirmed-unavailable player "
    "— is conservative: the worst case is a missed opportunity, not a false claim published.",
    S["b"])]

story += [Paragraph("Measured results", S["h"])]
story += [table([
    ["Metric", "Value", "What it means"],
    ["Leaked", "2 / 16", "Wrong <b>and</b> not escalated. The only outcome that can reach a "
                         "subscriber. The number that matters."],
    ["Caught", "3 / 16", "Wrong but escalated. The safety net worked."],
    ["Over-escalated", "2 / 16", "Correct but escalated anyway. Pure cost in human attention."],
    ["Blocking precision", "50%", "Half of injury/suspension flags over-read the source. "
                                  "This is why only confirmed flags act unilaterally."],
    ["Blocking recall", "100%", "No genuinely unavailable player was missed on this set."],
    ["Cost", "$10.30/GW", "600 players, 3 samples each. Manual equivalent ~$800/GW."],
], [30 * mm, 20 * mm, W - 50 * mm])]

story += [PageBreak()]

# ---------------------------------------------------------------- page 2
story += [Paragraph("Trade-offs, and which way I called them", S["h"])]
story += [bullets([
    "<b>Escalate more, or leak less.</b> Every threshold that reduces wrong answers "
    "reaching an editor increases the number of correct answers they waste time "
    "confirming. There is no setting that minimises both. Exposed as an operator control "
    "with the trade-off curve rendered in-app, because the right point differs on a "
    "Tuesday and an hour before deadline.",

    "<b>Sample the model, not the search.</b> Self-consistency needs several opinions, but "
    "search dominates cost roughly 4:1. The search runs once; the model is sampled three "
    "times at varying temperature against the same evidence. This measures instability in "
    "the judgement, which is where it was observed.",

    "<b>A simple model I can validate over a complex one I cannot.</b> The earlier version "
    "used gradient boosting fit on season-to-date points and predicted on its own training "
    "rows — an in-sample restatement presented as a forecast. Replaced with cross-validated "
    "ridge regression carrying a residual-derived prediction interval. Lower ceiling, but "
    "the output is a forecast with an error bar, which is what a human needs to weigh it.",

    "<b>Recorded snapshot as the default.</b> Reproducible for review, and immune to a "
    "third-party outage mid-demo. Live mode hits the real API and degrades to the snapshot "
    "rather than erroring.",
])]

story += [Paragraph("Judgement under uncertainty: three real failures, three responses", S["h"])]
story += [table([
    ["Observed failure", "Response"],
    ["Same player, same input, 60 seconds apart: the agent returned "
     "<i>suspended</i> once and <i>injured</i> the next time.",
     "Sampled three times at varying temperature. <b>Disagreement is itself the signal</b> — "
     "when samples do not converge the item escalates rather than one verdict being "
     "presented confidently."],
    ["A verdict of &ldquo;found guilty of violent conduct, therefore suspended&rdquo; with "
     "no source and no date — a match report read as current squad status.",
     "Every verdict must carry a source URL, publication date and verbatim quote. Uncited "
     "blocking claims are inadmissible and escalate."],
    ["News from weeks earlier treated as establishing availability today.",
     "Freshness window: evidence beyond it cannot on its own justify a blocking flag."],
], [W * 0.46, W * 0.54])]

story += [Paragraph("Known limits", S["h"])]
story += [bullets([
    "<b>Comprehension errors pass every guardrail.</b> Two eval cases are unanimous, cited, "
    "fresh and wrong — a training report mentioning a minor knock read as full fitness, and "
    "transfer speculation read as rotation risk. Disagreement, citation and freshness checks "
    "cannot catch a misreading. Roughly 1 in 8. This is the residual an editor absorbs, and "
    "the reason the review step exists.",
    "<b>Search coverage is uneven.</b> Well-covered players yield good evidence; a squad "
    "player at a promoted club may return nothing usable. The system returns UNKNOWN and "
    "escalates rather than guessing — but that loads the human exactly where they also know least.",
    "<b>Pre-season forecasts are priors, not predictions.</b> With no minutes played the model "
    "leans entirely on last season's rates. The prior weight is displayed in the header rather "
    "than buried.",
    "<b>Synthetic snapshot inflates model metrics.</b> Priors were generated from price, so "
    "cross-validated R&#178; is optimistic. Stated in-app rather than left to be discovered.",
    "<b>No provider redundancy.</b> One search provider, one model provider. Either failing "
    "degrades the system to &ldquo;everything escalates&rdquo; — safe, but not useful.",
])]

story += [Paragraph("What I would build next, in order", S["h"])]
story += [bullets([
    "<b>Close the feedback loop on overrides.</b> Editor rejections with reasons are labelled "
    "data currently being discarded. This turns every week of operation into eval set growth "
    "— the highest-value missing piece.",
    "<b>Cache verdicts per club, not per player.</b> Team news is club-level. One search per "
    "club rather than per player cuts the dominant cost by roughly an order of magnitude.",
    "<b>A second source before any blocking call.</b> Blocking precision is the weakest "
    "measured number; cross-checking against a structured injury feed is the direct fix.",
    "<b>Batch overnight, review in the morning.</b> Latency stops mattering, and a cheaper, "
    "slower model becomes viable.",
])]

story += [rule(8, 5)]
story += [Paragraph(
    "Assumptions are stated explicitly on the Assumptions page in the app. The scenario "
    "(team size, subscriber count, manual baseline) is an assumption; the measured results, "
    "cost figures and failure modes are computed from the running system. Player club and "
    "position data in the snapshot is illustrative and may not reflect current transfers.",
    S["foot"])]

SimpleDocTemplate(
    OUT, pagesize=A4,
    leftMargin=17 * mm, rightMargin=17 * mm,
    topMargin=15 * mm, bottomMargin=14 * mm,
    title="Gameweek Desk — Prototype Submission", author="Amaan Javed",
).build(story)

print(f"wrote {OUT}")
