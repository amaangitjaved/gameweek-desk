# Video walkthrough script — 4 min 40 s

Record at 1080p. Screen only, face cam optional in a corner. Don't read this
verbatim; the timings assume a normal speaking pace with a couple of pauses.

**Before you hit record:** open the app, set Verification scope to **Full squad**,
Free transfers to **2**, and click Run once so the page is warm. Then reload for a
clean start. Clear the audit log from page 2.

---

## 0:00 – 0:35 · The problem

> "This is Gameweek Desk. It's an internal tool for a six-person FPL advice service
> with forty thousand subscribers.
>
> Every gameweek that team has to check injury, suspension and rotation news across
> about six hundred players before the deadline, then decide what to publish. Done by
> hand that's roughly twenty-five hours a week of repetitive lookup — most of one
> person's time — and it's the thing that caps the product. You can't cover more
> players or more segments without hiring.
>
> So the split I've built is: the AI does the research and drafts the recommendation.
> A human decides what actually reaches a subscriber. Nothing gets published without a
> named editor pressing the button."

## 0:35 – 1:05 · One honest thing up front

*(point at the pre-season warning banner and the Prior weight metric)*

> "One thing to flag immediately. It's August — the season hasn't started, so every
> player has zero points and zero minutes.
>
> My first version of this trained a model on season-to-date points. In pre-season
> that target is constant, so the model predicted a constant, every transfer scored a
> gain of zero, and the system returned an empty list. It didn't error. It just quietly
> did nothing, which is the worst kind of failure.
>
> Now the system measures how mature the season is and tells you how much of the
> forecast is last season's data versus this season's form. Right now that's a hundred
> percent prior. That's shown in the header, not buried — because a forecast built
> entirely on priors is a different thing from one built on evidence, and the person
> reading it should know which they're getting."

## 1:05 – 1:55 · The run

*(click Run gameweek analysis)*

> "Three stages. The model ranks the pool and proposes the best legal transfer bundle
> — that's a real constraint problem: one shared budget across both transfers,
> max three players per club, correct FPL sell-price rule, and the minus-four hit
> costed properly.
>
> Then the agent researches everyone affected. Both sides — who's coming in and who's
> going out. My first version only checked incoming players, which misses the case
> where the player you're *selling* just got ruled out.
>
> And then — this is the part I care about — look at stage three. The model wanted
> Dalot. The agent found a club source saying he's out for the opening fixtures:
> unanimous across samples, cited, published yesterday. So the system didn't staple a
> warning onto the recommendation and leave a human to spot the contradiction. It
> removed Dalot from the pool, re-ran the optimiser, and verified the replacement too."

## 1:55 – 2:55 · Where the AI stops

*(scroll to "Needs your judgement", open Garner)*

> "But that's the easy case, where the AI is confident and right. This section is the
> interesting one — three players where the agent reached an answer and refused to
> stand behind it.
>
> Garner is the best example, and this is a real failure from my first build. Same
> player, same input, sixty seconds apart, the agent said 'suspended' once and
> 'injured' the next time. One of those runs was reading a match report about a
> reckless challenge and inferring a ban from it.
>
> So now the agent is sampled three times at different temperatures, and *disagreement
> is itself the signal*. Look — suspended, injured, doubtful. Thirty-three percent
> agreement. Rather than pick one and present it confidently, it escalates and says
> exactly why.
>
> The other two are different failure modes. Martinez — the agent claims a suspension
> with no source at all, so it's inadmissible. Onana — genuine selection uncertainty,
> but confidence below threshold.
>
> Three escalations, three different reasons, all named. That's what I mean by building
> trust: the human doesn't have to guess when to pay attention."

## 2:55 – 3:35 · The gate

*(scroll to Draft recommendation, expand an availability panel, edit the draft, publish)*

> "Here's the draft note for subscribers, with the evidence behind it — the quote, the
> source, the publication date, and the model's own error bar. That band matters: if
> the xP gain is smaller than the uncertainty, there's no real edge, and the editor can
> see that rather than trusting a point estimate.
>
> I'll edit this line... and publish.
>
> *(go to Audit Log)*
>
> Every decision is logged before anything goes out — who approved it, what the AI
> drafted, what actually went out, and whether the human changed it.
>
> That override rate is the metric I'd actually watch. If it's near zero, the editor
> has stopped reading and the review step is theatre. If it's very high, the AI isn't
> good enough to be drafting. It tells you whether the boundary is in the right place."

## 3:35 – 4:20 · Does it work, what does it cost

*(go to Evaluation)*

> "Sixteen hand-labelled cases. The number I care about isn't accuracy — it's *leaked*:
> wrong AND not escalated. That's the only outcome that can reach a subscriber. Two of
> sixteen.
>
> And I'd rather show you those two than hide them. Palmer — a training report mentions
> a minor knock, the agent reads 'took part in training' and says available. Unanimous,
> confident, well sourced, fresh. Every guardrail passes. It's a comprehension error,
> not a process error, and no threshold catches it.
>
> Blocking precision is fifty percent — half the time the agent says injured or
> suspended, it's over-reading. That's *why* a confirmed block is the only thing the
> system acts on alone, and everything shaky goes to a human.
>
> This curve is the real trade-off. Tighten the threshold and you leak less but you
> waste more human attention. There's no setting that minimises both, so it's an
> operator control, not a constant.
>
> On cost: about ten dollars a gameweek to check all six hundred players. The manual
> equivalent is twenty-five reviewer-hours, around eight hundred dollars. Search
> dominates the bill, which is why I sample the model three times but the search only
> once — sampling the model is nearly free, sampling search isn't."

## 4:20 – 4:40 · Close

> "Two trade-offs I'd defend if you push on them.
>
> One: I replaced a gradient-boosted model with ridge regression. Almost certainly less
> accurate at its ceiling — but the old one was fit on its own training rows and had no
> error bar, so it wasn't a forecast, it was a restatement. A simpler model I can
> validate beats a better one I can't.
>
> Two: I picked a domain I know well so the design effort could go into the human/AI
> boundary rather than into learning a new problem space. The architecture — deterministic
> scorer, agent verification against live sources, human gate before anything ships —
> is the transferable part. Point it at a trust and safety queue or a refunds desk and
> only the vocabulary changes.
>
> Assumptions and known failure modes are all written up in the app. Thanks."

---

## If they ask

**"Why not just use the FPL API's own injury flags?"**
It has a `chance_of_playing` field, but it's often stale or null and it doesn't explain
itself. The agent's job is reading the press around a player, which is where the
information appears first. In production I'd use the structured flag as a second source
before any blocking call — that's item three on my roadmap, and it's the direct fix for
the 50% blocking precision.

**"Isn't three LLM samples expensive?"**
Groq is fast and cheap, and search dominates the bill roughly 4:1. Sampling the model
three times against one set of search results costs almost nothing and buys the single
most useful signal in the system. If I sampled the search too, cost would triple for
almost no gain — the evidence would be the same.

**"What breaks first at scale?"**
Search cost and rate limits. Team news is per-club, not per-player, so the fix is caching
one search per club per day instead of one per player — roughly a 10× reduction. Second
thing is the audit log; JSONL is fine for a 6-person team, it becomes Postgres beyond that.

**"How would you know if the model degraded?"**
Two signals. Override rate on the audit log, which is free and continuous. And re-running
the labelled eval set on a schedule, growing it from editor rejections — those rejections
are labelled data the system currently throws away, which is the biggest missing piece.

**"Why is the data synthetic?"**
Reproducibility, and pre-season. I wanted the demo to show the same queue you'll see when
you click it, without depending on a third-party API being up. Live mode hits the real FPL
API and falls back to the snapshot rather than erroring. The synthetic priors do inflate
the model's R², and I say so in the app rather than letting you find it.
