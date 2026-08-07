"""End-to-end checks. Run with: python test_smoke.py

Asserts the invariants that the original prototype violated, so a regression
on any of them fails loudly rather than producing a plausible-looking but
unexecutable recommendation.
"""

from __future__ import annotations

import sys

from core import store
from core.data import load, resolve_players
from core.evals import cost_projection, run as run_evals, threshold_sweep
from core.model import score_players
from core.optimizer import demo_squad, optimise, sell_price
from core.pipeline import run_gameweek
from core.verify import BLOCKING

FAILURES: list[str] = []


def check(label: str, condition: bool, detail: str = "") -> None:
    if condition:
        print(f"  PASS  {label}")
    else:
        print(f"  FAIL  {label}" + (f" — {detail}" if detail else ""))
        FAILURES.append(label)


print("\n[1] FPL sell-price rule (purchase price + half the rise, rounded down)")
for purchase, current, expected in [
    (5.0, 5.6, 5.3), (5.0, 5.5, 5.2), (6.0, 5.4, 5.4),
    (4.5, 5.0, 4.7), (7.0, 7.1, 7.0), (4.0, 4.0, 4.0),
]:
    got = sell_price(purchase, current)
    check(f"sell({purchase}, {current}) == {expected}", abs(got - expected) < 1e-9, f"got {got}")


print("\n[2] Data layer and cold start")
res = load("snapshot", horizon=4)
check("snapshot loads", len(res.players) > 100, f"{len(res.players)} players")
check("pre-season detected", res.prior_weight > 0.9, f"prior_weight={res.prior_weight}")
check("cold start is warned about, not silent", any("Pre-season" in w for w in res.warnings))

r = resolve_players(res.players, ["Henderson", "Salah", "Dean Henderson", "Nobody At All"])
check("ambiguous surname is surfaced, not guessed",
      any(a["input"] == "Henderson" for a in r.ambiguous))
check("full name disambiguates", any(m["full_name"] == "Dean Henderson" for m in r.matched))
check("unknown name reported", "Nobody At All" in r.missing)


print("\n[3] Model produces a usable signal in pre-season")
scored, report = score_players(res.players, res.prior_weight, 4)
check("xP is non-zero pre-season (the bug that broke the original)",
      float(scored["xp_horizon"].max()) > 1.0, f"max={scored['xp_horizon'].max():.2f}")
check("xP varies across players", scored["xp_horizon"].std() > 0.5)
check("validation is out-of-fold and reported", 0.0 < report.cv_r2 <= 1.0, f"R2={report.cv_r2}")
check("prediction interval is present and ordered",
      bool((scored["xp_low"] <= scored["xp_horizon"]).all()
           and (scored["xp_horizon"] <= scored["xp_high"]).all()))


print("\n[4] Optimiser produces legal, executable bundles")
squad = demo_squad(scored)
bank = round(100.0 - float(squad["price"].sum()), 1)
check("squad has 15 players", len(squad) == 15, f"{len(squad)}")
check("squad composition is legal",
      squad["position"].value_counts().to_dict() == {"DEF": 5, "MID": 5, "FWD": 3, "GK": 2})
check("no more than 3 per club", int(squad["team_name"].value_counts().max()) <= 3)
check("squad is affordable", float(squad["price"].sum()) <= 100.0)

bundles = optimise(squad, scored, bank=bank, free_transfers=2, max_transfers=3, top_k=8)
check("bundles were found", len(bundles) > 0)
for i, b in enumerate(bundles):
    check(f"bundle {i}: bank never goes negative", b.bank_after >= -1e-6, f"{b.bank_after}")
    check(f"bundle {i}: no duplicate incoming player",
          len({t.in_id for t in b.transfers}) == len(b.transfers))
    check(f"bundle {i}: no duplicate outgoing player",
          len({t.out_id for t in b.transfers}) == len(b.transfers))
    check(f"bundle {i}: respects 3-per-club", max(b.club_counts.values()) <= 3)
    check(f"bundle {i}: like-for-like positions",
          all(t.position for t in b.transfers))
    check(f"bundle {i}: hit accounted correctly",
          abs(b.net_xp_gain - (b.gross_xp_gain - b.hit_points)) < 0.02)
    check(f"bundle {i}: incoming not already in squad",
          not ({t.in_id for t in b.transfers} & set(squad["id"].astype(int))))

# The specific bug: two transfers must not each spend the whole bank.
multi = [b for b in bundles if b.n >= 2]
if multi:
    b = multi[0]
    spend = sum(t.in_cost for t in b.transfers)
    proceeds = sum(t.out_sell for t in b.transfers)
    check("multi-transfer bundle shares one budget",
          abs((bank + proceeds - spend) - b.bank_after) < 0.05,
          f"expected {bank + proceeds - spend:.2f}, reported {b.bank_after}")


print("\n[5] Verification agent and escalation policy")
for scope in ("bundle", "squad"):
    gw = run_gameweek(squad, scored, bank, free_transfers=2, max_transfers=2, scope=scope)
    check(f"{scope}: verdicts produced", len(gw.verdicts) > 0)
    check(f"{scope}: both sides of a transfer are checked",
          {v.side for v in gw.verdicts} & {"IN", "OUT", "HELD"} != set())
    check(f"{scope}: a confirmed injury blocks and triggers a rerun",
          "Dalot" in gw.blocked_players and gw.revised)
    check(f"{scope}: the replacement is not the blocked player",
          gw.final is None or all(t.in_name.split(" (")[0] not in gw.blocked_players
                                  for t in gw.final.transfers))
    check(f"{scope}: escalated verdicts never act unilaterally",
          all(not v.is_blocking for v in gw.verdicts if v.escalate))
    check(f"{scope}: every escalation names a reason",
          all(v.escalation_reasons for v in gw.verdicts if v.escalate))
    for v in gw.verdicts:
        if v.status in BLOCKING and not v.escalate:
            check(f"{scope}: blocking verdict for {v.player} carries a source",
                  bool(v.source_url))

gw_squad = run_gameweek(squad, scored, bank, free_transfers=2, max_transfers=2, scope="squad")
check("unstable verdict escalates rather than picking one (Garner)",
      "Garner" in gw_squad.escalated_players)
check("full-squad scope checks more players than bundle scope",
      len(gw_squad.verdicts) > len(run_gameweek(
          squad, scored, bank, free_transfers=2, max_transfers=2, scope="bundle").verdicts))


print("\n[6] Evaluation harness")
ev = run_evals()
check("eval set loaded", ev.n >= 12, f"n={ev.n}")
check("harness reports leaks rather than a perfect score", ev.leaked > 0)
check("harness catches some errors", ev.caught > 0)
check("outcomes partition the set",
      ev.caught + ev.leaked + ev.over_escalated + ev.clean_pass == ev.n)
sweep = threshold_sweep()
check("threshold sweep shows a real trade-off",
      sweep[0]["over_escalated"] != sweep[-1]["over_escalated"])
proj = cost_projection(600)
check("cost projection is positive and bounded", 0 < proj["usd_per_run"] < 1000)
check("automation is cheaper than the manual baseline",
      proj["usd_per_run"] < proj["manual_usd_per_run"])


print("\n[7] Audit log")
before = len(store.read_all())
store.record("PUBLISHED", "smoke-test", "A → B", {"x": 1},
             ai_recommendation="draft", human_changed_it=False)
store.record("REJECTED", "smoke-test", "C → D", {"x": 2},
             reason="Disagreed with the availability call", human_changed_it=True)
after = store.read_all()
check("entries are appended", len(after) == before + 2)
check("entries are fingerprinted", all(e.get("fingerprint") for e in after[:2]))
check("newest entry is first", after[0]["summary"] == "C → D")
s = store.stats()
check("override rate is computed", 0.0 <= s["override_rate"] <= 1.0)


print("\n" + "=" * 62)
if FAILURES:
    print(f"{len(FAILURES)} CHECK(S) FAILED:")
    for f in FAILURES:
        print(f"  - {f}")
    sys.exit(1)
print("ALL CHECKS PASSED")
