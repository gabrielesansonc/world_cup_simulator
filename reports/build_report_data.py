"""
build_report_data.py — Generate the data backing the World Cup 2026 consensus report.

Runs N stochastic simulations of the tournament using the consensus (avg) match
probabilities and aggregates, per team:
  - probability of reaching each stage (group -> KO -> R16 -> QF -> SF -> Final -> Champion)
  - championship / podium / group-winner counts
  - the opponent that eliminated them most often
  - their single most common full tournament path

Also computes, from the 5 individual model CSVs:
  - each model's favorites / least-favorites vs the consensus (centered delta)
  - a confidence / decisiveness ranking (who gives the most unbalanced probabilities)
  - a deterministic "chalk" bracket (favorite wins every match)
  - assorted interesting findings

Output: reports/report_data.json
"""

import json
import os
import random
import sys
from collections import Counter, defaultdict

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
sys.path.insert(0, os.path.join(ROOT, "src"))

from wc_2026_simulator_function import (  # noqa: E402
    GROUPS, R32_BRACKET, load_probabilities,
    _get_probs, _assign_third_place, _sim_group_match, _rand_scoreline,
)

DATA_DIR = os.path.join(ROOT, "data")
N_SIMS   = int(os.environ.get("N_SIMS", "10000"))
SEED     = 20260607

MODELS = ["gemini_pro_31", "gpt_52", "grok_42", "opus_46", "sonnet_46"]
MODEL_LABELS = {
    "gemini_pro_31": "Gemini Pro 3.1",
    "gpt_52": "GPT-5",
    "grok_42": "Grok 4.2",
    "opus_46": "Claude Opus 4.6",
    "sonnet_46": "Claude Sonnet 4.6",
}
ALL_TEAMS = sorted({t for g in GROUPS.values() for t in g})
TEAM_GROUP = {t: g for g, teams in GROUPS.items() for t in teams}

# Stage ladder (monotonic): reaching stage k implies reaching all earlier.
STAGES = ["ko", "r16", "qf", "sf", "final", "champion"]
STAGE_LABEL = {
    "ko": "Knockouts (R32)", "r16": "Round of 16", "qf": "Quarter-final",
    "sf": "Semi-final", "final": "Final", "champion": "Champion",
}


# ─────────────────────────────────────────────────────────────────────────────
#  One stochastic tournament — instrumented to record per-team journeys.
# ─────────────────────────────────────────────────────────────────────────────
def _ko(probs, t1, t2):
    p1, p2, _ = _get_probs(probs, t1, t2)
    total = p1 + p2
    if random.random() < p1 / total:
        return t1, t2
    return t2, t1


def simulate_once(probs):
    """Return per-team dict: {team: {stage_reached, group_rank, eliminator, path}}."""
    group_standings = {}
    for grp, teams in GROUPS.items():
        rec = {t: {"team": t, "pts": 0, "gf": 0, "ga": 0} for t in teams}
        for i in range(4):
            for j in range(i + 1, 4):
                t1, t2 = teams[i], teams[j]
                res = _sim_group_match(probs, t1, t2)
                g1, g2 = _rand_scoreline(res)
                if res == "team1":
                    rec[t1]["pts"] += 3
                elif res == "team2":
                    rec[t2]["pts"] += 3
                else:
                    rec[t1]["pts"] += 1
                    rec[t2]["pts"] += 1
                rec[t1]["gf"] += g1; rec[t1]["ga"] += g2
                rec[t2]["gf"] += g2; rec[t2]["ga"] += g1
        standings = sorted(
            rec.values(),
            key=lambda r: (-r["pts"], -(r["gf"] - r["ga"]), -r["gf"], random.random()),
        )
        group_standings[grp] = standings

    info = {t: {"stage": "group", "group_rank": None, "eliminator": "(Group stage)",
                "path": []} for t in ALL_TEAMS}
    for grp, st in group_standings.items():
        for i, r in enumerate(st):
            info[r["team"]]["group_rank"] = i + 1

    slots = {}
    for grp, st in group_standings.items():
        slots[f"1{grp}"] = st[0]["team"]
        slots[f"2{grp}"] = st[1]["team"]

    all_thirds = [{**st[2], "group": grp} for grp, st in group_standings.items()]
    all_thirds.sort(key=lambda r: (-r["pts"], -(r["gf"] - r["ga"]), -r["gf"], random.random()))
    best_thirds = all_thirds[:8]
    third_assignment = _assign_third_place(best_thirds)

    def resolve(slot):
        return slots[slot] if isinstance(slot, str) else third_assignment[slot]

    # All 32 qualifiers reach the knockouts.
    r32_teams = [(resolve(a), resolve(b)) for a, b in R32_BRACKET]
    for t1, t2 in r32_teams:
        for t in (t1, t2):
            info[t]["stage"] = "ko"
            rank = info[t]["group_rank"]
            placing = {1: "1st", 2: "2nd", 3: "3rd"}.get(rank, f"{rank}th")
            info[t]["path"].append(f"Grp {TEAM_GROUP[t]} {placing}")

    def play_round(pairs, stage_name):
        winners = []
        for t1, t2 in pairs:
            w, l = _ko(probs, t1, t2)
            winners.append(w)
            info[l]["eliminator"] = w
            info[w]["stage"] = stage_name
            info[w]["path"].append(f"def {l}")
        return winners

    r32_w = play_round(r32_teams, "r16")
    r16_w = play_round([(r32_w[i], r32_w[i + 1]) for i in range(0, 16, 2)], "qf")
    qf_w  = play_round([(r16_w[i], r16_w[i + 1]) for i in range(0, 8, 2)], "sf")
    sf_pairs = [(qf_w[0], qf_w[1]), (qf_w[2], qf_w[3])]
    sf_w  = play_round(sf_pairs, "final")
    champ = play_round([(sf_w[0], sf_w[1])], "champion")[0]

    return info, champ, sf_w


# ─────────────────────────────────────────────────────────────────────────────
#  Monte-Carlo aggregation
# ─────────────────────────────────────────────────────────────────────────────
def run_monte_carlo(probs, n):
    reach = {t: Counter() for t in ALL_TEAMS}          # stage -> count
    group_rank = {t: Counter() for t in ALL_TEAMS}     # rank -> count
    eliminators = {t: Counter() for t in ALL_TEAMS}    # who knocked them out
    paths = {t: Counter() for t in ALL_TEAMS}          # full journey (with opponents) -> count
    journeys = {t: Counter() for t in ALL_TEAMS}       # coarse journey (placement+exit) -> count
    champions = Counter()
    finals_pairs = Counter()

    for s in range(n):
        info, champ, finalists = simulate_once(probs)
        champions[champ] += 1
        finals_pairs[tuple(sorted(finalists))] += 1
        for t, d in info.items():
            group_rank[t][d["group_rank"]] += 1
            # monotonic ladder credit
            reached = d["stage"]
            idx = STAGES.index(reached) if reached in STAGES else -1
            for k in range(idx + 1):
                reach[t][STAGES[k]] += 1
            eliminators[t][d["eliminator"]] += 1
            paths[t][" → ".join(d["path"])] += 1
            # coarse journey, stored STRUCTURED (rank, exit stage, eliminator) so the
            # front-end can format it in any language.
            st = d["stage"]
            elim = None if st in ("champion", "group") else d["eliminator"]
            # the champion narrative ignores group placement, so collapse rank there
            rank_key = None if st == "champion" else d["group_rank"]
            journeys[t][(rank_key, st, elim)] += 1

    teams_out = {}
    for t in ALL_TEAMS:
        elim = eliminators[t].most_common()
        # most common eliminator that is an actual team (not group stage)
        top_team_elim = next(((k, v) for k, v in elim if k != "(Group stage)"), None)
        modal_path, modal_path_n = paths[t].most_common(1)[0]

        def _jobj(key, count):
            rank, stage, elim = key
            return {"rank": rank, "stage": stage, "eliminator": elim, "prob": count / n}
        modal_key, modal_journey_n = journeys[t].most_common(1)[0]
        modal_journey = _jobj(modal_key, modal_journey_n)
        top_journeys = [_jobj(k, v) for k, v in journeys[t].most_common(4)]
        teams_out[t] = {
            "team": t,
            "group": TEAM_GROUP[t],
            "stage_prob": {st: reach[t][st] / n for st in STAGES},
            "champion_count": champions[t],
            "champion_prob": champions[t] / n,
            "group_win_count": group_rank[t][1],
            "group_win_prob": group_rank[t][1] / n,
            "group_rank_dist": {str(r): group_rank[t][r] / n for r in (1, 2, 3, 4)},
            "group_stage_exit_prob": eliminators[t]["(Group stage)"] / n,
            "top_eliminator": (
                {"team": top_team_elim[0], "count": top_team_elim[1],
                 "share_of_elims": top_team_elim[1] / max(1, n - champions[t])}
                if top_team_elim else None
            ),
            "eliminator_breakdown": [
                {"by": k, "count": v, "prob": v / n} for k, v in elim[:6]
            ],
            "modal_path": {"path": modal_path, "count": modal_path_n, "prob": modal_path_n / n},
            "modal_journey": modal_journey,
            "top_journeys": top_journeys,
        }

    final_pair_top = [
        {"pair": list(p), "count": c, "prob": c / n}
        for p, c in finals_pairs.most_common(8)
    ]
    return teams_out, champions, final_pair_top


# ─────────────────────────────────────────────────────────────────────────────
#  Group-stage predictions (per group, per team advance / win probabilities)
# ─────────────────────────────────────────────────────────────────────────────
def group_predictions(teams_out):
    out = {}
    for grp, teams in GROUPS.items():
        rows = []
        for t in teams:
            d = teams_out[t]
            rows.append({
                "team": t,
                "win_group": d["group_win_prob"],
                "advance": d["stage_prob"]["ko"],
                "rank_dist": d["group_rank_dist"],
            })
        rows.sort(key=lambda r: -r["advance"])
        out[grp] = rows
    return out


# ─────────────────────────────────────────────────────────────────────────────
#  Deterministic "chalk" bracket — favorite wins every match.
# ─────────────────────────────────────────────────────────────────────────────
def chalk_bracket(probs):
    def exp_points(t, opp):
        p1, p2, pd_ = _get_probs(probs, t, opp)
        return 3 * p1 + 1 * pd_

    def winprob(t, opp):
        p1, p2, _ = _get_probs(probs, t, opp)
        return p1 / (p1 + p2)

    group_tables = {}
    slots = {}
    all_thirds = []
    for grp, teams in GROUPS.items():
        scored = []
        for t in teams:
            ep = sum(exp_points(t, o) for o in teams if o != t)
            avg_wp = sum(winprob(t, o) for o in teams if o != t) / 3
            scored.append({"team": t, "xpts": round(ep, 2), "strength": avg_wp})
        scored.sort(key=lambda r: (-r["xpts"], -r["strength"]))
        for i, r in enumerate(scored):
            r["rank"] = i + 1
        group_tables[grp] = scored
        slots[f"1{grp}"] = scored[0]["team"]
        slots[f"2{grp}"] = scored[1]["team"]
        all_thirds.append({**scored[2], "group": grp})

    all_thirds.sort(key=lambda r: (-r["xpts"], -r["strength"]))
    best_thirds = all_thirds[:8]
    third_assignment = _assign_third_place(
        [{"team": t["team"], "group": t["group"]} for t in best_thirds]
    )

    def resolve(slot):
        return slots[slot] if isinstance(slot, str) else third_assignment[slot]

    def play(pairs):
        res = []
        for t1, t2 in pairs:
            w = t1 if winprob(t1, t2) >= 0.5 else t2
            res.append({"t1": t1, "t2": t2, "winner": w, "p": round(max(winprob(t1, t2), 1 - winprob(t1, t2)), 3)})
        return res

    r32 = play([(resolve(a), resolve(b)) for a, b in R32_BRACKET])
    w32 = [m["winner"] for m in r32]
    r16 = play([(w32[i], w32[i + 1]) for i in range(0, 16, 2)])
    w16 = [m["winner"] for m in r16]
    qf  = play([(w16[i], w16[i + 1]) for i in range(0, 8, 2)])
    w8  = [m["winner"] for m in qf]
    sf  = play([(w8[0], w8[1]), (w8[2], w8[3])])
    w4  = [m["winner"] for m in sf]
    final = play([(w4[0], w4[1])])
    return {
        "groups": group_tables,
        "best_thirds": [t["team"] for t in best_thirds],
        "r32": r32, "r16": r16, "qf": qf, "sf": sf, "final": final,
        "champion": final[0]["winner"],
    }


# ─────────────────────────────────────────────────────────────────────────────
#  Model bias + confidence analysis (from the 5 model CSVs).
# ─────────────────────────────────────────────────────────────────────────────
def team_scores(df, n=47):
    as_t1 = df.groupby("team_1")["team_1_win_prob"].sum()
    as_t2 = df.groupby("team_2")["team_2_win_prob"].sum()
    return (as_t1.add(as_t2, fill_value=0)) / n


def model_analysis():
    dfs = {m: pd.read_csv(os.path.join(DATA_DIR, f"wc_2026_match_probabilities_{m}.csv"))
           for m in MODELS}
    avg = pd.read_csv(os.path.join(DATA_DIR, "wc_2026_match_probabilities_avg.csv"))
    avg_scores = team_scores(avg)

    bias = {}
    for m, df in dfs.items():
        sc = team_scores(df)
        raw_delta = (sc - avg_scores).dropna()
        delta = (raw_delta - raw_delta.mean()).sort_values(ascending=False)
        loved = [{"team": t, "delta": round(v, 4), "score": round(sc[t], 4)}
                 for t, v in delta.head(6).items()]
        dread = [{"team": t, "delta": round(v, 4), "score": round(sc[t], 4)}
                 for t, v in delta.tail(6).sort_values().items()]
        top_fav = [{"team": t, "score": round(v, 4)}
                   for t, v in sc.sort_values(ascending=False).head(8).items()]
        bias[m] = {"label": MODEL_LABELS[m], "loved": loved, "dreaded": dread,
                   "top_favorites": top_fav}

    # Confidence / decisiveness metrics
    conf = []
    for m, df in dfs.items():
        p1 = df["team_1_win_prob"]; p2 = df["team_2_win_prob"]; pdr = df["draw_prob"]
        maxout = df[["team_1_win_prob", "team_2_win_prob", "draw_prob"]].max(axis=1)
        # mean absolute deviation from uniform (1/3,1/3,1/3)
        dev = ((p1 - 1/3).abs() + (p2 - 1/3).abs() + (pdr - 1/3).abs()) / 3
        conf.append({
            "model": m, "label": MODEL_LABELS[m],
            "mean_draw": round(pdr.mean(), 4),
            "mean_top_outcome": round(maxout.mean(), 4),
            "mean_decisiveness": round((p1 - p2).abs().mean(), 4),
            "mean_dev_from_uniform": round(dev.mean(), 4),
        })
    conf.sort(key=lambda r: -r["mean_dev_from_uniform"])

    # Cross-model disagreement per matchup (spread in stronger-side win prob).
    norm = {}
    for m, df in dfs.items():
        d = df.copy()
        # canonical pair key (string so pandas .loc treats it as a label)
        d["key"] = d.apply(lambda r: " ||| ".join(sorted([r["team_1"], r["team_2"]])), axis=1)
        # store team_1 (alphabetical-first) win prob
        def first_win(r):
            a, b = sorted([r["team_1"], r["team_2"]])
            if r["team_1"] == a:
                return r["team_1_win_prob"]
            return r["team_2_win_prob"]
        d["first_win"] = d.apply(first_win, axis=1)
        norm[m] = d.set_index("key")["first_win"]
    mat = pd.DataFrame(norm)
    spread = (mat.max(axis=1) - mat.min(axis=1))
    biggest_disagreements = [
        {"pair": k.split(" ||| "), "spread": round(v, 4),
         "by_model": {MODEL_LABELS[m]: round(mat.loc[k, m], 3) for m in MODELS}}
        for k, v in spread.sort_values(ascending=False).head(8).items()
    ]
    return bias, conf, biggest_disagreements


def consensus_extremes():
    avg = pd.read_csv(os.path.join(DATA_DIR, "wc_2026_match_probabilities_avg.csv"))
    avg = avg.copy()
    avg["maxwin"] = avg[["team_1_win_prob", "team_2_win_prob"]].max(axis=1)
    coin = avg.sort_values("maxwin").head(8)
    coin_flips = [{"t1": r.team_1, "t2": r.team_2,
                   "p1": r.team_1_win_prob, "p2": r.team_2_win_prob, "draw": r.draw_prob}
                  for r in coin.itertuples()]
    lop = avg.sort_values("maxwin", ascending=False).head(8)
    lopsided = [{"t1": r.team_1, "t2": r.team_2,
                 "p1": r.team_1_win_prob, "p2": r.team_2_win_prob, "draw": r.draw_prob}
                for r in lop.itertuples()]
    return coin_flips, lopsided


# ─────────────────────────────────────────────────────────────────────────────
def main():
    random.seed(SEED)
    print(f"Loading consensus probabilities, running {N_SIMS} simulations…")
    probs = load_probabilities(os.path.join(DATA_DIR, "wc_2026_match_probabilities_avg.csv"))

    teams_out, champions, final_pairs = run_monte_carlo(probs, N_SIMS)
    print("  monte-carlo done")
    groups = group_predictions(teams_out)
    chalk = chalk_bracket(probs)
    print(f"  chalk champion: {chalk['champion']}")
    bias, conf, disagreements = model_analysis()
    coin_flips, lopsided = consensus_extremes()

    title_odds = sorted(
        [{"team": t, "prob": teams_out[t]["champion_prob"],
          "reach_final": teams_out[t]["stage_prob"]["final"],
          "reach_sf": teams_out[t]["stage_prob"]["sf"]}
         for t in ALL_TEAMS],
        key=lambda r: -r["prob"],
    )

    # dark horses: deterministic seeding (chalk depth) vs MC title odds
    chalk_depth = {}
    for r in chalk["r32"]:
        chalk_depth[r["t1"]] = chalk_depth.get(r["t1"], 0)
        chalk_depth[r["t2"]] = chalk_depth.get(r["t2"], 0)
    for rd, pts in [("r32", 1), ("r16", 2), ("qf", 3), ("sf", 4), ("final", 5)]:
        for m in chalk[rd]:
            chalk_depth[m["winner"]] = pts + 1
    chalk_depth[chalk["champion"]] = 7

    out = {
        "meta": {
            "n_sims": N_SIMS, "seed": SEED, "source": "consensus (avg of 5 models)",
            "models": [MODEL_LABELS[m] for m in MODELS],
            "stages": STAGES, "stage_label": STAGE_LABEL,
        },
        "title_odds": title_odds,
        "teams": teams_out,
        "groups": groups,
        "chalk": chalk,
        "model_bias": bias,
        "model_confidence": conf,
        "disagreements": disagreements,
        "coin_flips": coin_flips,
        "lopsided": lopsided,
        "final_pairs": final_pairs,
        "most_common_champion": champions.most_common(1)[0][0],
    }
    out_path = os.path.join(HERE, "report_data.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"Wrote {out_path}")

    # quick console sanity
    print("\nTop 10 title odds (consensus, %s sims):" % N_SIMS)
    for r in title_odds[:10]:
        print(f"  {r['team']:<22} {r['prob']*100:5.2f}%")


if __name__ == "__main__":
    main()
