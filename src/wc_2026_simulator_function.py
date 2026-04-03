"""
wc_2026_simulator.py
Simulate the 2026 FIFA World Cup (48-team format).

Usage
-----
    python wc_2026_simulator.py [csv_path] [--silent] [--seed N]

Examples
--------
    python wc_2026_simulator.py wc_2026_match_probabilities_sonnet_opus.csv
    python wc_2026_simulator.py wc_2026_match_probabilities_opus.csv --silent
    python wc_2026_simulator.py wc_2026_match_probabilities_sonnet_opus.csv --seed 42
"""

import math
import pandas as pd
import random
import sys
from typing import Optional

# ─── 2026 World Cup Groups ────────────────────────────────────────────────────
# Official draw — December 5, 2025, Washington D.C.
# Hosts: Mexico (A), Canada (B), USA (D).
GROUPS: dict[str, list[str]] = {
    "A": ["Mexico",    "South Korea",             "South Africa",  "Czechia"],
    "B": ["Canada",    "Switzerland",             "Qatar",         "Bosnia and Herzegovina"],
    "C": ["Brazil",    "Morocco",                 "Scotland",      "Haiti"],
    "D": ["USA",       "Paraguay",                "Australia",     "Turkey"],
    "E": ["Germany",   "Ecuador",                 "Côte d'Ivoire", "Curaçao"],
    "F": ["Netherlands","Japan",                  "Tunisia",       "Sweden"],
    "G": ["Belgium",   "Iran",                    "Egypt",         "New Zealand"],
    "H": ["Spain",     "Uruguay",                 "Saudi Arabia",  "Cape Verde"],
    "I": ["France",    "Senegal",                 "Norway",        "Iraq"],
    "J": ["Argentina", "Austria",                 "Algeria",       "Jordan"],
    "K": ["Portugal",  "Colombia",                "Uzbekistan",    "DR Congo"],
    "L": ["England",   "Croatia",                 "Panama",        "Ghana"],
}

# Round of 32 — official 2026 bracket.
# Left side: matches 1-8.  Right side: matches 9-16.
# Regular slots are strings ("1A", "2B" …).
# Third-place slots are tuples of the eligible source groups.
R32_BRACKET: list[tuple] = [
    # Left side
    ("1E",  ("A","B","C","D","F")),
    ("1I",  ("C","D","E","F","G","H")),
    ("2A",  "2B"),
    ("1F",  "2C"),
    ("2K",  "2L"),
    ("1H",  "2J"),
    ("1D",  ("B","E","F","I","J")),
    ("1G",  ("A","E","H","I","J")),
    # Right side
    ("1C",  "2F"),
    ("2E",  "2I"),
    ("1A",  ("C","E","F","H","I")),
    ("1L",  ("E","H","I","J","K")),
    ("1J",  "2H"),
    ("2D",  "2G"),
    ("1B",  ("E","F","G","I","J")),
    ("1K",  ("D","E","I","J","L")),
]

# Tiebreaker: used when teams cannot otherwise be separated.
_TINY = 1e-9


# ─── Probability Loader ───────────────────────────────────────────────────────

def load_probabilities(csv_path: str) -> dict[tuple[str, str], tuple[float, float, float]]:
    """
    Load match probabilities from CSV.

    Expected columns: team_1, team_2, team_1_win_prob, team_2_win_prob, draw_prob.
    Both orderings (t1,t2) and (t2,t1) are indexed so lookups are order-independent.

    Returns
    -------
    dict mapping (team_a, team_b) -> (p_a_wins, p_b_wins, p_draw)
    """
    df = pd.read_csv(csv_path)
    probs: dict = {}
    for _, row in df.iterrows():
        t1, t2 = row["team_1"], row["team_2"]
        p1, p2, pd_ = row["team_1_win_prob"], row["team_2_win_prob"], row["draw_prob"]
        probs[(t1, t2)] = (p1, p2, pd_)
        probs[(t2, t1)] = (p2, p1, pd_)
    return probs


def _get_probs(probs: dict, t1: str, t2: str) -> tuple[float, float, float]:
    """Return (p1_win, p2_win, p_draw).  Falls back to equal thirds if unknown."""
    return probs.get((t1, t2), (1/3, 1/3, 1/3))


def _apply_temperature(probs: dict, temperature: float) -> dict:
    """
    Return a new probs dict with temperature scaling applied.

    Works exactly like LLM temperature: divide log-probabilities by `temperature`,
    then re-normalise via softmax.

      temperature < 1.0  →  sharper  (favourites win more, fewer upsets)
      temperature = 1.0  →  unchanged
      temperature > 1.0  →  flatter  (closer to 33/33/33, more chaos)

    Example: (0.60, 0.30, 0.10) at temperature=0.7 → (~0.69, ~0.26, ~0.05)
    """
    if temperature == 1.0:
        return probs
    scaled: dict = {}
    for key, (p1, p2, pd) in probs.items():
        logs = [math.log(max(p, 1e-10)) / temperature for p in (p1, p2, pd)]
        m    = max(logs)                              # numeric stability
        exps = [math.exp(l - m) for l in logs]
        s    = sum(exps)
        scaled[key] = (exps[0] / s, exps[1] / s, exps[2] / s)
    return scaled


# ─── Match Simulators ─────────────────────────────────────────────────────────

def _sim_group_match(probs: dict, t1: str, t2: str) -> str:
    """Simulate a group-stage match (draws allowed).  Returns 'team1'|'draw'|'team2'."""
    p1, p2, pd_ = _get_probs(probs, t1, t2)
    r = random.random()
    if r < p1:          return "team1"
    elif r < p1 + pd_:  return "draw"
    else:               return "team2"


def _sim_ko_match(probs: dict, t1: str, t2: str) -> tuple[str, str]:
    """
    Simulate a knockout match — no draws.

    Draw probability is split proportionally between the two win probabilities:
        p1_adj = p1 / (p1 + p2),  p2_adj = p2 / (p1 + p2)
    e.g. (0.35, 0.35, 0.30) → (0.50, 0.50)

    Returns (winner, loser).
    """
    p1, p2, _ = _get_probs(probs, t1, t2)
    total = p1 + p2
    winner = t1 if random.random() < p1 / total else t2
    loser  = t2 if winner == t1 else t1
    return winner, loser


def _rand_scoreline(result: str) -> tuple[int, int]:
    """Generate a plausible scoreline for GD / display purposes."""
    if result == "team1":
        w = random.choices([1, 2, 3, 4], weights=[35, 40, 18, 7])[0]
        l = random.randint(0, w - 1)
        return w, l
    elif result == "team2":
        w, l = _rand_scoreline("team1")
        return l, w
    else:  # draw
        g = random.choices([0, 1, 2, 3], weights=[25, 45, 25, 5])[0]
        return g, g


# ─── Group Stage ──────────────────────────────────────────────────────────────

def _simulate_group(grp: str, teams: list[str], probs: dict, verbose: bool) -> list[dict]:
    """
    Simulate a 4-team round-robin group.

    Tiebreaker order: points → goal difference → goals for → random.
    Returns standings (list of dicts) sorted best → worst.
    """
    rec = {t: {"team": t, "pts": 0, "gf": 0, "ga": 0} for t in teams}
    matches = [(teams[i], teams[j]) for i in range(4) for j in range(i + 1, 4)]

    W = 56
    if verbose:
        print(f"\n  ┌─ Group {grp} {'─' * (W - 11)}")

    for t1, t2 in matches:
        result = _sim_group_match(probs, t1, t2)
        g1, g2 = _rand_scoreline(result)

        if result == "team1":
            rec[t1]["pts"] += 3
        elif result == "team2":
            rec[t2]["pts"] += 3
        else:
            rec[t1]["pts"] += 1
            rec[t2]["pts"] += 1

        rec[t1]["gf"] += g1;  rec[t1]["ga"] += g2
        rec[t2]["gf"] += g2;  rec[t2]["ga"] += g1

        if verbose:
            print(f"  │  {t1:<30} {g1}–{g2}  {t2}")

    standings = sorted(
        rec.values(),
        key=lambda r: (-r["pts"], -(r["gf"] - r["ga"]), -r["gf"], random.random()),
    )

    if verbose:
        print("  │")
        for i, r in enumerate(standings, 1):
            gd  = r["gf"] - r["ga"]
            tag = "Q" if i <= 2 else "·"
            print(f"  │  {tag} {i}. {r['team']:<30} {r['pts']} pts  GD {gd:+d}")
        print(f"  └{'─' * (W - 2)}")

    return list(standings)


# ─── Knockout Round ───────────────────────────────────────────────────────────

def _play_round(
    label: str,
    matchups: list[tuple[str, str]],
    probs: dict,
    verbose: bool,
) -> list[tuple[str, str]]:
    """
    Simulate a complete knockout round.

    Returns list of (winner, loser) tuples in bracket order.
    """
    results: list[tuple[str, str]] = []
    if verbose:
        pad = max(0, 48 - len(label))
        print(f"\n━━━  {label}  {'━' * pad}")

    for t1, t2 in matchups:
        winner, loser = _sim_ko_match(probs, t1, t2)
        results.append((winner, loser))
        if verbose:
            print(f"  {t1:<32} vs  {t2:<32}  →  {winner}")

    return results


# ─── Third-Place Bracket Assignment ──────────────────────────────────────────

def _assign_third_place(best_thirds: list[dict]) -> dict[tuple, str]:
    """
    Match the 8 best 3rd-place teams to their R32 bracket slots.

    Uses a most-constrained-slot-first heuristic: repeatedly pick the slot
    with the fewest eligible remaining teams and assign the highest-ranked
    eligible team to it.

    Parameters
    ----------
    best_thirds : list of dicts with 'team' and 'group' keys, sorted best → worst.

    Returns
    -------
    dict mapping eligible-group-tuple → team name.
    """
    third_specs: list[tuple] = [b for _, b in R32_BRACKET if isinstance(b, tuple)]

    remaining   = list(best_thirds)   # mutated as teams are assigned
    assignment: dict[tuple, str] = {}
    open_slots  = list(third_specs)

    while open_slots:
        slot = min(open_slots, key=lambda s: sum(1 for t in remaining if t["group"] in s))
        open_slots.remove(slot)

        for i, t in enumerate(remaining):
            if t["group"] in slot:
                assignment[slot] = t["team"]
                remaining.pop(i)
                break
        else:
            # Fallback: no eligible team left — assign best remaining
            assignment[slot] = remaining.pop(0)["team"]

    return assignment


# ─── Main Entry Point ─────────────────────────────────────────────────────────

def simulate_world_cup(
    csv_path: str,
    verbose: bool = True,
    seed: Optional[int] = None,
    temperature: float = 1.0,
) -> dict:
    """
    Simulate the full 2026 FIFA World Cup.

    Parameters
    ----------
    csv_path : str
        Path to a match-probabilities CSV.
        Required columns: team_1, team_2, team_1_win_prob, team_2_win_prob, draw_prob.
    verbose : bool
        True  → print group-stage results and every knockout match.
        False → print only the final podium.
    seed : int, optional
        Random seed for reproducibility.
    temperature : float
        Scales how extreme the probabilities are (default 1.0 = unchanged).
        < 1.0 → favourites win more often.  > 1.0 → more upsets.

    Returns
    -------
    dict with keys: champion, runner_up, third, fourth.
    """
    if seed is not None:
        random.seed(seed)

    probs = load_probabilities(csv_path)
    probs = _apply_temperature(probs, temperature)

    # ── Group Stage ─────────────────────────────────────────────────────────
    if verbose:
        print("=" * 62)
        print("        2026 FIFA WORLD CUP SIMULATION")
        print("=" * 62)
        print("\n━━━  GROUP STAGE  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

    group_standings: dict[str, list[dict]] = {}
    for grp, teams in GROUPS.items():
        group_standings[grp] = _simulate_group(grp, teams, probs, verbose)

    # ── Determine qualifiers ─────────────────────────────────────────────────
    slots: dict[str, str] = {}
    for grp, standings in group_standings.items():
        slots[f"1{grp}"] = standings[0]["team"]
        slots[f"2{grp}"] = standings[1]["team"]

    # 8 best 3rd-place teams (ranked by pts → GD → GF → random)
    all_thirds = [
        {**s[2], "group": grp}
        for grp, s in group_standings.items()
    ]
    all_thirds.sort(
        key=lambda r: (-r["pts"], -(r["gf"] - r["ga"]), -r["gf"], random.random())
    )
    best_thirds = all_thirds[:8]
    if verbose:
        print("\n━━━  3RD-PLACE QUALIFIERS  (best 8 of 12)  ━━━━━━━━━━━━━━━━━━")
        for i, t in enumerate(best_thirds, 1):
            gd = t["gf"] - t["ga"]
            print(f"  {i}. {t['team']:<32} (Grp {t['group']})  {t['pts']} pts  GD {gd:+d}")

    # ── Round of 32  (16 matches) ────────────────────────────────────────────
    third_assignment = _assign_third_place(best_thirds)

    def _resolve(slot) -> str:
        return slots[slot] if isinstance(slot, str) else third_assignment[slot]

    r32_matchups: list[tuple[str, str]] = [
        (_resolve(a), _resolve(b)) for a, b in R32_BRACKET
    ]

    r32 = _play_round("ROUND OF 32", r32_matchups, probs, verbose)

    # ── Round of 16  (8 matches) ─────────────────────────────────────────────
    r32_winners = [w for w, _ in r32]
    r16 = _play_round(
        "ROUND OF 16",
        [(r32_winners[i], r32_winners[i + 1]) for i in range(0, 16, 2)],
        probs, verbose,
    )

    # ── Quarter-Finals  (4 matches) ──────────────────────────────────────────
    r16_winners = [w for w, _ in r16]
    qf = _play_round(
        "QUARTER-FINALS",
        [(r16_winners[i], r16_winners[i + 1]) for i in range(0, 8, 2)],
        probs, verbose,
    )

    # ── Semi-Finals  (2 matches) ─────────────────────────────────────────────
    qf_winners = [w for w, _ in qf]
    sf = _play_round(
        "SEMI-FINALS",
        [(qf_winners[0], qf_winners[1]), (qf_winners[2], qf_winners[3])],
        probs, verbose,
    )
    sf_winners = [w for w, _ in sf]
    sf_losers  = [l for _, l in sf]

    # ── Third-Place Match ────────────────────────────────────────────────────
    tp = _play_round("THIRD PLACE MATCH", [(sf_losers[0], sf_losers[1])], probs, verbose)
    third_place  = tp[0][0]
    fourth_place = tp[0][1]

    # ── Final ────────────────────────────────────────────────────────────────
    final = _play_round("FINAL", [(sf_winners[0], sf_winners[1])], probs, verbose)
    champion  = final[0][0]
    runner_up = final[0][1]

    # ── Podium ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 62)
    print("                  FINAL RESULTS")
    print("=" * 62)
    print(f"  1st  {champion}")
    print(f"  2nd  {runner_up}")
    print(f"  3rd  {third_place}")
    print(f"  4th  {fourth_place}")
    print("=" * 62)

    return {
        "champion":   champion,
        "runner_up":  runner_up,
        "third":      third_place,
        "fourth":     fourth_place,
    }


# ─── Full-Data Simulation (for frontend) ─────────────────────────────────────

def _simulate_group_full(teams: list[str], probs: dict) -> tuple[list[dict], list[dict]]:
    """
    Like _simulate_group but returns (standings, matches) without printing.
    matches: list of {t1, g1, t2, g2}
    standings: list of {team, pts, gf, ga}  sorted best→worst
    """
    rec = {t: {"team": t, "pts": 0, "gf": 0, "ga": 0} for t in teams}
    match_pairs = [(teams[i], teams[j]) for i in range(4) for j in range(i + 1, 4)]
    matches_out = []

    for t1, t2 in match_pairs:
        result = _sim_group_match(probs, t1, t2)
        g1, g2 = _rand_scoreline(result)

        if result == "team1":
            rec[t1]["pts"] += 3
        elif result == "team2":
            rec[t2]["pts"] += 3
        else:
            rec[t1]["pts"] += 1
            rec[t2]["pts"] += 1

        rec[t1]["gf"] += g1;  rec[t1]["ga"] += g2
        rec[t2]["gf"] += g2;  rec[t2]["ga"] += g1
        matches_out.append({"t1": t1, "g1": g1, "t2": t2, "g2": g2})

    standings = sorted(
        rec.values(),
        key=lambda r: (-r["pts"], -(r["gf"] - r["ga"]), -r["gf"], random.random()),
    )
    return list(standings), matches_out


def simulate_world_cup_full(
    csv_path: str,
    seed: Optional[int] = None,
    temperature: float = 1.0,
) -> dict:
    """
    Simulate the full 2026 FIFA World Cup and return complete bracket data.

    Returns a dict with:
      groups, third_place_qualifiers, r32, r16, qf, sf,
      third_place_match, final, podium
    """
    if seed is not None:
        random.seed(seed)

    probs = load_probabilities(csv_path)
    probs = _apply_temperature(probs, temperature)

    # ── Group Stage ─────────────────────────────────────────────────────────
    group_data: dict = {}
    group_standings: dict = {}
    for grp, teams in GROUPS.items():
        standings, matches = _simulate_group_full(teams, probs)
        group_standings[grp] = standings
        group_data[grp] = {
            "standings": [
                {
                    "team": s["team"],
                    "pts":  s["pts"],
                    "gf":   s["gf"],
                    "ga":   s["ga"],
                    "gd":   s["gf"] - s["ga"],
                    "rank": i + 1,
                }
                for i, s in enumerate(standings)
            ],
            "matches": matches,
        }

    # ── Slots & 3rd-place qualifiers ────────────────────────────────────────
    slots: dict[str, str] = {}
    for grp, standings in group_standings.items():
        slots[f"1{grp}"] = standings[0]["team"]
        slots[f"2{grp}"] = standings[1]["team"]

    all_thirds = [
        {**s[2], "group": grp}
        for grp, s in group_standings.items()
    ]
    all_thirds.sort(
        key=lambda r: (-r["pts"], -(r["gf"] - r["ga"]), -r["gf"], random.random())
    )
    best_thirds = all_thirds[:8]

    # ── Round of 32 ─────────────────────────────────────────────────────────
    third_assignment = _assign_third_place(best_thirds)

    def _resolve(slot) -> str:
        return slots[slot] if isinstance(slot, str) else third_assignment[slot]

    r32_results = []
    for a, b in R32_BRACKET:
        t1, t2 = _resolve(a), _resolve(b)
        winner, _ = _sim_ko_match(probs, t1, t2)
        r32_results.append({"t1": t1, "t2": t2, "winner": winner})

    r32_winners = [m["winner"] for m in r32_results]

    # ── Round of 16 ─────────────────────────────────────────────────────────
    r16_results = []
    for i in range(0, 16, 2):
        t1, t2 = r32_winners[i], r32_winners[i + 1]
        winner, _ = _sim_ko_match(probs, t1, t2)
        r16_results.append({"t1": t1, "t2": t2, "winner": winner})

    r16_winners = [m["winner"] for m in r16_results]

    # ── Quarter-Finals ───────────────────────────────────────────────────────
    qf_results = []
    for i in range(0, 8, 2):
        t1, t2 = r16_winners[i], r16_winners[i + 1]
        winner, _ = _sim_ko_match(probs, t1, t2)
        qf_results.append({"t1": t1, "t2": t2, "winner": winner})

    qf_winners = [m["winner"] for m in qf_results]

    # ── Semi-Finals ─────────────────────────────────────────────────────────
    sf_results = []
    sf_losers  = []
    for t1, t2 in [(qf_winners[0], qf_winners[1]), (qf_winners[2], qf_winners[3])]:
        winner, loser = _sim_ko_match(probs, t1, t2)
        sf_results.append({"t1": t1, "t2": t2, "winner": winner})
        sf_losers.append(loser)

    sf_winners = [m["winner"] for m in sf_results]

    # ── Third-Place Match ────────────────────────────────────────────────────
    tp_winner, tp_loser = _sim_ko_match(probs, sf_losers[0], sf_losers[1])
    third_place_match = {"t1": sf_losers[0], "t2": sf_losers[1], "winner": tp_winner}

    # ── Final ────────────────────────────────────────────────────────────────
    f_winner, f_loser = _sim_ko_match(probs, sf_winners[0], sf_winners[1])
    final_match = {"t1": sf_winners[0], "t2": sf_winners[1], "winner": f_winner}

    return {
        "groups": group_data,
        "third_place_qualifiers": [
            {
                "team":  t["team"],
                "group": t["group"],
                "pts":   t["pts"],
                "gf":    t["gf"],
                "ga":    t["ga"],
                "gd":    t["gf"] - t["ga"],
            }
            for t in best_thirds
        ],
        "r32":               r32_results,
        "r16":               r16_results,
        "qf":                qf_results,
        "sf":                sf_results,
        "third_place_match": third_place_match,
        "final":             final_match,
        "podium": {
            "champion":  f_winner,
            "runner_up": f_loser,
            "third":     tp_winner,
            "fourth":    tp_loser,
        },
    }


# ─── CLI ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    args = sys.argv[1:]
    verbose = "--silent" not in args
    seed: Optional[int] = None
    temperature: float = 1.0

    # Parse --seed N and --temp F
    clean_args = []
    skip_next = False
    for i, a in enumerate(args):
        if skip_next:
            skip_next = False
            continue
        if a == "--seed" and i + 1 < len(args):
            seed = int(args[i + 1])
            skip_next = True
        elif a == "--temp" and i + 1 < len(args):
            temperature = float(args[i + 1])
            skip_next = True
        elif not a.startswith("--"):
            clean_args.append(a)

    csv = clean_args[0] if clean_args else "../data/wc_2026_match_probabilities_avg.csv"
    simulate_world_cup(csv, verbose=verbose, seed=seed, temperature=temperature)
