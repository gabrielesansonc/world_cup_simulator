"""
app.py — Flask backend for the World Cup 2026 Simulator frontend.
Serves the static HTML frontend and exposes three JSON endpoints.
"""

import logging
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS

from wc_2026_simulator_function import simulate_world_cup_full

# ── App setup ─────────────────────────────────────────────────────────────────

FRONTEND_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "frontend")
DATA_DIR     = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")

app = Flask(__name__, static_folder=FRONTEND_DIR, static_url_path="")
CORS(app)

MODELS = {
    "avg":          "Average (Consensus)",
    "gemini_pro_31":"Gemini Pro 3.1",
    "gpt_52":       "GPT-5",
    "grok_42":      "Grok 4.2",
    "opus_46":      "Claude Opus 4.6",
    "sonnet_46":    "Claude Sonnet 4.6",
    "sonnet_opus":  "Sonnet + Opus Blend",
}

# ── Static frontend ───────────────────────────────────────────────────────────

@app.route("/")
def index():
    return send_from_directory(FRONTEND_DIR, "index.html")


# ── API: list models ──────────────────────────────────────────────────────────

@app.route("/api/models")
def get_models():
    return jsonify([{"id": k, "name": v} for k, v in MODELS.items()])


# ── API: single simulation ────────────────────────────────────────────────────

@app.route("/api/simulate/single", methods=["POST"])
def simulate_single():
    data        = request.get_json(force=True) or {}
    model       = data.get("model", "avg")
    seed        = data.get("seed", None)
    temperature = float(data.get("temperature", 1.0))

    if model not in MODELS:
        return jsonify({"error": "Invalid model"}), 400
    if seed is not None:
        seed = int(seed)

    csv_path = os.path.join(DATA_DIR, f"wc_2026_match_probabilities_{model}.csv")
    t0 = time.time()
    result = simulate_world_cup_full(csv_path, seed=seed, temperature=temperature)
    logging.info("SINGLE model=%s seed=%s temp=%s elapsed=%.2fs", model, seed, temperature, time.time() - t0)
    return jsonify(result)


# ── API: multiple simulations ─────────────────────────────────────────────────

@app.route("/api/simulate/multiple", methods=["POST"])
def simulate_multiple():
    data        = request.get_json(force=True) or {}
    model       = data.get("model", "avg")
    n           = min(int(data.get("n", 100)), 100000)
    temperature = float(data.get("temperature", 1.0))

    if model not in MODELS:
        return jsonify({"error": "Invalid model"}), 400

    csv_path = os.path.join(DATA_DIR, f"wc_2026_match_probabilities_{model}.csv")
    t0 = time.time()

    podium_table  = []
    podium_counts = {"champion": {}, "runner_up": {}, "third": {}, "fourth": {}}

    # bracket slot counters: round → [match_idx → {(t1,t2): count}]
    bracket_counts = {
        "r32":         [{} for _ in range(16)],
        "r16":         [{} for _ in range(8)],
        "qf":          [{} for _ in range(4)],
        "sf":          [{} for _ in range(2)],
        "third_place": [{}],
        "final":       [{}],
    }

    # group position counters: group → position_str → {team: count}
    group_counts = {
        grp: {"1": {}, "2": {}, "3": {}, "4": {}}
        for grp in "ABCDEFGHIJKL"
    }

    # last-stage counters: team → stage → count
    # stages: groups, r32, r16, qf, sf, final  (3rd-place match is ignored)
    last_stage_counts: dict = {}

    def _track_stage(team, stage):
        if team not in last_stage_counts:
            last_stage_counts[team] = {}
        last_stage_counts[team][stage] = last_stage_counts[team].get(stage, 0) + 1

    # opponent counters: team → stage → opponent → count (across all sims)
    stage_opponents: dict = {}

    def _track_opponent(team, stage, opponent):
        if team not in stage_opponents:
            stage_opponents[team] = {}
        if stage not in stage_opponents[team]:
            stage_opponents[team][stage] = {}
        d = stage_opponents[team][stage]
        d[opponent] = d.get(opponent, 0) + 1

    for i in range(n):
        r   = simulate_world_cup_full(csv_path, seed=i, temperature=temperature)
        pod = r["podium"]

        podium_table.append({
            "sim":       i + 1,
            "champion":  pod["champion"],
            "runner_up": pod["runner_up"],
            "third":     pod["third"],
            "fourth":    pod["fourth"],
        })

        for pos in ("champion", "runner_up", "third", "fourth"):
            team = pod[pos]
            podium_counts[pos][team] = podium_counts[pos].get(team, 0) + 1

        # bracket frequencies
        for round_name, matches in [
            ("r32",         r["r32"]),
            ("r16",         r["r16"]),
            ("qf",          r["qf"]),
            ("sf",          r["sf"]),
            ("third_place", [r["third_place_match"]]),
            ("final",       [r["final"]]),
        ]:
            for idx, match in enumerate(matches):
                key = tuple(sorted([match["t1"], match["t2"]]))
                d   = bracket_counts[round_name][idx]
                d[key] = d.get(key, 0) + 1

        # group position frequencies
        for grp, gdata in r["groups"].items():
            for standing in gdata["standings"]:
                pos  = str(standing["rank"])
                team = standing["team"]
                group_counts[grp][pos][team] = group_counts[grp][pos].get(team, 0) + 1

        # ── last-stage tracking ──────────────────────────────────────────────
        r32_teams = {m["t1"] for m in r["r32"]} | {m["t2"] for m in r["r32"]}
        all_group_teams = {
            s["team"]
            for gdata in r["groups"].values()
            for s in gdata["standings"]
        }
        for team in all_group_teams - r32_teams:
            _track_stage(team, "groups")
        for m in r["r32"]:
            _track_stage(m["t1"] if m["winner"] == m["t2"] else m["t2"], "r32")
        for m in r["r16"]:
            _track_stage(m["t1"] if m["winner"] == m["t2"] else m["t2"], "r16")
        for m in r["qf"]:
            _track_stage(m["t1"] if m["winner"] == m["t2"] else m["t2"], "qf")
        for m in r["sf"]:
            _track_stage(m["t1"] if m["winner"] == m["t2"] else m["t2"], "sf")
        _track_stage(r["final"]["t1"], "final")
        _track_stage(r["final"]["t2"], "final")

        # ── opponent tracking ────────────────────────────────────────────────
        for grp, gdata in r["groups"].items():
            for match in gdata["matches"]:
                _track_opponent(match["t1"], "groups", match["t2"])
                _track_opponent(match["t2"], "groups", match["t1"])
        for stage_name, matches in [
            ("r32",   r["r32"]),
            ("r16",   r["r16"]),
            ("qf",    r["qf"]),
            ("sf",    r["sf"]),
            ("final", [r["final"]]),
        ]:
            for match in matches:
                _track_opponent(match["t1"], stage_name, match["t2"])
                _track_opponent(match["t2"], stage_name, match["t1"])

    # ── Build probability table (teams with >= 1% in any position) ────────────
    all_teams = set()
    for pd in podium_counts.values():
        all_teams.update(pd.keys())

    probabilities = {}
    for team in all_teams:
        row    = {}
        passes = False
        for pos in ("champion", "runner_up", "third", "fourth"):
            pct = podium_counts[pos].get(team, 0) / n * 100
            row[pos] = round(pct, 1)
            if pct >= 1.0:
                passes = True
        if passes:
            row["top4"] = round(
                sum(podium_counts[pos].get(team, 0) for pos in podium_counts) / n * 100, 1
            )
            probabilities[team] = row

    # sort by champion probability descending, return as list to preserve order
    probabilities = [
        {"team": team, **row}
        for team, row in sorted(probabilities.items(), key=lambda x: -x[1]["champion"])
    ]

    # ── Build bracket top-3 per slot ──────────────────────────────────────────
    bracket_top3 = {}
    for round_name, slot_dicts in bracket_counts.items():
        bracket_top3[round_name] = []
        for counts in slot_dicts:
            top = sorted(counts.items(), key=lambda x: -x[1])[:3]
            bracket_top3[round_name].append([
                {"t1": pair[0], "t2": pair[1], "pct": round(cnt / n * 100, 1)}
                for pair, cnt in top
            ])

    # ── Build group top-3 per position ────────────────────────────────────────
    group_frequencies = {}
    for grp, positions in group_counts.items():
        group_frequencies[grp] = {}
        for pos, counts in positions.items():
            top = sorted(counts.items(), key=lambda x: -x[1])[:3]
            group_frequencies[grp][pos] = [
                {"team": team, "pct": round(cnt / n * 100, 1)}
                for team, cnt in top
            ]

    # ── Build last-stage probabilities (sorted alphabetically by team) ──────────
    stage_order = ["groups", "r32", "r16", "qf", "sf", "final"]
    team_stage_probs = {
        team: {s: round(counts.get(s, 0) / n * 100, 1) for s in stage_order}
        for team, counts in sorted(last_stage_counts.items())
    }

    # ── Build top-3 opponents per team per stage ─────────────────────────────
    team_stage_opponents = {}
    for team, stages in sorted(stage_opponents.items()):
        team_stage_opponents[team] = {}
        for stage, opp_counts in stages.items():
            top3 = sorted(opp_counts.items(), key=lambda x: -x[1])[:3]
            team_stage_opponents[team][stage] = [
                {"team": opp, "pct": round(cnt / n * 100, 1)}
                for opp, cnt in top3
            ]

    logging.info("MULTI model=%s n=%d temp=%s elapsed=%.2fs", model, n, temperature, time.time() - t0)
    return jsonify({
        "n":                    n,
        "podium_table":         podium_table,
        "probabilities":        probabilities,
        "bracket_top3":         bracket_top3,
        "group_frequencies":    group_frequencies,
        "team_stage_probs":     team_stage_probs,
        "team_stage_opponents": team_stage_opponents,
    })


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    app.run(debug=True, port=5001)
