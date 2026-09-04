import math
import os

from flask import Flask, jsonify, render_template, request
import numpy as np
import pandas as pd

from engine import DraftIQEngine

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 8 * 1024 * 1024  # 8MB upload cap

UPLOAD_DIR = os.path.join(os.path.dirname(__file__), "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Single shared engine instance - this is a personal single-user draft tool,
# mirroring the original notebook's single global draft_state.
engine = DraftIQEngine()

# Auto-load the bundled sample dataset if present, so the app has something
# to show immediately. Real usage: upload your own players.csv from the UI.
_sample_path = os.path.join(UPLOAD_DIR, "sample_players.csv")
if os.path.exists(_sample_path):
    engine.load_players(_sample_path)


def clean(obj):
    """Recursively convert numpy/pandas scalar types to native Python and
    replace NaN/inf with None, so jsonify doesn't choke on either."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        val = float(obj)
        return None if (math.isnan(val) or math.isinf(val)) else val
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, float):
        return None if (math.isnan(obj) or math.isinf(obj)) else obj
    if obj is pd.NA or (isinstance(obj, float) and pd.isna(obj)):
        return None
    if isinstance(obj, dict):
        return {k: clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [clean(v) for v in obj]
    return obj


def df_records(df):
    if df is None or len(df) == 0:
        return []
    return clean(df.replace({pd.NA: None}).to_dict("records"))


# ----------------------------------------------------------------------
# Pages
# ----------------------------------------------------------------------

@app.route("/")
def index():
    return render_template("index.html")


# ----------------------------------------------------------------------
# Setup / league config
# ----------------------------------------------------------------------

@app.route("/api/upload", methods=["POST"])
def upload_players():
    file = request.files.get("file")
    if not file or file.filename == "":
        return jsonify({"ok": False, "message": "No file provided"}), 400

    save_path = os.path.join(UPLOAD_DIR, "players.csv")
    file.save(save_path)

    try:
        engine.load_players(save_path)
    except Exception as e:
        return jsonify({"ok": False, "message": f"Failed to parse CSV: {e}"}), 400

    return jsonify({"ok": True, "players_loaded": len(engine.players_df)})


@app.route("/api/settings", methods=["GET", "POST"])
def settings():
    if request.method == "POST":
        data = request.get_json(force=True) or {}

        structural_change = False
        if "num_teams" in data and int(data["num_teams"]) != engine.num_teams:
            engine.num_teams = int(data["num_teams"])
            structural_change = True
        if "rounds" in data and int(data["rounds"]) != engine.rounds:
            engine.rounds = int(data["rounds"])
            structural_change = True

        # Changing your draft slot does NOT need to wipe the draft in
        # progress - only team-count/round changes do, since those change
        # the league's actual structure.
        if "my_team" in data:
            engine.set_my_team(int(data["my_team"]))

        if "scoring" in data and data["scoring"]:
            engine.scoring = str(data["scoring"])

        if "starters" in data and isinstance(data["starters"], dict):
            engine.set_starters(data["starters"])

        if "aggressiveness" in data:
            engine.set_aggressiveness(data["aggressiveness"])

        if structural_change and engine.loaded:
            engine.league = engine._create_league()
            engine.draft_order = engine._snake_order()
            engine.reset_draft()

    return jsonify({
        "num_teams": engine.num_teams,
        "my_team": engine.my_team,
        "roster_size": engine.roster_size,
        "rounds": engine.rounds,
        "scoring": engine.scoring,
        "starters": engine.starters,
        "aggressiveness": engine.aggressiveness,
        "loaded": engine.loaded,
        "players_loaded": len(engine.players_df) if engine.loaded else 0,
    })


# ----------------------------------------------------------------------
# Draft state
# ----------------------------------------------------------------------

def _require_loaded():
    if not engine.loaded:
        return jsonify({"ok": False, "message": "No player data loaded yet. Upload a players.csv first."}), 400
    return None


@app.route("/api/state")
def state():
    err = _require_loaded()
    if err:
        return err
    return jsonify({
        "pick": engine.draft_state["pick"],
        "round": engine.draft_state["round"],
        "current_team": engine.get_current_team_number(),
        "my_team": engine.my_team,
        "on_the_clock": engine.get_current_team_number() == engine.my_team,
        "players_remaining": len(engine.get_available_players()),
        "history": engine.draft_state["history"][-15:],
        "draft_complete": engine.is_draft_complete(),
        "total_rounds": engine.rounds,
    })


@app.route("/api/roster")
def roster():
    err = _require_loaded()
    if err:
        return err
    return jsonify({
        "roster": engine.get_team_roster(engine.my_team),
        "slots": engine.get_roster_slots(engine.my_team),
        "needs": engine.get_roster_needs(),
        "position_counts": engine.get_position_counts(),
    })


@app.route("/api/league")
def league():
    err = _require_loaded()
    if err:
        return err
    teams = []
    for i, team in enumerate(engine.league, start=1):
        teams.append({
            "team": team["team"],
            "is_user": i == engine.my_team,
            "slots": engine.get_roster_slots(i),
        })
    return jsonify({"league": engine.league, "teams": teams})


@app.route("/api/board")
def board():
    err = _require_loaded()
    if err:
        return err
    return jsonify(clean(engine.get_draft_board()))


@app.route("/api/available")
def available():
    err = _require_loaded()
    if err:
        return err
    count = int(request.args.get("count", 25))
    df = engine.get_available_players().sort_values("rank").head(count)
    return jsonify({"players": df_records(df)})


@app.route("/api/reset", methods=["POST"])
def reset():
    err = _require_loaded()
    if err:
        return err
    engine.reset_draft()
    return jsonify({"ok": True})


# ----------------------------------------------------------------------
# Draft actions
# ----------------------------------------------------------------------

@app.route("/api/pick", methods=["POST"])
def pick():
    err = _require_loaded()
    if err:
        return err
    data = request.get_json(force=True)
    player_name = data.get("player")
    if not player_name:
        return jsonify({"ok": False, "message": "player is required"}), 400
    result = engine.draftiq_pick(player_name)
    return jsonify(clean(result))


@app.route("/api/target", methods=["POST"])
def target():
    err = _require_loaded()
    if err:
        return err
    data = request.get_json(force=True)
    player_name = data.get("player")
    action = data.get("action", "add")  # "add" or "remove"
    if not player_name:
        return jsonify({"ok": False, "message": "player is required"}), 400
    if action == "remove":
        engine.remove_target_player(player_name)
    else:
        engine.add_target_player(player_name)
    return jsonify({
        "ok": True,
        "targets": sorted(engine.target_players),
        "avoids": sorted(engine.avoid_players),
    })


@app.route("/api/avoid", methods=["POST"])
def avoid():
    err = _require_loaded()
    if err:
        return err
    data = request.get_json(force=True)
    player_name = data.get("player")
    action = data.get("action", "add")  # "add" or "remove"
    if not player_name:
        return jsonify({"ok": False, "message": "player is required"}), 400
    if action == "remove":
        engine.remove_avoid_player(player_name)
    else:
        engine.add_avoid_player(player_name)
    return jsonify({
        "ok": True,
        "targets": sorted(engine.target_players),
        "avoids": sorted(engine.avoid_players),
    })


@app.route("/api/flags")
def flags():
    """Current target/avoid lists, for the UI to mark player rows on load
    (e.g. after a page refresh) without re-deriving them from anywhere
    else."""
    err = _require_loaded()
    if err:
        return err
    return jsonify({
        "targets": sorted(engine.target_players),
        "avoids": sorted(engine.avoid_players),
    })


@app.route("/api/simulate_to_me", methods=["POST"])
def simulate_to_me():
    err = _require_loaded()
    if err:
        return err
    picks = engine.simulate_opponents_until_user()
    return jsonify({"ok": True, "picks": picks, "state": {
        "pick": engine.draft_state["pick"],
        "round": engine.draft_state["round"],
        "current_team": engine.get_current_team_number(),
    }})


@app.route("/api/start_draft", methods=["POST"])
def start_draft():
    """Resets the draft, then simulates opponent picks up to the user's
    first turn (a no-op if the user's slot picks first)."""
    err = _require_loaded()
    if err:
        return err
    engine.reset_draft()
    picks = engine.simulate_opponents_until_user()
    return jsonify({"ok": True, "picks": picks, "state": {
        "pick": engine.draft_state["pick"],
        "round": engine.draft_state["round"],
        "current_team": engine.get_current_team_number(),
    }})


@app.route("/api/draft_any", methods=["POST"])
def draft_any():
    """Assign a pick to whichever team is currently on the clock (for
    manually running a live draft where opponents pick for themselves)."""
    err = _require_loaded()
    if err:
        return err
    data = request.get_json(force=True)
    player_name = data.get("player")
    result = engine.draft_player(player_name)
    return jsonify(clean(result))


@app.route("/api/undo", methods=["POST"])
def undo():
    err = _require_loaded()
    if err:
        return err
    result = engine.undo_last_pick()
    return jsonify(clean(result))


# ----------------------------------------------------------------------
# Intelligence endpoints
# ----------------------------------------------------------------------

@app.route("/api/recommendations")
def recommendations():
    err = _require_loaded()
    if err:
        return err
    count = int(request.args.get("count", 10))
    df = engine.get_draft_recommendations(count)
    return jsonify({"recommendations": df_records(df)})


@app.route("/api/decision")
def decision():
    err = _require_loaded()
    if err:
        return err
    count = int(request.args.get("count", 10))
    df = engine.draftiq_decision_engine(count)
    return jsonify({"decisions": df_records(df)})


@app.route("/api/on_clock")
def on_clock():
    err = _require_loaded()
    if err:
        return err
    result = engine.draftiq_on_clock_decision()
    return jsonify(clean(result))


@app.route("/api/alerts")
def alerts():
    err = _require_loaded()
    if err:
        return err
    return jsonify({
        "draft_alerts": engine.get_draft_alerts(),
        "position_runs": engine.get_position_run_alerts(),
        "target_alerts": engine.get_target_alerts(),
    })


@app.route("/api/compare", methods=["POST"])
def compare():
    err = _require_loaded()
    if err:
        return err
    data = request.get_json(force=True)
    names = data.get("players", [])
    df = engine.compare_players(names)
    return jsonify({"comparison": df_records(df)})


@app.route("/api/opportunity_cost", methods=["POST"])
def opportunity_cost():
    err = _require_loaded()
    if err:
        return err
    data = request.get_json(force=True)
    names = data.get("players", [])
    df = engine.opportunity_cost(names)
    return jsonify({"opportunity_cost": df_records(df)})


@app.route("/api/strategy_report", methods=["POST"])
def strategy_report():
    err = _require_loaded()
    if err:
        return err
    data = request.get_json(force=True) or {}
    names = data.get("players")
    result = engine.draft_strategy_report(names)
    return jsonify(clean(result))


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
