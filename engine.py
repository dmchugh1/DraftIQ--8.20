"""
DraftIQ Engine
==============
Consolidated, de-duplicated version of the DraftIQ draft assistant.

Source consolidation notes (kept for reference):
- calculate_position_urgency: 6 identical copies collapsed to 1 (was v0.3.10)
- calculate_draft_urgency: kept v0.3.9 (most developed), dropped earlier plain version
- get_draft_recommendations: kept v0.3.8 (most developed), dropped plain + v0.3.7
- what_if_i_wait: kept the non-simulation version. The "simulation-based" version
  called simulated_next_pick_probability(), which was never defined anywhere in the
  source material, so it would crash. If you want real simulation-backed wait
  probabilities, build_simulation_scores()/simulate_draft_path_fast() below can be
  wired up to power that later.
- Fixed: availability_score() referenced an undefined `drafted_players` global — this
  function was unused elsewhere (superseded by next_pick_probability) and dropped.
- Fixed: tier_table used to be computed once at import time and never refreshed, so
  tiers silently went stale as the draft progressed. It's now recomputed on demand.
- Fixed: get_roster_needs() only ever returns "HIGH"/"OK", never "MEDIUM" — the
  "MEDIUM" branches in calculate_draftiq_score/compare_players were dead code and
  are left out.
- Fixed: player-team indexing. MY_TEAM/settings["user_slot"] are 1-indexed team
  numbers; draft_state now stores a consistent 0-indexed "user_team_index".
- All state (league, draft_state, players_df) now lives on a DraftIQEngine instance
  instead of module-level globals, so it behaves correctly under a web server.
- All display()/print()-based output has been converted to return values (dicts /
  list-of-dicts) so the Flask layer can render them.
"""

import random
from collections import Counter

import numpy as np
import pandas as pd


class DraftIQEngine:

    def __init__(self):
        self.num_teams = 12
        self.my_team = 12          # 1-indexed team number
        self.roster_size = 16
        self.rounds = 16
        self.scoring = "0.5 PPR"
        self.starters = {
            "QB": 1, "RB": 2, "WR": 2, "TE": 1, "FLEX": 1, "DEF": 1
        }
        self.bench = 9

        # 1 (conservative) - 10 (aggressive), 5 = baseline/default behavior.
        # Scales the urgency-related portions of scoring (calculate_draft_urgency,
        # calculate_position_urgency), which in turn drives how readily the
        # recommendation labels ("Must Draft" / "Draft Soon" vs "Can Wait")
        # trigger. It does NOT change base DraftIQ score (rank/ADP/tier).
        self.aggressiveness = 5

        self.players_df = pd.DataFrame()
        self.league = []
        # Cached Monte Carlo result from simulate_availability_probabilities(),
        # keyed by pick number so it's only recomputed when a real pick
        # happens (not on every recommendation/decision-engine call).
        self._availability_sim_cache = None
        # Player names the user has explicitly flagged. Mutually exclusive -
        # adding a name to one removes it from the other (see
        # add_target_player/add_avoid_player).
        self.target_players = set()
        self.avoid_players = set()
        self.draft_order = []
        self.draft_state = {
            "pick": 1,
            "round": 1,
            "history": [],
            "drafted_players": [],
        }
        self.loaded = False

        # Tiers are computed once, from the FULL original player pool, and
        # cached here - a player's tier is a fixed property of their
        # original ranking, not something that should shift as other
        # players get drafted (see _compute_full_tier_table).
        self._full_tier_table = None

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def load_players(self, filepath_or_buffer):
        df = pd.read_csv(filepath_or_buffer)

        df.columns = (
            df.columns.str.strip().str.lower().str.replace(" ", "_")
        )

        df = df.rename(columns={
            "player": "name",
            "player_name": "name",
            "pos": "position",
        })

        for col in ["team", "adp", "rank"]:
            if col not in df.columns:
                df[col] = None

        df["position"] = df["position"].astype(str).str.upper().str.strip()
        df["rank"] = pd.to_numeric(df["rank"], errors="coerce")
        df["adp"] = pd.to_numeric(df["adp"], errors="coerce")
        # Some real rank sheets use 0 (or negative) as a sentinel for "no
        # real ADP data" rather than leaving the cell blank. Treated as a
        # literal ADP of 0, that player's ADP score formula computes as if
        # they were the best-possible pick (score is highest at adp=1,
        # near-max at adp=0) - exactly backwards for what's meant to signal
        # "unranked". Treat non-positive ADP as missing, same as blank.
        df.loc[df["adp"] <= 0, "adp"] = pd.NA

        # Players with no rank at all can't be scored - drop them rather than
        # letting them silently NaN-poison sorts/scores downstream.
        df = df.dropna(subset=["rank"]).reset_index(drop=True)

        self._set_players_df(df)

    def _set_players_df(self, df):
        self.players_df = df
        self._full_tier_table = None  # invalidate - recomputed lazily from the new pool
        self._availability_sim_cache = None
        self.target_players = set()
        self.avoid_players = set()
        self.league = self._create_league()
        self.draft_order = self._snake_order()
        self.draft_state = {
            "pick": 1,
            "round": 1,
            "history": [],
            "drafted_players": [],
            "user_team_index": self.my_team - 1,
        }
        self.loaded = True

    def _create_league(self):
        return [{"team": f"Team {i}", "roster": []} for i in range(1, self.num_teams + 1)]

    def _snake_order(self):
        order = []
        for r in range(self.rounds):
            picks = list(range(1, self.num_teams + 1))
            if r % 2 == 1:
                picks.reverse()
            order.extend(picks)
        return order

    def reset_draft(self):
        for team in self.league:
            team["roster"] = []
        self.draft_state = {
            "pick": 1,
            "round": 1,
            "history": [],
            "drafted_players": [],
            "user_team_index": self.my_team - 1,
        }
        self._availability_sim_cache = None
        self.target_players = set()
        self.avoid_players = set()

    def is_draft_complete(self):
        """True once every roster slot across the whole league (num_teams *
        rounds) has been filled. Based on actual picks logged rather than
        solely the round counter, so it stays correct even if the player
        pool runs out before every slot is technically filled."""
        total_slots = self.num_teams * self.rounds
        return len(self.draft_state["history"]) >= total_slots

    def set_my_team(self, team_number):
        """Change which draft slot is "yours" WITHOUT resetting the draft
        in progress - only num_teams/rounds changes require a full reset,
        since those change the league's structure."""
        self.my_team = int(team_number)
        if self.draft_state:
            self.draft_state["user_team_index"] = self.my_team - 1

    def set_starters(self, starters_dict):
        """Manually set starting-lineup requirements, e.g.
        {"QB":1,"RB":2,"WR":2,"TE":1,"FLEX":1,"DEF":1}. Positions with a
        count of 0 or less are dropped. Roster needs are computed live from
        this dict on every call, so no draft reset is required."""
        cleaned = {}
        for pos, count in (starters_dict or {}).items():
            try:
                count = int(count)
            except (TypeError, ValueError):
                continue
            if count > 0:
                cleaned[str(pos).upper().strip()] = count
        if cleaned:
            self.starters = cleaned

    def set_aggressiveness(self, level):
        self.aggressiveness = max(1, min(10, int(level)))

    def _aggressiveness_factor(self):
        """Maps the 1-10 aggressiveness dial to a multiplier applied to
        urgency scoring. 5 = 1.0x (matches original tuned behavior).
        1 = 0.3x (conservative - rarely flags urgency). 10 = 2.0x
        (aggressive - flags urgency much more readily)."""
        factor = self.aggressiveness / 5.0
        return max(0.3, min(2.0, factor))

    # ------------------------------------------------------------------
    # Target / avoid players
    # ------------------------------------------------------------------
    def add_target_player(self, name):
        """Flag a player as a target. Mutually exclusive with avoid - being
        both would be contradictory, so this clears any existing avoid flag
        on the same player."""
        self.avoid_players.discard(name)
        self.target_players.add(name)

    def remove_target_player(self, name):
        self.target_players.discard(name)

    def add_avoid_player(self, name):
        """Flag a player to avoid. Mutually exclusive with target."""
        self.target_players.discard(name)
        self.avoid_players.add(name)

    def remove_avoid_player(self, name):
        self.avoid_players.discard(name)

    def get_target_alerts(self, at_risk_threshold=35):
        """Returns one entry per currently-available target player, using
        the same Monte Carlo availability simulation that drives
        next_pick_probability elsewhere - so "should I grab my target now"
        reflects real simulated bot behavior and roster needs, not a
        guess. at_risk=True when a target's odds of surviving to your next
        pick fall at or below at_risk_threshold, meaning waiting risks
        losing them.

        Sorted most-at-risk first, so the most time-sensitive target
        surfaces at the top."""
        if not self.target_players:
            return []
        available = self.get_available_players()
        alerts = []
        for _, player in available[available["name"].isin(self.target_players)].iterrows():
            probability = self.next_pick_probability(player)
            alerts.append({
                "name": player["name"],
                "position": player["position"],
                "adp": player["adp"],
                "probability_available": probability,
                "at_risk": probability <= at_risk_threshold,
            })
        alerts.sort(key=lambda a: a["probability_available"])
        return alerts

    # ------------------------------------------------------------------
    # Draft state
    # ------------------------------------------------------------------

    def get_team_for_pick(self, pick):
        num_teams = len(self.league)
        round_num = ((pick - 1) // num_teams) + 1
        if round_num % 2 == 1:
            team_index = (pick - 1) % num_teams
        else:
            team_index = num_teams - 1 - ((pick - 1) % num_teams)
        return team_index

    def get_current_team_index(self):
        return self.get_team_for_pick(self.draft_state["pick"])

    def get_current_team_number(self):
        return self.get_current_team_index() + 1

    def make_pick(self, player_name):
        team_index = self.get_team_for_pick(self.draft_state["pick"])
        team = self.league[team_index]
        team["roster"].append(player_name)

        self.draft_state["history"].append({
            "pick": self.draft_state["pick"],
            "round": self.draft_state["round"],
            "team": team["team"],
            "player": player_name,
        })
        self.draft_state["drafted_players"].append(player_name)
        self.draft_state["pick"] += 1
        self.draft_state["round"] = (
            ((self.draft_state["pick"] - 1) // len(self.league)) + 1
        )

    def get_team_roster(self, team_number):
        return [
            pick["player"]
            for pick in self.draft_state["history"]
            if pick["team"] == f"Team {team_number}"
        ]

    def _pick_label(self, round_number, pick_number):
        """Formats an overall pick as round.pick-within-round, e.g. the
        13th overall pick in a 12-team league is round 2, pick 1 -> "2.01"."""
        pick_in_round = pick_number - (round_number - 1) * len(self.league)
        return f"{round_number}.{pick_in_round:02d}"

    def get_roster_slots(self, team_number=None):
        """Assigns a team's drafted players into starter slots (in draft
        order - earliest pick at a position fills that position's slot
        first) plus a bench for everyone else. Slot counts come from
        self.starters, so this reflects whatever league settings are
        currently configured, not a
        hardcoded roster shape. Empty slots return player=None so the UI
        can render them as open. Each filled slot/bench entry also
        includes when that player was actually drafted (pick_label, e.g.
        "1.01"), pulled from real draft history rather than re-derived."""
        if team_number is None:
            team_number = self.my_team
        roster_names = self.get_team_roster(team_number)
        position_by_name = dict(zip(self.players_df["name"], self.players_df["position"]))
        pick_label_by_name = {
            h["player"]: self._pick_label(h["round"], h["pick"])
            for h in self.draft_state["history"]
        }
        remaining = [(n, position_by_name.get(n)) for n in roster_names]

        FLEX_ELIGIBLE = {"RB", "WR", "TE"}
        SLOT_ORDER = ["QB", "RB", "WR", "TE", "FLEX", "DEF"]

        def take(match_positions):
            idx = next((i for i, (_, p) in enumerate(remaining) if p in match_positions), None)
            return remaining.pop(idx) if idx is not None else (None, None)

        slots = []
        for pos in SLOT_ORDER:
            count = self.starters.get(pos, 0)
            for i in range(count):
                label = f"{pos}{i + 1}" if count > 1 else pos
                if pos == "FLEX":
                    name, filled_pos = take(FLEX_ELIGIBLE)
                    slots.append({
                        "slot": label, "position": filled_pos or "FLEX", "player": name,
                        "pick_label": pick_label_by_name.get(name),
                    })
                else:
                    name, filled_pos = take({pos})
                    slots.append({
                        "slot": label, "position": pos, "player": name,
                        "pick_label": pick_label_by_name.get(name),
                    })

        # Any starter slots configured beyond the standard set (unlikely,
        # but self.starters is user/league-configurable) still get shown.
        for pos, count in self.starters.items():
            if pos in SLOT_ORDER:
                continue
            for i in range(count):
                label = f"{pos}{i + 1}" if count > 1 else pos
                name, filled_pos = take({pos})
                slots.append({
                    "slot": label, "position": pos, "player": name,
                    "pick_label": pick_label_by_name.get(name),
                })

        bench = [
            {"player": n, "position": p, "pick_label": pick_label_by_name.get(n)}
            for n, p in remaining
        ]
        return {"starters": slots, "bench": bench}

    def get_draft_board(self):
        """Classic snake-draft board: rows are rounds, columns are teams in
        FIXED order (Team 1..N), matching how real draft boards look - the
        pick sequence snakes back and forth across columns rather than the
        columns themselves reordering."""
        total_picks = self.num_teams * self.rounds
        history_by_pick = {h["pick"]: h for h in self.draft_state["history"]}
        current_pick = self.draft_state["pick"]

        grid = [[None] * self.num_teams for _ in range(self.rounds)]

        for pick_num in range(1, total_picks + 1):
            round_idx = (pick_num - 1) // self.num_teams
            team_idx = self.get_team_for_pick(pick_num)

            cell = {
                "pick": pick_num,
                "team": team_idx + 1,
                "is_current": pick_num == current_pick,
                "is_my_team": (team_idx + 1) == self.my_team,
                "player": None,
                "position": None,
            }

            entry = history_by_pick.get(pick_num)
            if entry:
                player_row = self.players_df[self.players_df["name"] == entry["player"]]
                cell["player"] = entry["player"]
                if len(player_row) > 0:
                    cell["position"] = player_row.iloc[0]["position"]

            grid[round_idx][team_idx] = cell

        return {
            "rounds": self.rounds,
            "num_teams": self.num_teams,
            "my_team": self.my_team,
            "grid": grid,
        }

    def get_available_players(self):
        available = self.players_df[
            ~self.players_df["name"].isin(self.draft_state["drafted_players"])
        ]
        # Only show positions your league actually starts somewhere (FLEX
        # isn't a real position, so it doesn't count as "used" on its
        # own). A position with zero starter slots - e.g. K in a league
        # that doesn't roster kickers - never appears here, which in turn
        # keeps it out of recommendations, tiers, and the simulation,
        # since they all read from this same list.
        relevant_positions = set(self.starters.keys()) - {"FLEX"}
        if relevant_positions:
            available = available[available["position"].isin(relevant_positions)]
        return available

    def draft_player(self, player_name):
        if player_name not in self.players_df["name"].values:
            return {"ok": False, "message": f"{player_name} not found"}
        if player_name in self.draft_state["drafted_players"]:
            return {"ok": False, "message": f"{player_name} already drafted"}

        # Capture who is actually on the clock BEFORE make_pick() advances
        # draft_state["pick"] - calling get_current_team_number() after the
        # pick returns the *next* team, not the one that just picked.
        drafting_team_number = self.get_current_team_number()
        drafting_team_name = self.league[drafting_team_number - 1]["team"]
        pick_number = self.draft_state["pick"]

        self.make_pick(player_name)

        return {
            "ok": True,
            "message": f"Drafted {player_name}",
            "player": player_name,
            "pick": pick_number,
            "team_number": drafting_team_number,
            "team_name": drafting_team_name,
        }

    def undo_last_pick(self):
        """Removes the most recent pick from history, the drafted-players
        list, and the drafting team's roster, and rewinds pick/round
        counters. Used to correct manual-entry mistakes (e.g. wrong player
        typed in during a live draft)."""
        history = self.draft_state["history"]
        if len(history) == 0:
            return {"ok": False, "message": "No picks to undo."}

        last = history.pop()
        player_name = last["player"]
        team_name = last["team"]

        for team in self.league:
            if team["team"] == team_name and player_name in team["roster"]:
                team["roster"].remove(player_name)
                break

        if player_name in self.draft_state["drafted_players"]:
            self.draft_state["drafted_players"].remove(player_name)

        self.draft_state["pick"] = last["pick"]
        self.draft_state["round"] = last["round"]

        return {"ok": True, "undone": player_name, "team": team_name, "pick": last["pick"]}

    # ------------------------------------------------------------------
    # Tiers (recomputed on demand - NOT cached at load time)
    # ------------------------------------------------------------------

    def _compute_tier_table(self, df):
        """Raw tier computation over whatever player dataframe is passed in.
        Groups each position into tiers based on gaps in overall rank.

        The gap threshold that defines "a new tier starts here" is
        calibrated to each position's own typical rank spacing rather than
        a fixed constant. A flat threshold (e.g. "gap >= 5") only works if
        `rank` is itself position-specific (RB1, RB2, RB3...). Real rank
        sheets use an OVERALL rank, where other
        positions' players are naturally interleaved between same-position
        neighbors. That inflates the typical gap between, say, RB5 and RB6
        to well beyond 5 even when they're genuinely similar quality, which
        used to fracture a position into dozens of 1-2-player "tiers" and
        spam alerts. Scaling the threshold to each position's own median
        gap keeps tiers meaningful regardless of which rank convention the
        input data uses.

        A single global gap threshold still isn't enough on real data,
        though: the top of a position is usually tightly and consistently
        ranked (small, regular gaps), while the deep/replacement-level end
        is noisier (rankers have much less conviction distinguishing RB40
        from RB45, so raw gaps there are larger and more erratic even
        though there's no real talent cliff). That noise was still
        splitting the deep end of each position into a lot of spurious
        1-2-player tiers - which, among other things, made those isolated
        players falsely look like they were facing a scarcity "cliff" and
        get an unwarranted urgency boost. After gap-based tiering, any tier
        smaller than MIN_TIER_SIZE gets merged into its neighbor so a
        "tier" only exists when there are enough players in it to be a
        meaningful group, not noise.
        """
        MIN_TIER_SIZE = 3

        all_tiers = []

        for position in df["position"].unique():
            pos_df = df[df["position"] == position].sort_values("rank").reset_index(drop=True).copy()
            n = len(pos_df)
            if n == 0:
                continue

            ranks = pos_df["rank"].tolist()
            gaps = [ranks[i] - ranks[i - 1] for i in range(1, n)]

            if gaps:
                sorted_gaps = sorted(gaps)
                median_gap = sorted_gaps[len(sorted_gaps) // 2]
                gap_threshold = max(3, median_gap * 2.5)
            else:
                gap_threshold = 5

            tier = 1
            tiers = [1]
            for i in range(1, n):
                if (ranks[i] - ranks[i - 1]) >= gap_threshold:
                    tier += 1
                tiers.append(tier)

            # Merge any tier smaller than MIN_TIER_SIZE into its nearest
            # actually-existing neighboring tier (by position in the sorted
            # tier list, not by literal label+1/-1, which may not exist and
            # would otherwise "merge" into a brand new empty tier forever).
            # Repeat until stable, then relabel tiers 1..N with no gaps.
            if n >= MIN_TIER_SIZE:
                changed = True
                safety_counter = 0
                while changed and safety_counter < n:
                    safety_counter += 1
                    changed = False
                    counts = {}
                    for t in tiers:
                        counts[t] = counts.get(t, 0) + 1
                    if len(counts) <= 1:
                        break
                    sorted_tiers = sorted(counts.keys())
                    for t in sorted_tiers:
                        if counts[t] < MIN_TIER_SIZE:
                            idx = sorted_tiers.index(t)
                            if idx > 0:
                                target = sorted_tiers[idx - 1]
                            else:
                                target = sorted_tiers[idx + 1]
                            tiers = [target if x == t else x for x in tiers]
                            changed = True
                            break

            # Relabel to a clean 1..N sequence (merging can leave gaps,
            # e.g. tiers [1,1,3,3] after tier 2 was merged away).
            unique_sorted = sorted(set(tiers))
            relabel = {old: new for new, old in enumerate(unique_sorted, start=1)}
            tiers = [relabel[t] for t in tiers]

            pos_df["Tier"] = tiers
            all_tiers.append(pos_df[["name", "position", "Tier"]])

        if not all_tiers:
            return pd.DataFrame(columns=["name", "position", "Tier"])
        return pd.concat(all_tiers, ignore_index=True)

    def get_position_tier_table(self):
        """Returns tiers for currently-available players, using a FIXED
        tier assignment computed once from the full original player pool
        (cached in self._full_tier_table) rather than recomputed against
        whatever's left. A player's tier is meant to be an intrinsic
        property of their original ranking - the same way real fantasy
        tier sheets work - not something that inflates as better players
        get drafted away. Recomputing tiers from only the shrinking
        available pool used to reclassify plain Tier-3 players as "Tier 1"
        once everyone above them was gone, which both looked wrong on the
        Recommendations page and fed bogus "Elite Tier" bonuses into
        scoring deep into the draft.
        """
        if self._full_tier_table is None:
            self._full_tier_table = self._compute_tier_table(self.players_df)

        available_names = set(self.get_available_players()["name"])
        return self._full_tier_table[self._full_tier_table["name"].isin(available_names)].reset_index(drop=True)

    def player_tier(self, player_name, tier_table=None):
        if tier_table is None:
            tier_table = self.get_position_tier_table()
        result = tier_table[tier_table["name"] == player_name]
        if len(result) == 0:
            return None
        return int(result.iloc[0]["Tier"])

    def get_tier_counts(self):
        available = self.get_available_players().copy()
        tier_table = self.get_position_tier_table()
        available["tier"] = available["name"].apply(
            lambda n: self.player_tier(n, tier_table)
        )
        return (
            available.groupby(["position", "tier"])
            .size()
            .reset_index(name="remaining")
        )

    # ------------------------------------------------------------------
    # Roster intelligence
    # ------------------------------------------------------------------

    def get_my_team(self):
        return self.league[self.my_team - 1]

    def get_position_counts(self):
        roster = self.get_my_team()["roster"]
        counts = {"QB": 0, "RB": 0, "WR": 0, "TE": 0, "DEF": 0}
        for player_name in roster:
            player = self.players_df[self.players_df["name"] == player_name]
            if len(player) > 0:
                pos = player.iloc[0]["position"]
                if pos in counts:
                    counts[pos] += 1
        return counts

    def get_roster_needs(self):
        counts = self.get_position_counts()
        needs = {}
        for pos, required in self.starters.items():
            if pos == "FLEX":
                continue
            needs[pos] = "HIGH" if counts.get(pos, 0) < required else "OK"
        return needs

    # ------------------------------------------------------------------
    # Draft intelligence
    # ------------------------------------------------------------------

    def players_until_next_pick(self):
        current_pick = self.draft_state["pick"]
        user_team_index = self.draft_state.get("user_team_index", self.my_team - 1)
        picks = current_pick + 1
        while True:
            if self.get_team_for_pick(picks) == user_team_index:
                return picks - current_pick
            picks += 1

    def next_pick_probability(self, player):
        name = player["name"]
        picks_away = self.players_until_next_pick()
        current_pick = self.draft_state["pick"]

        # Primary source: a Monte Carlo simulation of the actual picks
        # between now and the user's next turn (see
        # simulate_availability_probabilities), reflecting real bot
        # behavior and roster needs rather than a static ADP curve.
        # Cached per pick number (shared with detect_position_run and
        # detect_tier_cliff) so it's computed once per real draft pick,
        # not once per call.
        cache = self._ensure_availability_sim_cache()
        simulated = cache["player_probability"].get(name)
        if simulated is not None:
            return simulated

        # Fallback: ADP-based logistic curve, used when the simulation
        # doesn't cover this player (missing ADP, picks_away <= 0, or any
        # other edge case).
        next_pick = current_pick + picks_away
        adp = player["adp"]
        if pd.isna(adp):
            return 50
        gap = adp - next_pick
        # Scale the gap relative to picks_away, not a fixed ADP-point slope:
        # an ADP gap of a given size means something very different
        # depending on how many actual picks happen before your next turn
        # (1 vs 20+). Feed that scaled gap through a logistic curve rather
        # than a linear ramp + hard clip: a hard clip makes every gap past
        # a small threshold land on the exact same floor/ceiling value -
        # harmless for one player, but when picks_away is small (e.g. "my
        # very next pick"), that threshold shrinks to just 1-2 ADP points,
        # so several clearly different players (e.g. ADP 9 vs 11 vs 12)
        # can all tie at the identical floor. The logistic curve keeps the
        # same center (50 at gap=0) and the same ~5-95 practical range, but
        # never fully flattens, so real ADP differences stay visible even
        # among players who are all "very likely gone" or "very likely
        # available".
        x = gap / max(picks_away, 1)
        probability = 5 + 90 / (1 + np.exp(-x))
        return round(probability)

    def detect_position_run(self, position, recent_picks=10, recent_position_counts=None):
        """Prefers the forward-looking Monte Carlo simulation (expected #
        picks at this position before the user's next turn, from
        simulate_availability_probabilities) over the backward-looking
        recent-picks count. Falls back to history when the simulation
        doesn't cover this position (e.g. before any real picks exist).

        recent_position_counts (an optional precomputed {position: count}
        dict) lets a caller looping over many players/positions skip
        recomputing this from scratch every time - its result only depends
        on draft history, not on which player is currently being scored, so
        it's identical for every player in a given scoring pass. Left
        uncached, this was doing a full dataframe filter per recently
        drafted player, called separately from three different scoring
        functions per player - the single largest remaining bottleneck on
        a large player pool once any picks have actually been made."""
        cache = self._ensure_availability_sim_cache()
        rate = cache.get("position_run_rate", {})
        if position in rate:
            expected_count = rate[position]
            return expected_count >= 4, round(expected_count)

        if recent_position_counts is not None:
            count = recent_position_counts.get(position, 0)
            return count >= 4, count

        drafted = self.draft_state["drafted_players"]
        if len(drafted) == 0:
            return False, 0
        recent = drafted[-recent_picks:]
        count = 0
        for player_name in recent:
            player = self.players_df[self.players_df["name"] == player_name]
            if len(player) > 0 and player.iloc[0]["position"] == position:
                count += 1
        return count >= 4, count

    def _recent_position_counts(self, recent_picks=10):
        """Computes how many of the last N drafted picks were each
        position, in one pass - used to avoid recomputing this per player
        in get_draft_recommendations/draftiq_decision_engine."""
        drafted = self.draft_state["drafted_players"]
        if len(drafted) == 0:
            return {}
        recent = drafted[-recent_picks:]
        name_to_pos = dict(zip(self.players_df["name"], self.players_df["position"]))
        counts = {}
        for name in recent:
            pos = name_to_pos.get(name)
            if pos:
                counts[pos] = counts.get(pos, 0) + 1
        return counts

    def get_position_run_alerts(self, window=10, threshold=4):
        history = self.draft_state["history"]
        if len(history) == 0:
            return []
        recent = history[-window:]
        positions = []
        for pick in recent:
            player = self.players_df[self.players_df["name"] == pick["player"]]
            if len(player) > 0:
                positions.append(player.iloc[0]["position"])
        counts = Counter(positions)
        return [
            f"\U0001F525 {pos} RUN ({total} in last {window} picks)"
            for pos, total in counts.items()
            if total >= threshold
        ]

    def detect_tier_cliff(self, player, tier_table=None, tier_lookup=None, tier_position_counts=None):
        """Prefers the forward-looking Monte Carlo simulation (expected #
        players remaining in this player's tier/position group at the
        user's next turn, from simulate_availability_probabilities) over
        a static current-snapshot count. Falls back to the static count
        when the simulation doesn't cover this group.

        tier_lookup ({name: tier}) and tier_position_counts
        ({(position, tier): count}) are optional precomputed caches for
        that static fallback - pass them when calling this in a per-player
        loop to avoid a pandas boolean-mask filter on every single call,
        which dominated runtime on a large player pool even with
        tier_table itself cached."""
        tier = None
        if tier_lookup is not None:
            tier = tier_lookup.get(player["name"])
        if tier is None:
            if tier_table is None:
                tier_table = self.get_position_tier_table()
            player_info = tier_table[tier_table["name"] == player["name"]]
            if len(player_info) > 0:
                tier = player_info.iloc[0]["Tier"]
        if tier is None:
            return False, 0

        position = player["position"]
        cache = self._ensure_availability_sim_cache()
        cliff_map = cache.get("tier_cliff_remaining", {})
        key = (position, tier)
        if key in cliff_map:
            expected_remaining = cliff_map[key]
            return expected_remaining <= 3, round(expected_remaining)

        if tier_position_counts is not None:
            remaining = tier_position_counts.get(key, 0)
            return remaining <= 3, remaining

        if tier_table is None:
            tier_table = self.get_position_tier_table()
        available = self.get_available_players()
        tier_players = tier_table[tier_table["Tier"] == tier]["name"]
        remaining = len(
            available[
                (available["position"] == position)
                & (available["name"].isin(tier_players))
            ]
        )
        return remaining <= 3, remaining

    def get_draft_alerts(self):
        """Flags positions where the CURRENT best available tier is at risk
        of running out before your next pick. Only considers each needed
        position's lowest (best) remaining tier - not every tier down the
        list, which used to fire an alert for every tier of every needed
        position simultaneously (e.g. "RB Tier 18: 1 remaining" when you're
        nowhere near drafting RB18-tier players yet)."""
        tier_counts = self.get_tier_counts()
        if len(tier_counts) == 0:
            return []

        alerts = []
        picks_until = self.players_until_next_pick()
        needs = self.get_roster_needs()

        for pos in tier_counts["position"].unique():
            if needs.get(pos) != "HIGH":
                continue

            pos_tiers = tier_counts[tier_counts["position"] == pos].sort_values("tier")
            if len(pos_tiers) == 0:
                continue

            current = pos_tiers.iloc[0]  # lowest/best tier still available
            tier, remaining = current["tier"], current["remaining"]
            if remaining <= picks_until:
                alerts.append(
                    f"\u26A0\uFE0F {pos} Tier {tier}: {remaining} remaining before your next pick"
                )

        return alerts

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    def calculate_draftiq_score(self, player, tier_table=None, needs=None,
                                  tier_lookup=None, tier_position_counts=None,
                                  recent_position_counts=None):
        if needs is None:
            needs = self.get_roster_needs()
        score = 0
        reasons = []

        rank = player["rank"]
        score += max(0, 30 * (1 - (rank - 1) / 199))

        # ADP intentionally has no direct value component here anymore.
        # Rank (plus Tier, which is itself just Rank bucketed for
        # scarcity) is the one signal answering "how good is this
        # player" - ADP's job is timing and market disagreement, not a
        # second vote on quality. It still drives next_pick_probability
        # below (survival risk) and the value_gap label logic in
        # get_draft_recommendations (steal/reach framing), which is where
        # ADP disagreeing with Rank is actually useful information.

        if needs.get(player["position"]) == "HIGH":
            score += 15
            reasons.append("Position Need")

        tier = tier_lookup.get(player["name"]) if tier_lookup is not None else self.player_tier(player["name"], tier_table)
        if tier is not None:
            score += max(0, 10 - ((tier - 1) * 2))
            if tier == 1:
                reasons.append("Elite Tier")

        probability = self.next_pick_probability(player)
        reasons.append(f"{probability}% chance available next pick")
        if probability < 20:
            score += 10
        elif probability < 40:
            score += 7
        elif probability < 60:
            score += 4

        # Position run and tier cliff intentionally have no bonus here.
        # They're risk signals ("is this player about to become
        # unavailable"), not base value - their home is
        # calculate_draft_urgency (and, for position run, also
        # calculate_position_urgency's position-level context). Adding
        # them here too was double-counting the same signal into two
        # heavily-weighted buckets of DraftIQ Score (draftiq_decision_engine) at once.

        return round(min(100, score), 1), reasons

    def calculate_draft_urgency(self, player, needs=None, tier_table=None,
                                  tier_lookup=None, tier_position_counts=None,
                                  recent_position_counts=None):
        """v0.3.9 - kept as the most developed version."""
        urgency = 0
        reasons = []
        current_pick = self.draft_state["pick"]

        probability = self.next_pick_probability(player)
        urgency += round((100 - probability) * 0.65)
        if probability <= 10:
            reasons.append("Extremely unlikely to return")
        elif probability <= 25:
            reasons.append("Very unlikely to return")
        elif probability <= 40:
            reasons.append("Unlikely to return")
        elif probability <= 60:
            reasons.append("Moderate return risk")

        # Tier's primary home is calculate_draftiq_score() (base player
        # value). It used to also be scored again here (+20 Tier 1, +10
        # Tier 2) and a third time as a flat bonus in
        # draftiq_decision_engine() - stacking the same "this player is
        # elite" signal into three separate weighted buckets that were
        # each meant to represent something different. Removed here;
        # calculate_draft_urgency now only reflects genuinely distinct
        # risk signals (return risk, tier-cliff scarcity, position run,
        # roster need), not the player's own tier a second time.

        # NOTE: an "ADP value" bonus based on (adp - current_pick) used to
        # live here. Removed: since current_pick is fixed across every
        # player being scored in a single pass, that gap is monotonic in
        # raw ADP - it always rewarded whichever player had the WORSE
        # (higher/later) ADP, for any two players compared, regardless of
        # which one was actually the better pick. It also had a bucket
        # discontinuity (adp_gap of 6-7 fell through to zero bonus). The
        # DraftIQ Score's own ADP component below already correctly
        # rewards lower ADP; this bonus was redundant AND backwards.

        cliff, remaining = self.detect_tier_cliff(player, tier_table, tier_lookup, tier_position_counts)
        if cliff:
            urgency += max(0, (6 - remaining) * 4)
            reasons.append(f"Tier cliff ({remaining} left)")

        run, count = self.detect_position_run(player["position"], recent_position_counts=recent_position_counts)
        if run:
            urgency += min(12, count * 2)
            reasons.append(f"{player['position']} run")

        # Roster need's primary home is calculate_draftiq_score() (base
        # player value) - it used to be re-added here as a second flat
        # +10 bonus on the exact same needs.get(position)=="HIGH"
        # condition, then a third time as a direct bonus in
        # draftiq_decision_engine() and compare_players(). Removed here;
        # calculate_position_urgency's own round-aware roster-need logic
        # is untouched, since that's a genuinely different (more nuanced)
        # signal, not a repeat of this flat bonus.

        urgency = urgency * self._aggressiveness_factor()
        # Raw, uncapped - callers use this directly in any further weighted
        # computation (decision_score, strategy_score, opportunity_score)
        # and only round/cap with min(100, round(...)) when displaying it
        # as the "Player Urgency" column. Capping here would flatten
        # multiple genuinely-different-urgency players to the same value
        # before those weighted sums ever see them, not just at display.
        return urgency, reasons

    def calculate_position_urgency(self, position, tier_table=None, position_counts=None,
                                     position_remaining_counts=None, tier_one_remaining_by_position=None,
                                     recent_position_counts=None):
        """v0.3.10 - deduplicated from 6 identical copies.

        The optional cache params let a caller looping over every available
        player (e.g. draftiq_decision_engine) avoid recomputing
        position-level aggregates (remaining count, tier-1 remaining count)
        from scratch on every single call - which otherwise turns an O(n)
        loop into O(n^2) work on a large player pool.
        """
        urgency = 0
        reasons = []
        current_pick = self.draft_state["pick"]
        num_teams = len(self.league)
        round_number = ((current_pick - 1) // num_teams) + 1

        counts = position_counts if position_counts is not None else self.get_position_counts()
        current_count = counts.get(position, 0)
        required = self.starters.get(position, 0)

        if current_count < required:
            base_need = {"DEF": 0, "QB": 5, "TE": 10, "RB": 15, "WR": 15}.get(position, 5)
            urgency += base_need
            reasons.append(f"No {position} drafted" if current_count == 0 else f"{position} starter still needed")

        if round_number <= 3:
            if position in ["RB", "WR"]:
                urgency += 15
                reasons.append("Early-round priority")
            elif position == "TE":
                urgency += 5
        elif round_number <= 6:
            if position in ["RB", "WR"]:
                urgency += 10
            elif position in ["TE", "QB"]:
                urgency += 8
                reasons.append("Middle-round priority")
        elif round_number <= 10:
            if position in ["TE", "QB"]:
                urgency += 15
                reasons.append("Middle/late-round priority")
            elif position in ["RB", "WR"]:
                urgency += 5
        else:
            if current_count < required:
                urgency += 25
                reasons.append("Late-round starting requirement")

        if position_remaining_counts is not None:
            remaining_count = position_remaining_counts.get(position, 0)
        else:
            available = self.get_available_players()
            remaining_count = len(available[available["position"] == position])

        if tier_one_remaining_by_position is not None:
            tier_one_remaining = tier_one_remaining_by_position.get(position, 0)
        else:
            if tier_table is None:
                tier_table = self.get_position_tier_table()
            pos_tier_table = tier_table[tier_table["position"] == position]
            tier_one_remaining = int((pos_tier_table["Tier"] == 1).sum())

        if round_number <= 6:
            if tier_one_remaining == 0:
                urgency += 5
                reasons.append(f"No Tier 1 {position}s remain")
            elif tier_one_remaining <= 2:
                urgency += 8
                reasons.append(f"Only {tier_one_remaining} Tier 1 {position}s remain")
        else:
            if tier_one_remaining == 0:
                urgency += 10
                reasons.append(f"No Tier 1 {position}s remain")
            elif tier_one_remaining <= 2:
                urgency += 12
                reasons.append(f"Only {tier_one_remaining} Tier 1 {position}s remain")

        run, run_count = self.detect_position_run(position, recent_position_counts=recent_position_counts)
        if run:
            urgency += min(10, run_count * 2)
            reasons.append(f"{position} run underway")

        if round_number >= 8:
            if remaining_count <= 5:
                urgency += 15
                reasons.append(f"Only {remaining_count} {position}s remain")
            elif remaining_count <= 10:
                urgency += 7
                reasons.append(f"{remaining_count} {position}s remain")

        urgency = urgency * self._aggressiveness_factor()

        if position == "DEF":
            urgency = min(urgency, 25)

        return min(100, round(urgency)), reasons

    # ------------------------------------------------------------------
    # Recommendations (v0.3.8 - most developed of 3 versions found)
    # ------------------------------------------------------------------

    def get_draft_recommendations(self, count=10):
        available = self.get_available_players().copy()
        if self.avoid_players:
            available = available[~available["name"].isin(self.avoid_players)]
        if len(available) == 0:
            return pd.DataFrame()

        tier_table = self.get_position_tier_table()
        needs = self.get_roster_needs()
        tier_lookup = dict(zip(tier_table["name"], tier_table["Tier"]))
        tier_position_counts = tier_table.groupby(["position", "Tier"]).size().to_dict()
        recent_position_counts = self._recent_position_counts()
        scores, reasons, tiers, urgencies, raw_urgencies = [], [], [], [], []

        for _, player in available.iterrows():
            score, reason = self.calculate_draftiq_score(player, tier_table, needs, tier_lookup, tier_position_counts, recent_position_counts)
            raw_urgency, urgency_reason = self.calculate_draft_urgency(player, needs, tier_table, tier_lookup, tier_position_counts, recent_position_counts)
            scores.append(score)
            combined_reason = reason + urgency_reason
            if player["name"] in self.target_players:
                combined_reason = ["\U0001F3AF Target"] + combined_reason
            reasons.append(", ".join(combined_reason))
            raw_urgencies.append(raw_urgency)
            urgencies.append(min(100, round(raw_urgency)))
            tiers.append(tier_lookup.get(player["name"]))

        available["DraftIQ_Score"] = scores
        available["Reason"] = reasons
        available["Tier"] = tiers
        available["Draft_Urgency"] = urgencies

        current_pick = self.draft_state["pick"]
        round_number = ((current_pick - 1) // len(self.league)) + 1

        # Rank/select by DraftIQ_Score blended with the RAW (uncapped)
        # urgency, so a player at real risk of being lost (tier cliff,
        # position run, pressing roster need) can actually surface higher
        # - not just get described as urgent after the score alone
        # already cut them, and not have that signal flattened against
        # other high-urgency players by the display cap.
        selection_score = available["DraftIQ_Score"] + pd.Series(raw_urgencies, index=available.index) * 0.3
        available = available.assign(_selection_score=selection_score) \
            .sort_values("_selection_score", ascending=False) \
            .drop(columns=["_selection_score"]) \
            .reset_index(drop=True)
        recommendations = available.head(count).copy()

        # Force-surface an at-risk target even if their score alone
        # wouldn't have made the cut - the whole point of flagging a
        # target is to not lose them to a quiet drop-off in the ranking.
        missing_targets = [
            n for n in self.target_players
            if n in available["name"].values and n not in recommendations["name"].values
        ]
        for name in missing_targets:
            row = available[available["name"] == name].iloc[0]
            probability = self.next_pick_probability(row)
            if probability <= 35:
                alert_row = row.copy()
                alert_row["Reason"] = f"\U0001F3AF Target at risk - {probability}% to survive to your next pick, " + alert_row["Reason"]
                recommendations = pd.concat([recommendations, alert_row.to_frame().T], ignore_index=True)

        labels = []
        for i, row in recommendations.iterrows():
            score, urgency, adp, tier = row["DraftIQ_Score"], row["Draft_Urgency"], row["adp"], row["Tier"]
            value_gap = 0 if pd.isna(adp) else (adp - current_pick)

            try:
                position_need = self.get_roster_needs().get(row["position"]) == "HIGH"
            except Exception:
                position_need = False

            tier_one = (tier == 1)
            high_urgency = (urgency >= 60)
            extreme_urgency = (urgency >= 75)
            strong_score = (score >= 85)
            elite_score = (score >= 95)
            significant_value = (value_gap >= 15)
            major_value = (value_gap >= 30)

            must_draft = (
                (tier_one and position_need and high_urgency and strong_score)
                or (tier_one and extreme_urgency and strong_score)
                or (significant_value and high_urgency and strong_score)
                or (round_number >= 10 and position_need and extreme_urgency and strong_score)
                or (major_value and elite_score)
            )

            if row["name"] in self.target_players and "Target at risk" in str(row.get("Reason", "")):
                labels.append("\U0001F3AF TARGET ALERT")
            elif must_draft:
                labels.append("\U0001F525 Must Draft")
            elif i == 0 and (high_urgency or tier_one or position_need):
                labels.append("\u2705 Best Pick")
            elif elite_score and high_urgency:
                labels.append("\U0001F44D Great Value")
            elif extreme_urgency:
                labels.append("\u26A0\uFE0F Draft Soon")
            elif significant_value and strong_score:
                labels.append("\U0001F44D Great Value")
            elif score >= 90 or i <= 5:
                labels.append("\U0001F4C8 Consider")
            else:
                labels.append("\u23F3 Can Wait")

        recommendations["Recommendation"] = labels
        return recommendations[
            ["name", "position", "team", "Recommendation", "adp", "rank",
             "Tier", "DraftIQ_Score", "Draft_Urgency", "Reason"]
        ]

    # ------------------------------------------------------------------
    # Comparison tools
    # ------------------------------------------------------------------

    def compare_players(self, player_names):
        available = self.get_available_players().copy()
        tier_table = self.get_position_tier_table()
        needs = self.get_roster_needs()
        results = []
        for name in player_names:
            matches = available[available["name"].str.lower() == name.lower()]
            if len(matches) == 0:
                continue
            player = matches.iloc[0]
            score, _ = self.calculate_draftiq_score(player, tier_table, needs)
            raw_urgency, _ = self.calculate_draft_urgency(player, needs, tier_table)
            urgency = min(100, round(raw_urgency))
            tier = self.player_tier(player["name"], tier_table)
            position_need = needs.get(player["position"], "OK")
            probability = self.next_pick_probability(player)

            # (100-probability) used to be added here again at 15% weight,
            # on top of the same value already driving urgency's own
            # return-risk component - same redundancy as
            # draftiq_decision_engine's removed risk_score. Folded into
            # urgency's weight instead of dropped.
            strategy_score = score * 0.45 + raw_urgency * 0.45
            # Keep the raw score for ranking; only cap/round for display.
            # Capping before sorting collapses every player whose blend
            # legitimately exceeds 100 down to the same displayed number,
            # losing real ranking order between them (same issue already
            # fixed in draftiq_decision_engine).
            raw_strategy_score = strategy_score
            strategy_score = round(min(100, strategy_score), 1)

            reasons = []
            if position_need == "HIGH":
                reasons.append(f"{player['position']} needed")
            if tier == 1:
                reasons.append("Elite Tier")
            if urgency >= 70:
                reasons.append("High urgency")
            if probability < 30:
                reasons.append("Very unlikely to return")
            if probability >= 70:
                reasons.append("Likely to return")

            results.append({
                "Player": player["name"], "Position": player["position"], "Tier": tier,
                "DraftIQ Score": round(score, 1), "Urgency": urgency, "Next Pick %": probability,
                "Strategy Score": strategy_score, "_RawStrategyScore": raw_strategy_score,
                "Reason": ", ".join(reasons),
            })

        if not results:
            return pd.DataFrame()
        results_df = pd.DataFrame(results).sort_values("_RawStrategyScore", ascending=False).reset_index(drop=True)
        return results_df.drop(columns=["_RawStrategyScore"])

    def opportunity_cost(self, player_names):
        available = self.get_available_players().copy()
        tier_table = self.get_position_tier_table()
        needs = self.get_roster_needs()
        results = []
        for name in player_names:
            matches = available[available["name"].str.lower() == name.lower()]
            if len(matches) == 0:
                continue
            player = matches.iloc[0]
            score, _ = self.calculate_draftiq_score(player, tier_table, needs)
            raw_urgency, _ = self.calculate_draft_urgency(player, needs, tier_table)
            urgency = min(100, round(raw_urgency))
            tier = self.player_tier(player["name"], tier_table)
            probability = self.next_pick_probability(player)

            tier_cost = max(0, 20 - (((tier or 5) - 1) * 4))
            # availability_cost (100-probability) used to be added here
            # again at 40% weight, on top of the same value already
            # driving urgency's own return-risk component - same
            # redundancy as draftiq_decision_engine's removed risk_score
            # and compare_players' removed direct probability term. Folded
            # into urgency_cost's weight instead of dropped.
            urgency_cost = raw_urgency * 0.70
            raw_opportunity_score = tier_cost * 1.5 + urgency_cost
            # Keep the raw score for ranking; only cap/round for display -
            # same reasoning as compare_players/draftiq_decision_engine.
            opportunity_score = round(min(100, raw_opportunity_score), 1)

            reasons = []
            if probability < 30:
                reasons.append("Very unlikely to return")
            elif probability < 60:
                reasons.append("Moderate return risk")
            else:
                reasons.append("Likely to return")
            if tier == 1:
                reasons.append("Elite tier")
            if urgency >= 75:
                reasons.append("High urgency")

            results.append({
                "Player": player["name"], "Position": player["position"], "Tier": tier,
                "DraftIQ Score": round(score, 1), "Urgency": urgency, "Next Pick %": probability,
                "Opportunity Cost": opportunity_score, "_RawOpportunityScore": raw_opportunity_score,
                "Reason": ", ".join(reasons),
            })

        if not results:
            return pd.DataFrame()
        results_df = pd.DataFrame(results).sort_values("_RawOpportunityScore", ascending=False).reset_index(drop=True)
        return results_df.drop(columns=["_RawOpportunityScore"])

    def draft_strategy_report(self, player_names=None):
        recommendations = self.get_draft_recommendations(10)
        if recommendations is None or len(recommendations) == 0:
            return {"error": "No recommendations available."}

        if player_names is None:
            player_names = recommendations["name"].head(5).tolist()

        comparison = self.compare_players(player_names)
        if comparison is None or len(comparison) == 0:
            return {"error": "Unable to compare players."}

        cost = self.opportunity_cost(player_names)
        best = comparison.iloc[0]
        best_name = best["Player"]

        match = recommendations[recommendations["name"] == best_name]
        recommendation = match.iloc[0].to_dict() if len(match) > 0 else None

        return {
            "pick": self.draft_state["pick"],
            "round": self.draft_state["round"],
            "my_roster": self.get_team_roster(self.my_team),
            "roster_needs": self.get_roster_needs(),
            "recommended_player": best_name,
            "recommendation": recommendation,
            "comparison": comparison.to_dict("records"),
            "opportunity_cost": cost.to_dict("records") if cost is not None else [],
        }

    # ------------------------------------------------------------------
    # Context decision engine (v0.4.7 / v0.4.8)
    # ------------------------------------------------------------------

    def _urgency_category(self, value, high_threshold, moderate_threshold):
        """Low/Moderate/High bucketing for display, using the same
        thresholds already established for the reason-tag text (just
        collapsing "Very high" and "High" into one High bucket)."""
        if value >= high_threshold:
            return "High"
        if value >= moderate_threshold:
            return "Moderate"
        return "Low"

    def _build_explanation(self, tier, position_need, probability, position, urgency_signal_reasons):
        """One or two plain-language sentences summarizing why this pick
        is recommended, built from the same underlying signals as the
        score (tier, roster need, survival odds, tier cliff/position run)
        rather than just restating the short reason tags."""
        if position_need == "HIGH":
            value_clause = "Strong value relative to your roster needs"
        elif tier == 1:
            value_clause = "Elite value at this point in the draft"
        elif tier == 2:
            value_clause = "Strong overall value at this point in the draft"
        else:
            value_clause = "Solid value based on his overall ranking"

        if probability <= 35:
            risk_clause = "a significant chance of being gone before your next pick"
        elif probability <= 65:
            risk_clause = "a moderate chance of being gone before your next pick"
        else:
            risk_clause = "a good chance he's still available if you wait"

        sentence = f"{value_clause}, with {risk_clause}."

        extra = []
        if any("Tier cliff" in r for r in urgency_signal_reasons):
            extra.append(f"The {position} tier is thinning out fast")
        if any(r.endswith("run") for r in urgency_signal_reasons):
            extra.append(f"There's been a run on {position}s recently")
        if extra:
            sentence += " " + " and ".join(extra) + "."
        return sentence

    def draftiq_decision_engine(self, count=10):
        available = self.get_available_players().copy()
        if self.avoid_players:
            available = available[~available["name"].isin(self.avoid_players)]
        if len(available) == 0:
            return pd.DataFrame()

        current_pick = self.draft_state["pick"]
        round_number = ((current_pick - 1) // len(self.league)) + 1
        tier_table = self.get_position_tier_table()
        tier_lookup = dict(zip(tier_table["name"], tier_table["Tier"]))
        tier_position_counts = tier_table.groupby(["position", "Tier"]).size().to_dict()
        needs = self.get_roster_needs()
        position_counts = self.get_position_counts()
        position_remaining_counts = available["position"].value_counts().to_dict()
        tier_one_remaining_by_position = (
            tier_table[tier_table["Tier"] == 1]["position"].value_counts().to_dict()
        )
        recent_position_counts = self._recent_position_counts()

        results = []
        for _, player in available.iterrows():
            score, _ = self.calculate_draftiq_score(player, tier_table, needs, tier_lookup, tier_position_counts, recent_position_counts)
            player_urgency, urgency_signal_reasons = self.calculate_draft_urgency(player, needs, tier_table, tier_lookup, tier_position_counts, recent_position_counts)
            tier = tier_lookup.get(player["name"])
            position_urgency, _ = self.calculate_position_urgency(
                player["position"], tier_table, position_counts,
                position_remaining_counts, tier_one_remaining_by_position,
                recent_position_counts,
            )
            probability = self.next_pick_probability(player)

            adp = player["adp"]
            position_need = needs.get(player["position"], "OK")

            # NOTE: a "value_score" bucketed on (adp - current_pick) used to
            # be added here at 5% weight. Removed for the same reason as the
            # urgency ADP bonus above - since current_pick is fixed across
            # every player scored in one pass, that gap is monotonic in raw
            # ADP, so it mathematically ALWAYS favored the worse-ADP player
            # between any two candidates. DraftIQ Score's own ADP component
            # (already inside `score` below, correctly oriented) covers this
            # signal properly, so this weight moves there instead of being
            # dropped, keeping the total weighting at 1.0.
            #
            # NOTE: a standalone `risk_score = 100 - probability` term used
            # to be added here too, at 20% weight. Removed: it was the exact
            # same next_pick_probability() value already driving
            # player_urgency's own return-risk component
            # (urgency += (100-probability)*0.65) - restating the identical
            # raw number as a second, separately-weighted top-level term
            # rather than adding any new information. player_urgency is the
            # designed home for "risk of losing this player" (it already
            # blends return risk with tier-cliff and position-run risk), so
            # its weight absorbs the removed 20% instead of dropping it.
            decision_score = (
                score * 0.45 + player_urgency * 0.45
                + position_urgency * 0.10
            )

            # Tier's contribution to decision_score now comes through
            # exactly once: inside `score` (calculate_draftiq_score's own
            # tier component, weighted 0.45 here). It used to also be
            # added again as a flat +8/+3 right here, on top of a second
            # copy that lived inside player_urgency - three additions of
            # the same "this player is elite" signal into one score. Both
            # duplicates are removed; `tier` is still used below for the
            # Tier column, reasons text, and the early-round position-type
            # modifier, none of which re-score the player's raw tier.

            # Roster need's contribution to decision_score now comes
            # through exactly once: inside `score` (calculate_draftiq_score's
            # own +15 for HIGH need, weighted 0.45 here). It used to also
            # be added again as a direct +4 right here, on top of a second
            # copy that lived inside player_urgency - the same three-way
            # stacking pattern as tier, just for a different signal. Both
            # duplicates are removed; `position_need` is still used below
            # for reasons text.

            if round_number <= 3:
                if player["position"] == "DEF":
                    decision_score -= 30
                elif player["position"] == "QB" and tier and tier > 1:
                    decision_score -= 8

            if round_number <= 3 and tier == 1 and player["position"] in ["RB", "WR", "TE"]:
                decision_score += 5

            # Sort ranking must use the UNCAPPED raw score. Capping it here
            # for display was collapsing every player who exceeds 100 (very
            # common for tier-1 players in early rounds, once urgency/risk/
            # tier/need bonuses stack up) down to the identical value 100.0
            # - which silently destroyed the real ranking among them and
            # left ties broken by arbitrary dataframe row order instead of
            # actual quality. A player who only just reaches 100 legitimately
            # could then out-sort players who are clearly better (higher raw
            # score) but got clipped to the same displayed number. Keep the
            # capped/rounded value for display and label thresholds (tuned
            # for a friendly 0-100 range), but rank by the raw value.
            raw_decision_score = decision_score
            decision_score = round(max(0, min(100, decision_score)), 1)

            reasons = []
            if tier == 1:
                reasons.append("Elite Tier")
            elif tier == 2:
                reasons.append("Strong Tier")
            if player_urgency >= 85:
                reasons.append("Very high player urgency")
            elif player_urgency >= 70:
                reasons.append("High player urgency")
            elif player_urgency >= 50:
                reasons.append("Moderate player urgency")
            if position_urgency >= 60:
                reasons.append(f"High {player['position']} urgency")
            elif position_urgency >= 35:
                reasons.append(f"Moderate {player['position']} urgency")
            if probability <= 20:
                reasons.append("Very unlikely to return")
            elif probability <= 40:
                reasons.append("Risk of losing player")
            if position_need == "HIGH":
                reasons.append(f"{player['position']} roster need")
            if player["name"] in self.target_players:
                reasons.insert(0, "\U0001F3AF Target")

            player_urgency_category = self._urgency_category(player_urgency, 70, 50)
            position_urgency_category = self._urgency_category(position_urgency, 60, 35)
            explanation = self._build_explanation(
                tier, position_need, probability, player["position"], urgency_signal_reasons
            )

            results.append({
                "Player": player["name"], "Position": player["position"],
                "DraftIQ Score": decision_score, "_RawDecisionScore": raw_decision_score,
                "Base Score": round(score, 1),
                "Player Urgency": player_urgency_category, "Position Urgency": position_urgency_category,
                "Next Pick %": probability, "ADP": adp, "Tier": tier,
                "Reason": ", ".join(reasons), "Explanation": explanation, "_Rank": player["rank"],
            })

        # Tie-break on rank (ascending - a lower rank number is the better
        # player) whenever two players land on the exact same raw Decision
        # Score, rather than leaving ties in arbitrary/incidental order.
        results_df = pd.DataFrame(results).sort_values(
            ["_RawDecisionScore", "_Rank"], ascending=[False, True]
        ).reset_index(drop=True)
        results_df = results_df.drop(columns=["_RawDecisionScore", "_Rank"])
        results_df.index += 1

        # Labels used to be assigned off fixed absolute score cutoffs
        # (DRAFT NOW >=78, Best Alternative >=72, ... else Wait). Decision
        # Score is built mostly from rank/ADP/tier, which mechanically
        # decline every round - by round 6-7 even the single best player
        # left commonly scores in the 50s, so every player fell to "Wait"
        # regardless of how good they were relative to what's actually on
        # the board. That's not useful: it's your pick, someone has to be
        # taken, and the best available option was being told to wait on
        # itself.
        #
        # Ranked tiers below are now based on rank position within the
        # CURRENT list (already sorted by raw score), so they scale with
        # round and remaining pool automatically - "top 3 available" means
        # the same thing in round 1 and round 10. Wait/Consider still use
        # score, but relative to the best score in this same list, not a
        # fixed number: a player only gets "Wait" when they're meaningfully
        # behind the best option actually on the board right now, not
        # merely because the whole board has gotten worse this round.
        DRAFT_THIS_RANK_CAP = 1
        BEST_ALT_RANK_CAP = 3
        STRONG_OPTION_RANK_CAP = 6
        CONSIDER_SCORE_RATIO = 0.65  # vs. this list's own top score

        best_score = results_df["DraftIQ Score"].max() if len(results_df) else 0
        labels = []
        for rank_position, (_, row) in enumerate(results_df.iterrows(), start=1):
            score = row["DraftIQ Score"]
            if rank_position <= DRAFT_THIS_RANK_CAP:
                labels.append("\U0001F525 DRAFT NOW")
            elif rank_position <= BEST_ALT_RANK_CAP:
                labels.append("\u2705 Best Alternative")
            elif rank_position <= STRONG_OPTION_RANK_CAP:
                labels.append("\U0001F44D Strong Option")
            elif best_score > 0 and score >= best_score * CONSIDER_SCORE_RATIO:
                labels.append("\U0001F4C8 Consider")
            else:
                labels.append("\u23F3 Wait")
        results_df["Decision"] = labels

        top = results_df.head(count).copy()

        missing_targets = [
            n for n in self.target_players
            if n in results_df["Player"].values and n not in top["Player"].values
        ]
        for name in missing_targets:
            row = results_df[results_df["Player"] == name].iloc[0].copy()
            probability = row["Next Pick %"]
            if probability <= 35:
                row["Decision"] = "\U0001F3AF TARGET ALERT"
                row["Reason"] = f"Target at risk - {probability}% to survive to your next pick, " + row["Reason"]
                top = pd.concat([top, row.to_frame().T], ignore_index=True)

        return top[
            ["Player", "Position", "Decision", "DraftIQ Score", "Base Score",
             "Player Urgency", "Position Urgency", "Next Pick %", "ADP", "Tier", "Reason", "Explanation"]
        ]

    def draftiq_on_clock_decision(self):
        recommendations = self.draftiq_decision_engine(10)
        if recommendations is None or len(recommendations) == 0:
            return {"error": "No recommendations available."}

        best = recommendations.iloc[0]
        current_pick = self.draft_state["pick"]
        round_number = ((current_pick - 1) // len(self.league)) + 1

        # Use the SAME roster-needs source of truth as the Recommendations
        # tab (get_roster_needs' HIGH/OK) for this specific factual claim,
        # rather than the Position Urgency composite score - those two can
        # legitimately disagree (e.g. WR shows HIGH roster need before
        # you've drafted one, while Position Urgency stays under its 60
        # threshold early in the draft), and saying "NOT a major need" here
        # while the Recommendations tab says "Position Need" for the same
        # player was a direct, confusing contradiction.
        roster_needs = self.get_roster_needs()
        position_is_needed = roster_needs.get(best["Position"]) == "HIGH"

        why = []
        if best["Tier"] == 1:
            why.append("Elite Tier player")
        if best["Player Urgency"] == "High":
            why.append("High player-specific urgency")
        elif best["Player Urgency"] == "Moderate":
            why.append("Moderate player-specific urgency")
        if best["Next Pick %"] <= 40:
            why.append(f"Only {best['Next Pick %']}% chance of returning")
        if position_is_needed:
            why.append(f"{best['Position']} is a roster need")
        else:
            why.append(f"{best['Position']} is not currently a roster need")

        if best["Tier"] == 1 and best["Player Urgency"] == "High" and best["Position Urgency"] == "Low":
            strategy = "Elite player value outweighs positional need. Do NOT force a position simply because your roster needs it."
        elif best["Position Urgency"] == "High" and best["Player Urgency"] == "High":
            strategy = "This pick addresses a positional need while protecting against losing the player."
        elif best["Player Urgency"] == "High":
            strategy = "The primary reason to draft now is the risk of losing this player."
        elif best["Position Urgency"] == "High":
            strategy = "The primary reason to draft now is positional scarcity."
        else:
            strategy = "This is primarily a value-based selection."

        if best["Next Pick %"] <= 30:
            cost = f"Passing on {best['Player']} carries significant risk because the player is unlikely to return."
        elif best["Next Pick %"] <= 60:
            cost = f"There is meaningful risk that {best['Player']} is gone at your next pick."
        else:
            cost = f"{best['Player']} has a reasonable chance of returning."

        return {
            "round": round_number,
            "pick": current_pick,
            "team": self.get_current_team_number(),
            "best": best.to_dict(),
            "why": why,
            "strategy": strategy,
            "opportunity_cost": cost,
            "alternatives": recommendations.iloc[1:4].to_dict("records"),
        }

    # ------------------------------------------------------------------
    # Draft-day simulation (bot picks) - v0.4.4 architecture
    # ------------------------------------------------------------------

    def build_simulation_scores(self, available):
        """Fast, vectorized approximate score for bot opponent picks.

        The full calculate_draftiq_score() does several per-player lookups
        (next_pick_probability, position-run detection, tier-cliff checks,
        roster needs) that are appropriately precise for a live human
        recommendation, but far too slow to re-run for every available
        player on every single simulated opponent pick - that's what made
        "Simulate opponents to my pick" effectively hang/time out with a
        real-sized player pool. This computes a rank/ADP/tier-based score
        for the whole available pool ONCE per simulate_to_me call and reuses
        it across all of that batch's simulated picks, which is what bot
        opponents' behavior needs to look reasonable (a good approximation
        of best-player-available), not perfect precision.
        """
        tier_table = self.get_position_tier_table()
        df = available.merge(tier_table[["name", "Tier"]], on="name", how="left")

        rank_score = (30 * (1 - (df["rank"] - 1) / 199)).clip(lower=0)
        tier_score = (10 - ((df["Tier"].fillna(5) - 1) * 2)).clip(lower=0)

        # ADP intentionally has no direct value component here, matching
        # calculate_draftiq_score - Rank+Tier is the one signal answering
        # "how good is this player" for bot opponents too. ADP still
        # drives which players get taken in the 10% "reach" branch of
        # simulate_opponent_pick_fast, so it isn't unused, just not a
        # second quality vote stacked on top of Rank/Tier.
        df["sim_score"] = rank_score + tier_score
        return dict(zip(df["name"], df["sim_score"]))

    def _bot_position_needs(self, roster_positions):
        needs = []
        if roster_positions.count("RB") < 3:
            needs.append("RB")
        if roster_positions.count("WR") < 3:
            needs.append("WR")
        if roster_positions.count("TE") < 1:
            needs.append("TE")
        if roster_positions.count("QB") < 1:
            needs.append("QB")
        return needs

    # Roughly double the corresponding _bot_position_needs threshold - a
    # team well past a normal starter+bench allotment at a position, not
    # just "already has enough".
    _STACK_THRESHOLD = {"RB": 5, "WR": 5, "TE": 3, "QB": 3}

    def _bot_stacked_positions(self, roster_positions):
        return {
            pos for pos, limit in self._STACK_THRESHOLD.items()
            if roster_positions.count(pos) >= limit
        }

    def simulate_opponent_pick_fast(self, available, roster_positions, score_map):
        if len(available) == 0:
            return None

        needs = self._bot_position_needs(roster_positions)
        stacked = self._bot_stacked_positions(roster_positions)

        roll = random.random()

        if roll < 0.70:
            # Best player available - but not at a position this team is
            # already heavily stacked at. Previously this branch ignored
            # roster composition entirely, so a team with 5+ RBs would
            # take a 6th just as readily as a team with none, as long as
            # it scored highest. Falls back to the full pool only if
            # excluding stacked positions leaves nothing (e.g. very late
            # in the draft).
            filtered = available[~available["position"].isin(stacked)] if stacked else available
            candidates = filtered if len(filtered) > 0 else available
        elif roll < 0.90 and needs:
            filtered = available[available["position"].isin(needs)]
            candidates = filtered if len(filtered) > 0 else available
        else:
            top_by_adp = available.sort_values("adp").head(min(10, len(available)))
            return top_by_adp.iloc[random.randint(0, len(top_by_adp) - 1)]

        candidates = candidates.copy()
        candidates["sim_score"] = candidates["name"].map(score_map).fillna(0)
        return candidates.sort_values("sim_score", ascending=False).iloc[0]

    def simulate_availability_probabilities(self, trials=150):
        """Monte Carlo estimate of (a) each available player's odds of
        still being on the board at the user's next pick, (b) how many
        picks each position is likely to see before then (forward-looking
        position run), and (c) how many players are likely to remain in
        each (position, tier) group by then (forward-looking tier cliff).

        Repeatedly simulates the picks between now and the user's next
        turn using the exact same bot behavior as
        simulate_opponent_pick_fast (need thresholds, 70/20/10 roll) -
        including each team's current real roster plus whatever it
        hypothetically drafts within that trial - so a run on a position,
        or a team that's already stacked at RB, actually shows up in the
        result. Never touches real draft_state; everything here works on
        local copies.

        Built on plain lists/dicts rather than DataFrame operations per
        simulated pick, since re-sorting a DataFrame for every single one
        of (trials x picks_away) picks is the actual cost driver in the
        existing pandas-based simulate_opponent_pick_fast path - a linear
        scan over a couple of presorted lists is much cheaper at this
        volume (a few hundred to ~700 simulated picks per call in the
        worst case).

        Returns {"player_probability": {name: 0-100}, "position_run_rate":
        {position: expected # picks at that position before your turn},
        "tier_cliff_remaining": {(position, tier): expected # remaining}}.
        """
        empty = {"player_probability": {}, "position_run_rate": {}, "tier_cliff_remaining": {}}
        picks_away = self.players_until_next_pick()
        available_df = self.get_available_players()
        names = available_df["name"].tolist()
        if picks_away <= 0 or len(names) == 0:
            return empty

        score_map = self.build_simulation_scores(available_df)
        position_by_name = dict(zip(available_df["name"], available_df["position"]))
        adp_by_name = dict(zip(available_df["name"], available_df["adp"]))

        tier_table = self.get_position_tier_table()
        tier_by_name = dict(zip(tier_table["name"], tier_table["Tier"]))
        group_key_by_name = {
            n: (position_by_name.get(n), tier_by_name.get(n))
            for n in names if tier_by_name.get(n) is not None
        }
        group_members = {}
        for n, key in group_key_by_name.items():
            group_members.setdefault(key, []).append(n)

        # Presort once per call (not per trial/pick) - each simulated pick
        # then just scans forward past whatever's already gone this trial.
        by_score_desc = sorted(names, key=lambda n: score_map.get(n, 0), reverse=True)
        by_adp_asc = sorted(
            names,
            key=lambda n: (pd.isna(adp_by_name.get(n)), adp_by_name.get(n) if not pd.isna(adp_by_name.get(n)) else 0),
        )
        by_score_desc_per_position = {}
        for n in names:
            by_score_desc_per_position.setdefault(position_by_name.get(n), []).append(n)
        for pos in by_score_desc_per_position:
            by_score_desc_per_position[pos].sort(key=lambda n: score_map.get(n, 0), reverse=True)

        current_pick = self.draft_state["pick"]
        pick_numbers = list(range(current_pick, current_pick + picks_away))
        team_for_pick = [self.get_team_for_pick(p) for p in pick_numbers]

        base_roster_positions = {}
        for team_number in set(team_for_pick):
            roster_names = self.get_team_roster(team_number)
            base_roster_positions[team_number] = [
                position_by_name.get(n) or self._roster_position_lookup(n) for n in roster_names
            ]

        survived_count = {n: 0 for n in names}
        position_run_totals = {}
        tier_cliff_totals = {key: 0 for key in group_members}

        def first_available(candidates_sorted, taken):
            for n in candidates_sorted:
                if n not in taken:
                    return n
            return None

        def first_available_excluding(candidates_sorted, taken, excluded_positions):
            fallback = None
            for n in candidates_sorted:
                if n in taken:
                    continue
                if fallback is None:
                    fallback = n
                if position_by_name.get(n) not in excluded_positions:
                    return n
            return fallback

        for _ in range(trials):
            taken = set()
            roster_positions = {t: list(v) for t, v in base_roster_positions.items()}
            for team_number in team_for_pick:
                needs = self._bot_position_needs(roster_positions.get(team_number, []))
                stacked = self._bot_stacked_positions(roster_positions.get(team_number, []))
                roll = random.random()
                picked = None

                if roll < 0.70:
                    picked = first_available_excluding(by_score_desc, taken, stacked)
                elif roll < 0.90 and needs:
                    for pos in needs:
                        picked = first_available(by_score_desc_per_position.get(pos, []), taken)
                        if picked:
                            break
                    if not picked:
                        picked = first_available(by_score_desc, taken)
                else:
                    top10 = [n for n in by_adp_asc if n not in taken][:10]
                    picked = random.choice(top10) if top10 else first_available(by_score_desc, taken)

                if picked is None:
                    break
                taken.add(picked)
                roster_positions.setdefault(team_number, []).append(position_by_name.get(picked))

            for n in names:
                if n not in taken:
                    survived_count[n] += 1

            trial_position_counts = {}
            for n in taken:
                pos = position_by_name.get(n)
                if pos:
                    trial_position_counts[pos] = trial_position_counts.get(pos, 0) + 1
            for pos, count in trial_position_counts.items():
                position_run_totals[pos] = position_run_totals.get(pos, 0) + count

            for key, members in group_members.items():
                remaining = sum(1 for n in members if n not in taken)
                tier_cliff_totals[key] += remaining

        return {
            "player_probability": {n: round(100 * survived_count[n] / trials) for n in names},
            "position_run_rate": {pos: total / trials for pos, total in position_run_totals.items()},
            "tier_cliff_remaining": {key: total / trials for key, total in tier_cliff_totals.items()},
        }

    def _ensure_availability_sim_cache(self):
        """Returns the cached Monte Carlo result for the current pick,
        building it once via simulate_availability_probabilities() if it's
        missing or stale. Shared by next_pick_probability, detect_position_run,
        and detect_tier_cliff so all three read from the exact same trial
        batch instead of each re-simulating separately."""
        current_pick = self.draft_state["pick"]
        cache = self._availability_sim_cache
        if cache is None or cache.get("pick") != current_pick:
            try:
                result = self.simulate_availability_probabilities()
            except Exception:
                result = {"player_probability": {}, "position_run_rate": {}, "tier_cliff_remaining": {}}
            cache = {"pick": current_pick, **result}
            self._availability_sim_cache = cache
        return cache

    def _roster_position_lookup(self, player_name):
        row = self.players_df.loc[self.players_df["name"] == player_name, "position"]
        return row.iloc[0] if len(row) else None

    def simulate_opponents_until_user(self):
        picks_made = []
        available = self.get_available_players().copy()
        score_map = self.build_simulation_scores(available)

        while self.get_current_team_number() != self.my_team:
            if len(available) == 0:
                break

            team_number = self.get_current_team_number()
            roster_names = self.get_team_roster(team_number)
            roster_positions = [
                self.players_df.loc[self.players_df["name"] == n, "position"].iloc[0]
                for n in roster_names
                if (self.players_df["name"] == n).any()
            ]

            selected = self.simulate_opponent_pick_fast(available, roster_positions, score_map)
            if selected is None:
                break

            selected_name = selected["name"]
            self.draft_player(selected_name)  # captures the correct team internally
            picks_made.append({"team": team_number, "player": selected_name})

            available = available[available["name"] != selected_name]

        return picks_made

    def draftiq_pick(self, player_name):
        current_team = self.get_current_team_number()
        if current_team != self.my_team:
            return {"ok": False, "message": f"It is not your turn. Team {current_team} is on the clock."}

        available = self.get_available_players()
        matches = available[available["name"].str.lower() == player_name.lower()]
        if len(matches) == 0:
            return {"ok": False, "message": f"{player_name} is not available."}

        player = matches.iloc[0]
        self.draft_player(player["name"])

        return {
            "ok": True,
            "drafted": player["name"],
            "position": player["position"],
            "pick": self.draft_state["pick"] - 1,
            "roster": self.get_team_roster(self.my_team),
            "roster_needs": self.get_roster_needs(),
            "next_pick": self.draft_state["pick"],
            "round": self.draft_state["round"],
        }
