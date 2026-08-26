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

        # Sleeper live-draft sync
        self.sleeper_draft_id = None
        self.sleeper_applied_count = 0

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

    def load_players_from_sleeper(self, sleeper_players_json):
        """Build a player pool from Sleeper's /v1/players/nfl payload.

        NOTE: Sleeper's public API does not expose expert rankings or ADP.
        The best available proxy is each player's `search_rank` field,
        which is Sleeper's own internal relevance ranking (used for their
        search/autocomplete), not a fantasy-specific ranking. It's useful
        as a rough ordering but shouldn't be treated as real ADP.
        """
        rows = []
        fantasy_positions = {"QB", "RB", "WR", "TE", "DEF", "K"}

        for player_id, p in sleeper_players_json.items():
            if not isinstance(p, dict):
                continue

            position = p.get("position")
            if position not in fantasy_positions:
                continue

            name = p.get("full_name")
            if not name:
                first = p.get("first_name") or ""
                last = p.get("last_name") or ""
                name = f"{first} {last}".strip()
            if not name:
                continue

            search_rank = p.get("search_rank")
            if search_rank is None:
                continue

            rows.append({
                "name": name,
                "position": position,
                "team": p.get("team"),
                "rank": search_rank,
            })

        df = pd.DataFrame(rows)
        if len(df) == 0:
            raise ValueError("No usable players found in Sleeper's player data.")

        df["rank"] = pd.to_numeric(df["rank"], errors="coerce")
        df = df.dropna(subset=["rank"]).sort_values("rank").reset_index(drop=True)

        # Sleeper's search_rank has large, uneven gaps between players and
        # duplicate values - collapse it to a clean 1..N ordering so the
        # tiering/scoring logic (which reasons about rank gaps) behaves
        # sensibly.
        df["rank"] = range(1, len(df) + 1)
        df["adp"] = df["rank"]

        self._set_players_df(df)

    def _set_players_df(self, df):
        self.players_df = df
        self._full_tier_table = None  # invalidate - recomputed lazily from the new pool
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

    def apply_sleeper_league_settings(self, league_json):
        """Pull team count, starting lineup, and scoring format from a
        Sleeper league's /v1/league/{league_id} payload."""
        total_rosters = league_json.get("total_rosters")
        if total_rosters:
            self.num_teams = int(total_rosters)

        roster_positions = league_json.get("roster_positions") or []
        starters = {}
        bench_count = 0
        flex_count = 0

        for slot in roster_positions:
            if slot == "BN":
                bench_count += 1
            elif slot == "IR":
                continue
            elif slot in ("FLEX", "WRRB_FLEX", "REC_FLEX", "SUPER_FLEX"):
                flex_count += 1
            else:
                starters[slot] = starters.get(slot, 0) + 1

        if flex_count:
            starters["FLEX"] = flex_count
        if starters:
            self.starters = starters
        if roster_positions:
            self.roster_size = len(roster_positions)
        if bench_count:
            self.bench = bench_count

        scoring = league_json.get("scoring_settings") or {}
        rec = scoring.get("rec", 0) or 0
        if rec >= 1:
            self.scoring = "Full PPR"
        elif rec >= 0.5:
            self.scoring = "0.5 PPR"
        elif rec > 0:
            self.scoring = f"{rec} PPR"
        else:
            self.scoring = "Standard"

        if self.loaded:
            self.league = self._create_league()
            self.draft_order = self._snake_order()
            self.reset_draft()

    def apply_sleeper_draft_settings(self, draft_json):
        """Fallback for mock/practice drafts with no attached league - pulls
        team count and round count straight from the draft object."""
        settings = draft_json.get("settings") or {}
        teams = settings.get("teams")
        rounds = settings.get("rounds")

        if teams:
            self.num_teams = int(teams)
        if rounds:
            self.rounds = int(rounds)

        if self.loaded:
            self.league = self._create_league()
            self.draft_order = self._snake_order()
            self.reset_draft()

    def resolve_my_slot(self, draft_json, sleeper_user_id):
        """Given a draft's draft_order map and a Sleeper user_id, return
        that user's 1-indexed draft slot, or None if not found."""
        draft_order = draft_json.get("draft_order") or {}
        return draft_order.get(sleeper_user_id)

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
        self.sleeper_applied_count = 0

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
        return self.players_df[
            ~self.players_df["name"].isin(self.draft_state["drafted_players"])
        ]

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

        # If this pick had come in via Sleeper sync, un-count it so a
        # future sync doesn't silently skip it as "already applied".
        if self.sleeper_applied_count > 0:
            self.sleeper_applied_count -= 1

        return {"ok": True, "undone": player_name, "team": team_name, "pick": last["pick"]}

    # ------------------------------------------------------------------
    # Sleeper live-draft sync
    # ------------------------------------------------------------------
    #
    # Sleeper's public API (no auth required) exposes picks for a draft at
    # GET https://api.sleeper.app/v1/draft/{draft_id}/picks
    # Each pick includes metadata with first_name/last_name/position.
    # We match those names against players_df and apply any pick we haven't
    # already applied via Sleeper sync, in pick order. This assumes your
    # DraftIQ league's team count matches the real draft's team count, so
    # pick order lines up 1:1 - team-by-team pick assignment on the DraftIQ
    # side is otherwise unaffected and still follows its own snake order.

    def _match_player_name(self, first_name, last_name, position=None, team=None):
        first_name = (first_name or "").strip()
        last_name = (last_name or "").strip()
        if not last_name:
            return None

        stored_lower = self.players_df["name"].str.lower()
        full_l = f"{first_name} {last_name}".strip().lower()

        # Exact full-name match first, in case the pool already uses full
        # first names (e.g. a Sleeper-imported pool) - keeps that case as
        # fast and unambiguous as before.
        exact = self.players_df[stored_lower == full_l]
        if len(exact) == 1:
            return exact.iloc[0]["name"]
        if len(exact) > 1:
            return None  # ambiguous - don't silently guess which one

        # Primary path: match by last name. Sleeper always reports a full
        # first name, but rank-sheet CSVs commonly abbreviate it - a
        # first+last comparison (equality or substring) can never bridge
        # "Jahmyr Gibbs" vs "J. Gibbs" in either direction. Last names are
        # written out in full by both conventions. Strip a trailing
        # "(POS)" disambiguator suffix (e.g. "J. Love (RB)") first so it
        # isn't mistaken for part of the name.
        stripped = stored_lower.str.replace(r"\s*\([a-z]+\)$", "", regex=True)
        last_l = last_name.lower()
        last_matches = self.players_df[stripped.str.contains(last_l, regex=False)]

        if len(last_matches) == 1:
            return last_matches.iloc[0]["name"]

        if len(last_matches) > 1:
            # More than one player shares this last name - narrow using
            # position/team from Sleeper's own pick metadata rather than
            # guessing. Only narrow when it actually resolves something;
            # never guess if it stays ambiguous.
            narrowed = last_matches
            if position:
                by_pos = narrowed[narrowed["position"].str.lower() == position.strip().lower()]
                if len(by_pos) > 0:
                    narrowed = by_pos
            if team:
                by_team = narrowed[narrowed["team"].str.lower() == team.strip().lower()]
                if len(by_team) > 0:
                    narrowed = by_team
            if len(narrowed) == 1:
                return narrowed.iloc[0]["name"]
            return None  # still ambiguous - don't guess

        return None

    def sync_sleeper_picks(self, sleeper_picks):
        """sleeper_picks: list of pick dicts from Sleeper's picks endpoint,
        sorted by pick_no ascending. Applies any picks not yet applied via
        a prior sync call. Returns a report of what happened."""
        sleeper_picks = sorted(sleeper_picks, key=lambda p: p.get("pick_no", 0))
        new_picks = sleeper_picks[self.sleeper_applied_count:]

        applied = []
        unmatched = []

        for p in new_picks:
            meta = p.get("metadata") or {}
            first = (meta.get("first_name") or "").strip()
            last = (meta.get("last_name") or "").strip()
            name = f"{first} {last}".strip()

            self.sleeper_applied_count += 1

            if not name:
                continue

            matched_name = self._match_player_name(
                first, last, position=meta.get("position"), team=meta.get("team")
            )
            if matched_name is None:
                unmatched.append(name)
                continue
            if matched_name in self.draft_state["drafted_players"]:
                continue

            self.make_pick(matched_name)
            applied.append(matched_name)

        return {"applied": applied, "unmatched": unmatched}

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
        sheets - and the Sleeper importer, which collapses every position
        into one combined 1..N ranking - use an OVERALL rank, where other
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
        picks_away = self.players_until_next_pick()
        current_pick = self.draft_state["pick"]
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
        """recent_position_counts (an optional precomputed {position: count}
        dict) lets a caller looping over many players/positions skip
        recomputing this from scratch every time - its result only depends
        on draft history, not on which player is currently being scored, so
        it's identical for every player in a given scoring pass. Left
        uncached, this was doing a full dataframe filter per recently
        drafted player, called separately from three different scoring
        functions per player - the single largest remaining bottleneck on
        a large player pool once any picks have actually been made."""
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
        """tier_lookup ({name: tier}) and tier_position_counts
        ({(position, tier): count}) are optional precomputed caches - pass
        them when calling this in a per-player loop to avoid a pandas
        boolean-mask filter on every single call, which dominated runtime
        on a large player pool even with tier_table itself cached."""
        if tier_lookup is not None and tier_position_counts is not None:
            tier = tier_lookup.get(player["name"])
            if tier is None:
                return False, 0
            remaining = tier_position_counts.get((player["position"], tier), 0)
            return remaining <= 3, remaining

        if tier_table is None:
            tier_table = self.get_position_tier_table()
        player_info = tier_table[tier_table["name"] == player["name"]]
        if len(player_info) == 0:
            return False, 0
        tier = player_info.iloc[0]["Tier"]
        position = player["position"]
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

        adp = player["adp"]
        score += max(0, 25 * (1 - (adp - 1) / 199))

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

        run_detected, run_count = self.detect_position_run(player["position"], recent_position_counts=recent_position_counts)
        if run_detected:
            score += 5
            reasons.append(f"{player['position']} run detected ({run_count} recent picks)")

        cliff, remaining = self.detect_tier_cliff(player, tier_table, tier_lookup, tier_position_counts)
        if cliff:
            score += 5
            reasons.append(f"Tier cliff - only {remaining} {player['position']} left")

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
        return min(100, round(urgency)), reasons

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
        if len(available) == 0:
            return pd.DataFrame()

        tier_table = self.get_position_tier_table()
        needs = self.get_roster_needs()
        tier_lookup = dict(zip(tier_table["name"], tier_table["Tier"]))
        tier_position_counts = tier_table.groupby(["position", "Tier"]).size().to_dict()
        recent_position_counts = self._recent_position_counts()
        scores, reasons, tiers, urgencies = [], [], [], []

        for _, player in available.iterrows():
            score, reason = self.calculate_draftiq_score(player, tier_table, needs, tier_lookup, tier_position_counts, recent_position_counts)
            urgency, urgency_reason = self.calculate_draft_urgency(player, needs, tier_table, tier_lookup, tier_position_counts, recent_position_counts)
            scores.append(score)
            reasons.append(", ".join(reason + urgency_reason))
            urgencies.append(urgency)
            tiers.append(tier_lookup.get(player["name"]))

        available["DraftIQ_Score"] = scores
        available["Reason"] = reasons
        available["Tier"] = tiers
        available["Draft_Urgency"] = urgencies

        current_pick = self.draft_state["pick"]
        round_number = ((current_pick - 1) // len(self.league)) + 1

        # Rank/select by DraftIQ_Score blended with Draft_Urgency, so a
        # player at real risk of being lost (tier cliff, position run,
        # pressing roster need) can actually surface higher - not just get
        # described as urgent after the score alone already cut them.
        selection_score = available["DraftIQ_Score"] + available["Draft_Urgency"] * 0.3
        available = available.assign(_selection_score=selection_score) \
            .sort_values("_selection_score", ascending=False) \
            .drop(columns=["_selection_score"]) \
            .reset_index(drop=True)
        recommendations = available.head(count).copy()

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

            if must_draft:
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
            urgency, _ = self.calculate_draft_urgency(player, needs, tier_table)
            tier = self.player_tier(player["name"], tier_table)
            position_need = needs.get(player["position"], "OK")
            probability = self.next_pick_probability(player)

            # (100-probability) used to be added here again at 15% weight,
            # on top of the same value already driving urgency's own
            # return-risk component - same redundancy as
            # draftiq_decision_engine's removed risk_score. Folded into
            # urgency's weight instead of dropped.
            strategy_score = score * 0.45 + urgency * 0.45
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
            urgency, _ = self.calculate_draft_urgency(player, needs, tier_table)
            tier = self.player_tier(player["name"], tier_table)
            probability = self.next_pick_probability(player)

            tier_cost = max(0, 20 - (((tier or 5) - 1) * 4))
            # availability_cost (100-probability) used to be added here
            # again at 40% weight, on top of the same value already
            # driving urgency's own return-risk component - same
            # redundancy as draftiq_decision_engine's removed risk_score
            # and compare_players' removed direct probability term. Folded
            # into urgency_cost's weight instead of dropped.
            urgency_cost = urgency * 0.70
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

    def what_if_i_wait(self, player_name=None, alternative_count=5):
        recommendations = self.get_draft_recommendations(10)
        if recommendations is None or len(recommendations) == 0:
            return {"error": "No recommendations available."}

        target_name = player_name or recommendations.iloc[0]["name"]
        matches = recommendations[recommendations["name"].str.lower() == target_name.lower()]
        if len(matches) == 0:
            return {"error": f"{target_name} is not currently in the recommendation group."}
        target = matches.iloc[0]

        available = self.get_available_players().copy()
        player_matches = available[available["name"].str.lower() == target_name.lower()]
        if len(player_matches) == 0:
            return {"error": f"{target_name} is no longer available."}
        player = player_matches.iloc[0]

        return_probability = self.next_pick_probability(player)
        loss_probability = round(100 - return_probability, 1)

        current_recommendation = target["Recommendation"]
        score, urgency, tier = target["DraftIQ_Score"], target["Draft_Urgency"], target["Tier"]

        alternatives = recommendations[
            recommendations["name"].str.lower() != target_name.lower()
        ].head(alternative_count).reset_index(drop=True)

        if return_probability < 20:
            wait_risk, wait_message = "\U0001F534 Very High", f"Only {return_probability}% chance this player returns."
        elif return_probability < 40:
            wait_risk, wait_message = "\U0001F7E0 High", f"Only {return_probability}% chance this player returns."
        elif return_probability < 60:
            wait_risk, wait_message = "\U0001F7E1 Moderate", f"{return_probability}% chance this player returns."
        else:
            wait_risk, wait_message = "\U0001F7E2 Low", f"{return_probability}% chance this player returns."

        if return_probability < 30 and (tier == 1 or urgency >= 60 or current_recommendation == "\U0001F525 Must Draft"):
            conclusion = f"TAKE {target_name} NOW"
            conclusion_reason = "Waiting carries significant risk because this player is unlikely to survive to your next pick."
        elif return_probability >= 60:
            conclusion = f"YOU CAN WAIT ON {target_name}"
            conclusion_reason = "The player has a reasonable chance of surviving to your next pick."
        else:
            conclusion = f"WAITING ON {target_name} IS RISKY"
            conclusion_reason = "There is meaningful risk of losing the player before your next selection."

        return {
            "target": target_name,
            "recommendation": current_recommendation,
            "draftiq_score": score,
            "draft_urgency": urgency,
            "tier": tier,
            "return_probability": return_probability,
            "loss_probability": loss_probability,
            "wait_risk": wait_risk,
            "wait_message": wait_message,
            "conclusion": conclusion,
            "conclusion_reason": conclusion_reason,
            "alternatives": alternatives.to_dict("records"),
        }

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

    def draftiq_decision_engine(self, count=10):
        available = self.get_available_players().copy()
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
            player_urgency, _ = self.calculate_draft_urgency(player, needs, tier_table, tier_lookup, tier_position_counts, recent_position_counts)
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

            results.append({
                "Player": player["name"], "Position": player["position"],
                "Decision Score": decision_score, "_RawDecisionScore": raw_decision_score,
                "DraftIQ Score": round(score, 1),
                "Player Urgency": player_urgency, "Position Urgency": position_urgency,
                "Next Pick %": probability, "ADP": adp, "Tier": tier,
                "Reason": ", ".join(reasons),
            })

        results_df = pd.DataFrame(results).sort_values("_RawDecisionScore", ascending=False).reset_index(drop=True)
        results_df = results_df.drop(columns=["_RawDecisionScore"])
        results_df.index += 1

        # Labels used to be assigned off fixed absolute score cutoffs
        # (DRAFT THIS >=78, Best Alternative >=72, ... else Wait). Decision
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
        DRAFT_THIS_RANK_CAP = 3
        BEST_ALT_RANK_CAP = 6
        STRONG_OPTION_RANK_CAP = 8
        CONSIDER_SCORE_RATIO = 0.65  # vs. this list's own top score

        best_score = results_df["Decision Score"].max() if len(results_df) else 0
        labels = []
        for rank_position, (_, row) in enumerate(results_df.iterrows(), start=1):
            score = row["Decision Score"]
            if rank_position <= DRAFT_THIS_RANK_CAP:
                labels.append("\U0001F525 DRAFT THIS")
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
        return top[
            ["Player", "Position", "Decision", "Decision Score", "DraftIQ Score",
             "Player Urgency", "Position Urgency", "Next Pick %", "ADP", "Tier", "Reason"]
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
        if best["Player Urgency"] >= 80:
            why.append("Very high player-specific urgency")
        elif best["Player Urgency"] >= 60:
            why.append("High player-specific urgency")
        if best["Next Pick %"] <= 40:
            why.append(f"Only {best['Next Pick %']}% chance of returning")
        if position_is_needed:
            why.append(f"{best['Position']} is a roster need")
        else:
            why.append(f"{best['Position']} is not currently a roster need")

        if best["Tier"] == 1 and best["Player Urgency"] >= 70 and best["Position Urgency"] < 35:
            strategy = "Elite player value outweighs positional need. Do NOT force a position simply because your roster needs it."
        elif best["Position Urgency"] >= 60 and best["Player Urgency"] >= 70:
            strategy = "This pick addresses a positional need while protecting against losing the player."
        elif best["Player Urgency"] >= 70:
            strategy = "The primary reason to draft now is the risk of losing this player."
        elif best["Position Urgency"] >= 60:
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
        adp_score = (25 * (1 - (df["adp"] - 1) / 199)).clip(lower=0)
        tier_score = (10 - ((df["Tier"].fillna(5) - 1) * 2)).clip(lower=0)

        df["sim_score"] = rank_score + adp_score + tier_score
        return dict(zip(df["name"], df["sim_score"]))

    def simulate_opponent_pick_fast(self, available, roster_positions, score_map):
        if len(available) == 0:
            return None

        needs = []
        if roster_positions.count("RB") < 3:
            needs.append("RB")
        if roster_positions.count("WR") < 3:
            needs.append("WR")
        if roster_positions.count("TE") < 1:
            needs.append("TE")
        if roster_positions.count("QB") < 1:
            needs.append("QB")

        roll = random.random()

        if roll < 0.70:
            candidates = available
        elif roll < 0.90 and needs:
            filtered = available[available["position"].isin(needs)]
            candidates = filtered if len(filtered) > 0 else available
        else:
            top_by_adp = available.sort_values("adp").head(min(10, len(available)))
            return top_by_adp.iloc[random.randint(0, len(top_by_adp) - 1)]

        candidates = candidates.copy()
        candidates["sim_score"] = candidates["name"].map(score_map).fillna(0)
        return candidates.sort_values("sim_score", ascending=False).iloc[0]

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
