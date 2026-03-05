"""
app.py — NHL IRT Dashboard
==========================
Run with:
    streamlit run dashboard/app.py
"""
from __future__ import annotations

import json
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

# ─────────────────────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────────────────────

DATA_DIR = Path(__file__).parent / "data"

SITUATION_LABELS: dict = {
    "all":  "All Situations",
    "even": "Even Strength (5v5)",
    "pp":   "Power Play",
}


def _data_dir(situation: str) -> Path:
    return DATA_DIR / situation


# ─────────────────────────────────────────────────────────────────────────────
# Data loading (cached)
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_data
def load_goalie_summary(situation: str = "all") -> pd.DataFrame:
    return pd.read_csv(_data_dir(situation) / "goalie_summary.csv")


@st.cache_data
def load_goalie_weekly(situation: str = "all") -> pd.DataFrame:
    return pd.read_csv(_data_dir(situation) / "goalie_weekly.csv")


@st.cache_data
def load_shooter_summary(situation: str = "all") -> pd.DataFrame:
    return pd.read_csv(_data_dir(situation) / "shooter_summary.csv")


@st.cache_data
def load_shooter_weekly(situation: str = "all") -> pd.DataFrame:
    return pd.read_csv(_data_dir(situation) / "shooter_weekly.csv")


@st.cache_data
def load_meta(situation: str = "all") -> dict:
    path = _data_dir(situation) / "meta.json"
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {}


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

GOALIE_COLS = {
    "player_name":            "Name",
    "team":                   "Team",
    "shots_faced":            "Shots",
    "goals_actual":           "GA",
    "goals_xg_only":          "xGA",
    "gsax_raw":               "GSAx (raw)",
    "shooter_adj":            "Shooter Adj",
    "gsax":                   "GSAx (IRT)",
    "goals_model_predicted":  "Model GA",
}

SHOOTER_COLS = {
    "player_name":             "Name",
    "team":                    "Team",
    "position":                "Pos",
    "shots_taken":             "Shots",
    "goals_actual":            "Goals",
    "goals_xg_only":           "xG",
    "fsax_raw":                "FSAx (raw)",
    "goalie_difficulty_adj":   "Goalie Adj",
    "fsax":                    "FSAx (IRT)",
}


def fmt_signed(val: float) -> str:
    return f"+{val:.2f}" if val >= 0 else f"{val:.2f}"


def min_shots_slider(df: pd.DataFrame, col: str, key: str, label: str = "Min shots") -> pd.DataFrame:
    if col not in df.columns:
        return df
    max_v = int(df[col].max()) if col in df.columns else 100
    default_v = max(0, min(50, int(df[col].quantile(0.75))))
    min_v = st.sidebar.slider(label, 0, max_v, default_v, step=5, key=key)
    return df[df[col] >= min_v]


# ─────────────────────────────────────────────────────────────────────────────
# Chart factory
# ─────────────────────────────────────────────────────────────────────────────

def _canonical_weekly(
    df_week: pd.DataFrame,
    selected_names: list,
    id_col: str,
) -> pd.DataFrame:
    """Filter weekly data to selected player names.

    When multiple player IDs share the same display name (e.g. two 'Sebastian Aho'),
    keep only the ID with the most career shots so the line chart isn't zigzagged.
    """
    sub = df_week[df_week["player_name"].isin(selected_names)].copy()
    if sub.empty:
        return sub
    # For each name, find the id with the most shots and drop the others.
    shots_by_id = (
        sub.groupby(["player_name", id_col])["shots_that_week"]
        .sum()
        .reset_index()
    )
    best_id = (
        shots_by_id.sort_values("shots_that_week", ascending=False)
        .groupby("player_name")[id_col]
        .first()
    )
    keep_ids = set(best_id.values)
    return sub[sub[id_col].isin(keep_ids)].copy()


def trajectory_chart(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    lo_col: str,
    hi_col: str,
    color_col: str,
    title: str,
    y_label: str,
    x_label: str = "Week",
) -> alt.Chart:
    """Line + confidence band for any skill over time."""
    df = df.copy()

    # Fix axis scale to the skill-line range (with padding) so wide
    # confidence bands in low-shot weeks don't collapse the chart.
    y_min = float(df[y_col].min())
    y_max = float(df[y_col].max())
    pad   = max((y_max - y_min) * 0.25, 0.15)
    dom_lo, dom_hi = y_min - pad, y_max + pad
    df[lo_col] = df[lo_col].clip(lower=dom_lo)
    df[hi_col] = df[hi_col].clip(upper=dom_hi)
    y_scale = alt.Scale(domain=[dom_lo, dom_hi])

    tooltip = [
        alt.Tooltip(f"{color_col}:N", title="Player"),
        alt.Tooltip(f"{x_col}:Q", title=x_label),
        alt.Tooltip(f"{y_col}:Q", title=y_label, format=".3f"),
        alt.Tooltip("shots_that_week:Q", title="Shots this week"),
    ]
    band = (
        alt.Chart(df)
        .mark_area(opacity=0.15)
        .encode(
            x=alt.X(f"{x_col}:Q", title=x_label),
            y=alt.Y(f"{lo_col}:Q", title="", scale=y_scale),
            y2=alt.Y2(f"{hi_col}:Q"),
            color=alt.Color(f"{color_col}:N", legend=alt.Legend(title="Player")),
            tooltip=tooltip,
        )
    )
    line = (
        alt.Chart(df)
        .mark_line(strokeWidth=2, point=alt.OverlayMarkDef(size=30))
        .encode(
            x=alt.X(f"{x_col}:Q", title=x_label),
            y=alt.Y(f"{y_col}:Q", title=y_label, scale=y_scale),
            color=alt.Color(f"{color_col}:N", legend=alt.Legend(title="Player")),
            tooltip=tooltip,
        )
    )
    return (band + line).properties(title=title, height=420, width="container")


# ─────────────────────────────────────────────────────────────────────────────
# Page: Goalie Leaderboard
# ─────────────────────────────────────────────────────────────────────────────

def page_goalie_leaderboard(situation: str = "all") -> None:
    st.header("Goalie Leaderboard")
    with st.expander("ℹ️ What do these numbers mean?"):
        st.markdown(
            "- **GSAx (IRT)** — Goals Saved Above Expected, adjusted for shooter quality. "
            "Positive means the goalie stopped more goals than a league-average goalie would have against the same shooters.\n"
            "- **GSAx (raw)** — Same idea but *without* adjusting for shooter skill (uses plain xG).\n"
            "- **Shooter Adj** — How much GSAx changes when we account for the quality of shooters each goalie faced. "
            "Positive means they faced tougher-than-average shooters.\n"
            "- **Model GA** — Goals the full IRT model expected this goalie to allow (given their skill, shooters & shot quality).\n"
            "- **xGA** — Expected goals from the raw xG model (no IRT adjustment).\n"
        )
    df_full = load_goalie_summary(situation)

    teams     = sorted(df_full["team"].dropna().unique().tolist())
    sel_teams = st.sidebar.multiselect("Filter by team", teams, key="gl_team")
    df_full   = min_shots_slider(df_full, "shots_faced", "gl_shots", "Min shots faced")
    sort_col  = st.sidebar.selectbox(
        "Sort by", ["gsax", "gsax_raw", "shots_faced"], index=0, key="gl_sort"
    )

    # Rank among the full shots-qualified pool, then apply team filter
    df_full = df_full.sort_values(sort_col, ascending=False).reset_index(drop=True)
    df_full["Overall Rank"] = df_full.index + 1

    if sel_teams:
        df = df_full[df_full["team"].isin(sel_teams)].copy().reset_index(drop=True)
        df["Team Rank"] = df.index + 1
    else:
        df = df_full

    rank_cols    = ["Overall Rank"] + (["Team Rank"] if sel_teams else [])
    display_cols = rank_cols + [c for c in GOALIE_COLS if c in df.columns]
    renamed = df[display_cols].rename(columns=GOALIE_COLS)
    st.dataframe(renamed, hide_index=True, use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# Page: Goalie Trajectory
# ─────────────────────────────────────────────────────────────────────────────

def page_goalie_trajectory(situation: str = "all") -> None:
    st.header("Goalie Skill Trajectory")
    with st.expander("ℹ️ What do these numbers mean?"):
        st.markdown(
            "- **φ (phi)** — The goalie's estimated save skill for a given week. "
            "φ > 0 means better than league average; φ < 0 means worse.\n"
            "- **Shaded band** — Uncertainty around the estimate. Wider bands appear in "
            "weeks with fewer shots (less information).\n"
            "- The chart shows how a goalie's form evolves across the season.\n"
        )
    df_sum  = load_goalie_summary(situation)
    df_week = load_goalie_weekly(situation)

    # Player selector
    all_goalies = sorted(df_sum["player_name"].dropna().unique().tolist())
    default_g = all_goalies[:5] if len(all_goalies) >= 5 else all_goalies
    selected = st.multiselect(
        "Select goalies (up to 10)", all_goalies, default=default_g, key="gtraj_sel"
    )
    if not selected:
        st.info("Select at least one goalie.")
        return

    sub = _canonical_weekly(df_week, selected, "goalie_id")
    if sub.empty:
        st.warning("No weekly data found for the selected players.")
        return

    st.altair_chart(
        trajectory_chart(
            sub, "week", "phi", "phi_lo", "phi_hi", "player_name",
            title="φ (Save Skill)  — higher = better goalie",
            y_label="φ",
        ),
        use_container_width=True,
    )
    st.caption(
        "φ > 0 → better than league average; φ < 0 → worse than average.  "
        "Shaded band shows uncertainty (widens in low-shot weeks)."
    )

    # Raw table
    with st.expander("Weekly data table"):
        cols = [c for c in ["player_name", "team", "week", "phi", "mu_phi",
                            "phi_lo", "phi_hi", "shots_that_week"] if c in sub.columns]
        st.dataframe(sub[cols].sort_values(["player_name", "week"]))


# ─────────────────────────────────────────────────────────────────────────────
# Page: Shooter Leaderboard
# ─────────────────────────────────────────────────────────────────────────────

def page_shooter_leaderboard(situation: str = "all") -> None:
    st.header("Shooter Leaderboard")
    with st.expander("ℹ️ What do these numbers mean?"):
        st.markdown(
            "- **FSAx (IRT)** — Finishing Skill Above Expected, adjusted for goalie quality. "
            "Positive means the shooter scored more than expected against the goalies they actually faced.\n"
            "- **FSAx (raw)** — Same idea but *without* adjusting for goalie skill (uses plain xG).\n"
            "- **Goalie Adj** — How much FSAx changes when we account for the quality of goalies the shooter faced. "
            "Positive means they faced tougher-than-average goalies.\n"
            "- **xG** — Expected goals from the raw xG model (no IRT adjustment).\n"
        )
    df_full = load_shooter_summary(situation)

    teams     = sorted(df_full["team"].dropna().unique().tolist())
    sel_teams = st.sidebar.multiselect("Filter by team", teams, key="sl_team")

    pos_opts = ["All"] + sorted(df_full["position"].dropna().unique().tolist()) if "position" in df_full.columns else ["All"]
    sel_pos  = st.sidebar.selectbox("Position", pos_opts, key="sl_pos")
    if sel_pos != "All" and "position" in df_full.columns:
        df_full = df_full[df_full["position"] == sel_pos]

    df_full  = min_shots_slider(df_full, "shots_taken", "sl_shots", "Min shots")
    sort_col = st.sidebar.selectbox(
        "Sort by", ["fsax", "fsax_raw", "goals_actual", "shots_taken"], index=0, key="sl_sort"
    )

    # Rank among the full qualified pool (position + min shots filtered), then apply team filter
    df_full = df_full.sort_values(sort_col, ascending=False).reset_index(drop=True)
    df_full["Overall Rank"] = df_full.index + 1

    if sel_teams:
        df = df_full[df_full["team"].isin(sel_teams)].copy().reset_index(drop=True)
        df["Team Rank"] = df.index + 1
    else:
        df = df_full

    rank_cols    = ["Overall Rank"] + (["Team Rank"] if sel_teams else [])
    display_cols = rank_cols + [c for c in SHOOTER_COLS if c in df.columns]
    renamed = df[display_cols].rename(columns=SHOOTER_COLS)
    st.dataframe(renamed, hide_index=True, use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# Page: Shooter Trajectory
# ─────────────────────────────────────────────────────────────────────────────

def page_shooter_trajectory(situation: str = "all") -> None:
    st.header("Shooter Skill Trajectory")
    with st.expander("ℹ️ What do these numbers mean?"):
        st.markdown(
            "- **θ (theta)** — The shooter's estimated finishing skill for a given week. "
            "θ > 0 means better than league average; θ < 0 means worse.\n"
            "- **Shaded band** — Uncertainty around the estimate. Wider bands appear in "
            "weeks with fewer shots.\n"
            "- The chart shows how a shooter's finishing ability evolves across the season.\n"
        )
    df_sum  = load_shooter_summary(situation)
    df_week = load_shooter_weekly(situation)

    # default to top-10 by fsax
    top10 = (
        df_sum.nlargest(10, "fsax")["player_name"].dropna().tolist()
        if "fsax" in df_sum.columns
        else df_sum["player_name"].dropna().head(10).tolist()
    )
    all_shooters = sorted(df_sum["player_name"].dropna().unique().tolist())
    selected = st.multiselect("Select shooters (up to 10)", all_shooters, default=top10[:5], key="straj_sel")
    if not selected:
        st.info("Select at least one shooter.")
        return

    sub = _canonical_weekly(df_week, selected, "shooter_id")
    if sub.empty:
        st.warning("No weekly data found for the selected players.")
        return

    st.altair_chart(
        trajectory_chart(
            sub, "week", "theta", "theta_lo", "theta_hi", "player_name",
            title="θ (Finishing Skill)  — higher = better finisher",
            y_label="θ",
        ),
        use_container_width=True,
    )
    st.caption(
        "θ > 0 → above-average finisher; θ < 0 → below average.  "
        "Shaded band shows uncertainty (widens in low-shot weeks)."
    )

    with st.expander("Weekly data table"):
        cols = [c for c in ["player_name", "team", "week", "theta", "mu_theta",
                            "theta_lo", "theta_hi", "shots_that_week"] if c in sub.columns]
        st.dataframe(sub[cols].sort_values(["player_name", "week"]))


# ─────────────────────────────────────────────────────────────────────────────
# Page: Head-to-Head
# ─────────────────────────────────────────────────────────────────────────────

def page_head_to_head(situation: str = "all") -> None:
    st.header("Head-to-Head Comparison")
    with st.expander("ℹ️ What do these numbers mean?"):
        st.markdown(
            "- **% value on metric cards** — How far above or below league average the player is, "
            "measured in standard deviations scaled to 100 (e.g., +100 = 1 SD above average).\n"
            "- **GSAx / FSAx** — Season-total goals saved (goalie) or scored (shooter) above expectation.\n"
            "- **Est. Goals / 100 shots** — If this shooter takes 100 average-difficulty shots against this goalie, "
            "how many would we expect to go in? Based on the full IRT model.\n"
            "- **Chart** — Both players' weekly skills on a common percent-above-average scale. "
            "Whichever player is higher on a given week has the advantage.\n"
        )
    st.markdown("Compare a **goalie** vs a **shooter** using their IRT skill tracks.")

    g_sum   = load_goalie_summary(situation)
    g_week  = load_goalie_weekly(situation)
    s_sum   = load_shooter_summary(situation)
    s_week  = load_shooter_weekly(situation)

    col1, col2 = st.columns(2)
    with col1:
        all_g = sorted(g_sum["player_name"].dropna().unique().tolist())
        sel_g = st.selectbox("Goalie", all_g, key="h2h_g")
    with col2:
        all_s = sorted(s_sum["player_name"].dropna().unique().tolist())
        sel_s = st.selectbox("Shooter", all_s, key="h2h_s")

    g_row = g_sum[g_sum["player_name"] == sel_g].iloc[0] if not g_sum[g_sum["player_name"] == sel_g].empty else None
    s_row = s_sum[s_sum["player_name"] == sel_s].iloc[0] if not s_sum[s_sum["player_name"] == sel_s].empty else None

    # ── Season-wide normalisation & per-player season means ───────────────────
    import math

    # Use the canonical (most-shots) weekly rows for each player
    _g_wk = _canonical_weekly(g_week, [sel_g], "goalie_id")
    _s_wk = _canonical_weekly(s_week, [sel_s], "shooter_id")
    _s_sm = s_sum[s_sum["player_name"] == sel_s]

    # Current-form skill = last week's estimate for the canonical player
    g_phi_latest   = float(_g_wk.sort_values("week")["phi"].iloc[-1])   if not _g_wk.empty else 0.0
    s_theta_latest = float(_s_wk.sort_values("week")["theta"].iloc[-1]) if not _s_wk.empty else 0.0

    # Normalise against the distribution of ALL players' season-end skills
    phi_ends   = g_week.groupby("goalie_id")["phi"].last()
    theta_ends = s_week.groupby("shooter_id")["theta"].last()
    phi_mean   = float(phi_ends.mean());   phi_std   = max(float(phi_ends.std()),   1e-6)
    theta_mean = float(theta_ends.mean()); theta_std = max(float(theta_ends.std()), 1e-6)

    def _norm_phi(v):   return (v - phi_mean)   / phi_std   * 100
    def _norm_theta(v): return (v - theta_mean) / theta_std * 100

    g_pct = _norm_phi(g_phi_latest)
    s_pct = _norm_theta(s_theta_latest)

    s_mu_theta = s_theta_latest  # used below for goals-per-100
    g_mu_phi   = g_phi_latest

    # Est. goals per 100 shots using the IRT model's calibrated baseline:
    #   p = σ(β₀ + α·mean(xG_logit) + θ − φ)
    meta = load_meta(situation)
    beta0         = meta.get("beta0", -0.25)
    alpha         = meta.get("alpha", 1.0)
    mean_xg_logit = meta.get("mean_xg_logit", -2.69)  # fallback: ~NHL average xG logit

    base_logit           = beta0 + alpha * mean_xg_logit
    p_goal               = 1.0 / (1.0 + math.exp(-(base_logit + s_mu_theta - g_mu_phi)))
    goals_per_100        = p_goal * 100
    league_goals_per_100 = 1.0 / (1.0 + math.exp(-base_logit)) * 100

    # ── Summary cards ─────────────────────────────────────────────────────────
    c1, c2, c3 = st.columns(3)
    if g_row is not None:
        with c1:
            st.metric(
                label=f"🥅 {sel_g} ({g_row.get('team', '')})",
                value=f"{g_pct:+.1f}%",
                delta=f"GSAx {fmt_signed(g_row['gsax'])}  (raw {fmt_signed(g_row['gsax_raw'])})",
            )
    if s_row is not None:
        with c2:
            st.metric(
                label=f"🏒 {sel_s} ({s_row.get('team', '')})",
                value=f"{s_pct:+.1f}%",
                delta=f"FSAx {fmt_signed(s_row['fsax'])}  (raw {fmt_signed(s_row.get('fsax_raw', 0))})",
            )
    with c3:
        st.metric(
            label="⚡ Est. Goals / 100 shots",
            value=f"{goals_per_100:.1f}",
            delta=f"{goals_per_100 - league_goals_per_100:+.1f} vs league avg ({league_goals_per_100:.1f})",
        )

    g_traj = _canonical_weekly(g_week, [sel_g], "goalie_id")[["week", "phi", "phi_lo", "phi_hi", "shots_that_week"]].copy()
    g_traj["skill"] = _norm_phi(g_traj["phi"])
    g_traj["lo"]    = _norm_phi(g_traj["phi_lo"])
    g_traj["hi"]    = _norm_phi(g_traj["phi_hi"])
    g_traj["label"] = f"{sel_g} (G)"

    s_traj = _canonical_weekly(s_week, [sel_s], "shooter_id")[["week", "theta", "theta_lo", "theta_hi", "shots_that_week"]].copy()
    s_traj["skill"] = _norm_theta(s_traj["theta"])
    s_traj["lo"]    = _norm_theta(s_traj["theta_lo"])
    s_traj["hi"]    = _norm_theta(s_traj["theta_hi"])
    s_traj["label"] = f"{sel_s} (S)"

    combined = pd.concat([g_traj, s_traj], ignore_index=True)
    if combined.empty:
        st.warning("No weekly data for one or both players.")
        return

    # Clamp bands to skill-line range so wide low-shot bands don't shrink the view
    y_min = float(combined["skill"].min())
    y_max = float(combined["skill"].max())
    pad   = max((y_max - y_min) * 0.25, 5.0)
    dom_lo, dom_hi = y_min - pad, y_max + pad
    combined["lo"] = combined["lo"].clip(lower=dom_lo)
    combined["hi"] = combined["hi"].clip(upper=dom_hi)
    y_scale = alt.Scale(domain=[dom_lo, dom_hi])

    tooltip = [
        alt.Tooltip("label:N",            title="Player"),
        alt.Tooltip("week:Q",             title="Week"),
        alt.Tooltip("skill:Q",            title="% above avg", format=".1f"),
        alt.Tooltip("shots_that_week:Q",  title="Shots this week"),
    ]

    zero_line = (
        alt.Chart(pd.DataFrame({"y": [0]}))
        .mark_rule(color="grey", strokeDash=[4, 4], opacity=0.6)
        .encode(y=alt.Y("y:Q", scale=y_scale))
    )
    band = (
        alt.Chart(combined)
        .mark_area(opacity=0.15)
        .encode(
            x=alt.X("week:Q", title="Week"),
            y=alt.Y("lo:Q",   title="", scale=y_scale),
            y2=alt.Y2("hi:Q"),
            color=alt.Color("label:N", legend=alt.Legend(title="Player")),
            tooltip=tooltip,
        )
    )
    line = (
        alt.Chart(combined)
        .mark_line(strokeWidth=2.5, point=alt.OverlayMarkDef(size=30))
        .encode(
            x=alt.X("week:Q"),
            y=alt.Y("skill:Q", title="% above league avg  (1 SD = 100)", scale=y_scale),
            color=alt.Color("label:N", legend=alt.Legend(title="Player")),
            tooltip=tooltip,
        )
    )
    st.altair_chart(
        (zero_line + band + line).properties(
            title=f"{sel_g}  vs  {sel_s}",
            height=420,
            width="container",
        ),
        use_container_width=True,
    )
    st.caption(
        "Both curves are expressed as % above the league-average for their role "
        "(goalie save skill φ, shooter finishing skill θ).  "
        "1 SD above average = +100.  Dashed line = league average.  "
        "Advantage goes to whichever player is higher on any given week."
    )


# ─────────────────────────────────────────────────────────────────────────────
# Page: Methodology
# ─────────────────────────────────────────────────────────────────────────────

def page_methodology(situation: str = "all") -> None:
    st.header("Methodology")

    # ── Non-technical overview ────────────────────────────────────────────────
    st.subheader("Overview (Non-Technical)")
    st.markdown("""
Not all goals (or saves) are created equal. A goal against an elite goalie is harder to score
than one against a backup; a save against a top shooter is harder to make than one against a
fourth-liner. Traditional stats like save percentage ignore this: they treat every shot the same.

This dashboard uses a statistical model called **Item Response Theory (IRT)**, borrowed from
educational testing, to disentangle goalie skill from shooter skill. The model observes every
shot in the NHL season, knows the pre-shot expected-goal probability (from a separate xG model),
and estimates two hidden quantities each week:

- **φ (phi)** for each goalie — their save skill above or below league average.
- **θ (theta)** for each shooter — their finishing skill above or below league average.

A goalie with φ > 0 stops more pucks than a league-average goalie facing the same shots;
a shooter with θ > 0 scores more than average against the same goalies. These abilities are
allowed to change week-by-week, so the model captures hot streaks, slumps, and genuine
improvement.

From these skill estimates we compute **GSAx (IRT)** for goalies and **FSAx (IRT)** for
shooters: goals saved or scored above expectation after adjusting for opponent quality.
The **Head-to-Head** page lets you match any goalie against any shooter and estimate how
many goals per 100 shots the shooter would score.

Results can be viewed for **all situations**, **even-strength (5v5)** only, or **power play** only.
""")

    # ── Technical description ─────────────────────────────────────────────────
    st.subheader("Technical Description")
    st.markdown("""
**Model structure.** The core is a dynamic IRT model estimated via MAP (maximum a posteriori)
using PyTorch with Adam + L-BFGS optimisation.

For each shot *i* in season *s*, week *t*, the probability of a goal is:

$$\\text{logit}(p_i) = \\beta_0 + \\alpha \\cdot \\text{logit}(xG_i) + \\theta_{j,s,t} - \\phi_{k,s,t}$$

where:
- $\\beta_0$ is a global intercept that recalibrates the xG baseline,
- $\\alpha$ scales the pre-shot xG log-odds (typically ≈ 1),
- $\\theta_{j,s,t}$ is the finishing skill of shooter *j* in season *s*, week *t*,
- $\\phi_{k,s,t}$ is the save skill of goalie *k* in season *s*, week *t*.

**Priors and dynamics.** Weekly skills follow a Gaussian random walk:

$$\\theta_{j,s,t+1} \\sim \\mathcal{N}(\\theta_{j,s,t},\\; \\tau_\\theta^2)$$

$$\\phi_{k,s,t+1} \\sim \\mathcal{N}(\\phi_{k,s,t},\\; \\tau_\\phi^2)$$

The random-walk standard deviations $\\tau_\\theta, \\tau_\\phi$ are learned during fitting
(clamped to a minimum to prevent collapse). Season-level means $\\mu_\\theta, \\mu_\\phi$ have
a sum-to-zero soft constraint to keep the baseline identifiable.

**Derived metrics.**
- *GSAx (IRT)*: For each goalie, GSAx = Σ (p_no_goalie − p_full) over all shots faced,
  where p_no_goalie sets the goalie skill to zero (league average) and p_full uses the
  goalie's actual estimated φ. This isolates the goalie's contribution relative to a
  replacement-level baseline.
- *FSAx (IRT)*: Analogous for shooters — the difference between the full model prediction
  and a prediction with the shooter set to league average (θ = 0).
- *Confidence bands*: Approximate ±2τ / √(shots_that_week) bands around the weekly point
  estimate, using the model's learned random-walk τ.

**Estimation procedure.** 5 000 epochs of Adam (lr = 0.01) followed by up to 20 L-BFGS
line-search steps for fine-tuning. If L-BFGS produces NaN (rare edge case), the optimizer
rolls back to the pre-L-BFGS state.

**Situation filtering.** Three separate models are fitted using subsets of the play-by-play
data filtered by on-ice strength state: all situations, even-strength (5v5), and power play.
""")

    meta = load_meta(situation)
    if meta:
        st.subheader("Current Model Parameters")
        st.markdown(
            f"- β₀ = {meta.get('beta0', '?')}  \n"
            f"- α = {meta.get('alpha', '?')}  \n"
            f"- Mean xG logit = {meta.get('mean_xg_logit', '?')}  \n"
            f"- Season: {meta.get('season', '?')}  \n"
            f"- Situation: {meta.get('situation_label', '?')}  \n"
            f"- Goalies: {meta.get('n_goalies', '?')}  |  Shooters: {meta.get('n_shooters', '?')}"
        )

    # ── Attribution ───────────────────────────────────────────────────────────
    st.subheader("Attribution")
    st.markdown(
        "Pre-shot expected goal (xG) probabilities are provided by the "
        "[**Statsyuk xGoals Model**](https://github.com/tannermanett/Statsyuk-xGoals-Model) "
        "by [tannermanett](https://github.com/tannermanett), "
        "an XGBoost-based pipeline trained on NHL shot-event data.  \n"
        "The IRT model in this dashboard re-calibrates the xG log-odds (via β₀ and α) "
        "and uses the resulting probabilities as shot-quality inputs."
    )


# ─────────────────────────────────────────────────────────────────────────────
# Main layout
# ─────────────────────────────────────────────────────────────────────────────

PAGE_ICONS = {
    "Goalie Leaderboard":   "🥅",
    "Goalie Trajectory":    "📈",
    "Shooter Leaderboard":  "🏒",
    "Shooter Trajectory":   "📈",
    "Head-to-Head":         "⚔️",
}

PAGES = {
    "🥅 Goalie Leaderboard":  page_goalie_leaderboard,
    "📈 Goalie Trajectory":   page_goalie_trajectory,
    "🏒 Shooter Leaderboard": page_shooter_leaderboard,
    "📈 Shooter Trajectory":  page_shooter_trajectory,
    "⚔️ Head-to-Head":        page_head_to_head,
    "📖 Methodology":         page_methodology,
}


def main() -> None:
    st.set_page_config(
        page_title="NHL IRT Analytics",
        page_icon="🏒",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # ── Sidebar ───────────────────────────────────────────────────────────────
    st.sidebar.title("🏒 NHL IRT Analytics")

    situation = st.sidebar.radio(
        "Situation",
        options=list(SITUATION_LABELS.keys()),
        format_func=lambda k: SITUATION_LABELS[k],
        index=0,
        key="situation",
    )

    meta = load_meta(situation)
    if meta:
        st.sidebar.caption(
            f"Season: **{meta.get('season', '?')}**  \n"
            f"Updated: {meta.get('last_updated', '?')}  \n"
            f"Situation: **{meta.get('situation_label', SITUATION_LABELS[situation])}**"
        )

    page_name = st.sidebar.radio("Page", list(PAGES.keys()))

    if not (_data_dir(situation) / "goalie_summary.csv").exists():
        st.warning(
            f"No data found for **{SITUATION_LABELS[situation]}**. "
            f"Run `python dashboard/build_data.py --situation {situation}` "
            "to generate it, then click **Refresh data cache** below."
        )
        st.stop()

    PAGES[page_name](situation=situation)

    # ── Footer ────────────────────────────────────────────────────────────────
    st.sidebar.markdown("---")
    if st.sidebar.button("🔄 Refresh data cache"):
        st.cache_data.clear()
        st.rerun()
    st.sidebar.caption(
        "Model: Dynamic IRT (PyTorch MAP)  \n"
        "Source: NHL play-by-play + Statsyuk xG"
    )


if __name__ == "__main__":
    main()
