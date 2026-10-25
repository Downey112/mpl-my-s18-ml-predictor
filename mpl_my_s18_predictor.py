"""
MPL Malaysia Season 18 (MPL MY S18) — Playoffs & Grand Final ML Predictor
=========================================================================
End-to-end Machine Learning & Monte Carlo Tournament Simulation Pipeline
Data Source: Official MPL Malaysia S18 Statistics & Match Schedule
(https://my.mpl.mobilelegends.com & Liquipedia MPL Malaysia Season 18)

Features Included:
1. Complete 56-match Double Round-Robin Regular Season dataset (Weeks 1-8).
2. Official 8-team macro telemetry (Kills, Deaths, Assists, Gold, Damage,
   Lord, Turtle, Tower across all regular season games).
3. Chronological Elo rating tracker (series score 2-0 / 2-1 / 1-2 / 0-2 mapped
   to 1.0 / 0.75 / 0.25 / 0.0) + Bradley-Terry Maximum Likelihood team skill
   estimation on game-level results.
4. L2-Regularized Logistic Regression Classifier (implemented via exact
   Newton-Raphson / IRLS in NumPy) plus a Gradient-Boosted Decision Stump
   Ensemble, both trained on Bo3 match outcomes and evaluated with 8-fold
   Leave-One-Week-Out Cross-Validation. The match-level probability is
   rescaled to a single-game probability p(A beats B) and blended with
   Bradley-Terry, Elo and head-to-head estimates.
5. Exact Negative-Binomial Bo5 / Bo7 series probability conversion and a
   50,000-iteration Monte Carlo simulation of the MPL MY S18 Hybrid
   Elimination Playoffs bracket (Oct 28 – Nov 1, 2026) to predict M8 World
   Championship qualification (Top 2) and the Season 18 Grand Final Champion.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

REPO_DIR = Path(__file__).resolve().parent
DATA_DIR = REPO_DIR / "data"
ASSETS_DIR = REPO_DIR / "assets"


# ---------------------------------------------------------------------------
# 1. Official MPL Malaysia S18 Team Macro Statistics (Regular Season)
# Source: https://my.mpl.mobilelegends.com
# ---------------------------------------------------------------------------
TEAM_STATS_RAW = [
    {
        "team": "SRG",
        "full_name": "Selangor Red Giants",
        "seed": 1,
        "match_w": 11,
        "match_l": 3,
        "game_w": 24,
        "game_l": 8,
        "match_pts": 11,
        "agg_pts": 16,
        "kills": 447,
        "deaths": 296,
        "assists": 1126,
        "gold": 1_519_682,
        "damage": 6_722_690,
        "lord": 43,
        "turtle": 51,
        "tower": 219,
    },
    {
        "team": "TR",
        "full_name": "Team Rey",
        "seed": 2,
        "match_w": 10,
        "match_l": 4,
        "game_w": 22,
        "game_l": 11,
        "match_pts": 10,
        "agg_pts": 11,
        "kills": 363,
        "deaths": 325,
        "assists": 896,
        "gold": 1_736_666,
        "damage": 7_788_998,
        "lord": 43,
        "turtle": 48,
        "tower": 207,
    },
    {
        "team": "VMS",
        "full_name": "Team Vamos",
        "seed": 3,
        "match_w": 10,
        "match_l": 4,
        "game_w": 22,
        "game_l": 14,
        "match_pts": 10,
        "agg_pts": 8,
        "kills": 446,
        "deaths": 349,
        "assists": 1125,
        "gold": 1_909_071,
        "damage": 8_589_569,
        "lord": 53,
        "turtle": 61,
        "tower": 234,
    },
    {
        "team": "iG",
        "full_name": "Invictus Gaming",
        "seed": 4,
        "match_w": 7,
        "match_l": 7,
        "game_w": 17,
        "game_l": 17,
        "match_pts": 7,
        "agg_pts": 0,
        "kills": 359,
        "deaths": 386,
        "assists": 912,
        "gold": 1_670_610,
        "damage": 7_609_743,
        "lord": 38,
        "turtle": 48,
        "tower": 178,
    },
    {
        "team": "BTRM",
        "full_name": "Bigetron MY by VIT",
        "seed": 5,
        "match_w": 6,
        "match_l": 8,
        "game_w": 17,
        "game_l": 19,
        "match_pts": 6,
        "agg_pts": -2,
        "kills": 401,
        "deaths": 476,
        "assists": 1018,
        "gold": 1_873_753,
        "damage": 8_012_647,
        "lord": 40,
        "turtle": 61,
        "tower": 189,
    },
    {
        "team": "FL",
        "full_name": "Team Flash",
        "seed": 6,
        "match_w": 5,
        "match_l": 9,
        "game_w": 13,
        "game_l": 19,
        "match_pts": 5,
        "agg_pts": -6,
        "kills": 375,
        "deaths": 343,
        "assists": 950,
        "gold": 1_634_493,
        "damage": 7_367_943,
        "lord": 30,
        "turtle": 44,
        "tower": 167,
    },
    {
        "team": "AC",
        "full_name": "AC Esports",
        "seed": 7,
        "match_w": 4,
        "match_l": 10,
        "game_w": 10,
        "game_l": 24,
        "match_pts": 4,
        "agg_pts": -14,
        "kills": 354,
        "deaths": 462,
        "assists": 899,
        "gold": 1_765_400,
        "damage": 7_453_081,
        "lord": 34,
        "turtle": 39,
        "tower": 143,
    },
    {
        "team": "RRQ",
        "full_name": "RRQ Tora",
        "seed": 8,
        "match_w": 3,
        "match_l": 11,
        "game_w": 11,
        "game_l": 24,
        "match_pts": 3,
        "agg_pts": -13,
        "kills": 316,
        "deaths": 426,
        "assists": 808,
        "gold": 1_822_857,
        "damage": 8_270_913,
        "lord": 38,
        "turtle": 49,
        "tower": 165,
    },
]

# ---------------------------------------------------------------------------
# 2. Complete 56-Match MPL Malaysia S18 Regular Season Schedule & Scores
# Format: (week, date_str, team_a, team_b, score_a, score_b)
# Source: Official team match logs at https://my.mpl.mobilelegends.com/team/*
# ---------------------------------------------------------------------------
REGULAR_SEASON_MATCHES: List[Tuple[int, str, str, str, int, int]] = [
    # Week 1 (Aug 14 - Aug 16, 2026)
    (1, "2026-08-14", "AC", "BTRM", 1, 2),
    (1, "2026-08-14", "VMS", "FL", 2, 1),
    (1, "2026-08-15", "FL", "BTRM", 1, 2),
    (1, "2026-08-15", "RRQ", "TR", 0, 2),
    (1, "2026-08-15", "SRG", "iG", 2, 0),
    (1, "2026-08-16", "iG", "RRQ", 2, 0),
    (1, "2026-08-16", "TR", "SRG", 0, 2),
    # Week 2 (Aug 21 - Aug 23, 2026)
    (2, "2026-08-21", "iG", "VMS", 2, 1),
    (2, "2026-08-21", "BTRM", "SRG", 1, 2),
    (2, "2026-08-22", "TR", "VMS", 1, 2),
    (2, "2026-08-22", "AC", "iG", 0, 2),
    (2, "2026-08-22", "BTRM", "RRQ", 2, 0),
    (2, "2026-08-23", "TR", "AC", 2, 0),
    (2, "2026-08-23", "FL", "SRG", 0, 2),
    # Week 3 (Aug 28 - Aug 30, 2026)
    (3, "2026-08-28", "BTRM", "TR", 1, 2),
    (3, "2026-08-28", "VMS", "RRQ", 2, 1),
    (3, "2026-08-29", "FL", "TR", 0, 2),
    (3, "2026-08-29", "BTRM", "VMS", 1, 2),
    (3, "2026-08-29", "SRG", "AC", 2, 0),
    (3, "2026-08-30", "iG", "FL", 0, 2),
    (3, "2026-08-30", "RRQ", "AC", 1, 2),
    # Week 4 (Sep 4 - Sep 6, 2026)
    (4, "2026-09-04", "AC", "FL", 0, 2),
    (4, "2026-09-04", "RRQ", "SRG", 2, 1),
    (4, "2026-09-05", "FL", "RRQ", 2, 0),
    (4, "2026-09-05", "TR", "iG", 2, 0),
    (4, "2026-09-05", "SRG", "VMS", 2, 0),
    (4, "2026-09-06", "iG", "BTRM", 2, 1),
    (4, "2026-09-06", "VMS", "AC", 2, 0),
    # Week 5 (Sep 11 - Sep 13, 2026)
    (5, "2026-09-11", "iG", "TR", 2, 0),
    (5, "2026-09-11", "SRG", "BTRM", 2, 0),
    (5, "2026-09-12", "FL", "AC", 2, 0),
    (5, "2026-09-12", "VMS", "TR", 0, 2),
    (5, "2026-09-12", "SRG", "RRQ", 2, 1),
    (5, "2026-09-13", "BTRM", "AC", 2, 0),
    (5, "2026-09-13", "RRQ", "FL", 2, 0),
    # Week 6 (Sep 18 - Sep 20, 2026)
    (6, "2026-09-18", "TR", "FL", 2, 1),
    (6, "2026-09-18", "RRQ", "VMS", 1, 2),
    (6, "2026-09-19", "VMS", "BTRM", 2, 0),
    (6, "2026-09-19", "RRQ", "iG", 1, 2),
    (6, "2026-09-19", "AC", "SRG", 0, 2),
    (6, "2026-09-20", "BTRM", "iG", 2, 1),
    (6, "2026-09-20", "SRG", "TR", 1, 2),
    # Week 7 (Oct 8 - Oct 11, 2026)
    (7, "2026-10-08", "TR", "BTRM", 2, 0),
    (7, "2026-10-08", "SRG", "FL", 2, 0),
    (7, "2026-10-09", "VMS", "iG", 2, 1),
    (7, "2026-10-09", "AC", "RRQ", 1, 2),
    (7, "2026-10-11", "FL", "VMS", 0, 2),
    (7, "2026-10-11", "iG", "AC", 1, 2),
    (7, "2026-10-11", "TR", "RRQ", 2, 0),
    # Week 8 (Oct 16 - Oct 18, 2026)
    (8, "2026-10-16", "FL", "iG", 0, 2),
    (8, "2026-10-16", "VMS", "SRG", 2, 0),
    (8, "2026-10-17", "BTRM", "FL", 1, 2),
    (8, "2026-10-17", "AC", "TR", 2, 1),
    (8, "2026-10-17", "iG", "SRG", 0, 2),
    (8, "2026-10-18", "AC", "VMS", 2, 1),
    (8, "2026-10-18", "RRQ", "BTRM", 0, 2),
]


def sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30.0, 30.0)))


class L2LogisticRegression:
    """Exact L2-Regularized Logistic Regression trained via Newton-Raphson (IRLS)."""

    def __init__(self, l2_penalty: float = 2.0, max_iter: int = 100, tol: float = 1e-6):
        self.l2_penalty = l2_penalty
        self.max_iter = max_iter
        self.tol = tol
        self.coef_: np.ndarray | None = None
        self.intercept_: float = 0.0

    def fit(self, X: np.ndarray, y: np.ndarray) -> "L2LogisticRegression":
        n_samples, n_features = X.shape
        X_aug = np.hstack([np.ones((n_samples, 1)), X])
        w = np.zeros(n_features + 1)
        reg_matrix = np.eye(n_features + 1) * self.l2_penalty
        reg_matrix[0, 0] = 0.0  # Do not regularize intercept

        for _ in range(self.max_iter):
            p = sigmoid(X_aug @ w)
            grad = X_aug.T @ (p - y) + reg_matrix @ w
            W_diag = np.clip(p * (1.0 - p), 1e-6, None)
            H = (X_aug.T * W_diag) @ X_aug + reg_matrix
            step = np.linalg.solve(H, grad)
            w -= step
            if np.linalg.norm(step) < self.tol:
                break

        self.intercept_ = float(w[0])
        self.coef_ = w[1:]
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        assert self.coef_ is not None
        p1 = sigmoid(self.intercept_ + X @ self.coef_)
        return np.vstack([1.0 - p1, p1]).T


class GradientBoostedStumpEnsemble:
    """
    Non-linear Gradient-Boosted Decision Stump Ensemble in pure NumPy.
    Captures non-linear thresholds in macro differentials (e.g. KDA + Tower diff).
    """

    def __init__(self, n_estimators: int = 80, learning_rate: float = 0.06):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.stumps: List[Tuple[int, float, float, float]] = []
        self.base_log_odds: float = 0.0

    def fit(self, X: np.ndarray, y: np.ndarray) -> "GradientBoostedStumpEnsemble":
        p_mean = np.clip(np.mean(y), 1e-4, 1.0 - 1e-4)
        self.base_log_odds = float(np.log(p_mean / (1.0 - p_mean)))
        f_pred = np.full(len(y), self.base_log_odds)
        self.stumps = []

        n_samples, n_features = X.shape
        for _ in range(self.n_estimators):
            p = sigmoid(f_pred)
            residuals = y - p
            best_loss = float("inf")
            best_stump = (0, 0.0, 0.0, 0.0)

            for j in range(n_features):
                col = X[:, j]
                thresholds = np.unique(np.percentile(col, [20, 35, 50, 65, 80]))
                for thr in thresholds:
                    left_mask = col <= thr
                    right_mask = ~left_mask
                    if np.sum(left_mask) < 4 or np.sum(right_mask) < 4:
                        continue
                    val_left = float(np.mean(residuals[left_mask]))
                    val_right = float(np.mean(residuals[right_mask]))
                    pred_res = np.where(left_mask, val_left, val_right)
                    loss = float(np.mean((residuals - pred_res) ** 2))
                    if loss < best_loss:
                        best_loss = loss
                        best_stump = (j, float(thr), val_left, val_right)

            j, thr, vl, vr = best_stump
            self.stumps.append(best_stump)
            f_pred += self.learning_rate * np.where(X[:, j] <= thr, vl, vr)

        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        f_pred = np.full(X.shape[0], self.base_log_odds)
        for j, thr, vl, vr in self.stumps:
            f_pred += self.learning_rate * np.where(X[:, j] <= thr, vl, vr)
        p1 = sigmoid(f_pred)
        return np.vstack([1.0 - p1, p1]).T


def fit_bradley_terry_game_ratings(teams: List[str]) -> Dict[str, float]:
    """
    Estimate Maximum-Likelihood Bradley-Terry skill ratings on game-level results
    across all 56 regular season matches using MM (Minorization-Maximization) iterations.
    """
    idx = {t: i for i, t in enumerate(teams)}
    n = len(teams)
    # Weak prior for stability: 0.5 pseudo-win per team and 0.15 pseudo-games per pair
    wins = np.zeros(n) + 0.5
    games_matrix = np.zeros((n, n)) + 0.15
    np.fill_diagonal(games_matrix, 0.0)

    for _, _, ta, tb, sa, sb in REGULAR_SEASON_MATCHES:
        i, j = idx[ta], idx[tb]
        wins[i] += sa
        wins[j] += sb
        games_matrix[i, j] += sa + sb
        games_matrix[j, i] += sa + sb

    strength = np.ones(n)
    for _ in range(200):
        new_strength = np.zeros(n)
        for i in range(n):
            denom = sum(
                games_matrix[i, j] / (strength[i] + strength[j]) for j in range(n) if i != j
            )
            new_strength[i] = wins[i] / denom
        # Normalize geometric mean to 1.0
        strength = new_strength / np.exp(np.mean(np.log(new_strength)))

    return {t: float(strength[idx[t]]) for t in teams}


def build_feature_table() -> pd.DataFrame:
    """Compute per-game normalized macro metrics, chronological Elo, and Bradley-Terry skill."""
    df_stats = pd.DataFrame(TEAM_STATS_RAW).set_index("team")
    df_stats["games_played"] = df_stats["game_w"] + df_stats["game_l"]
    df_stats["game_win_rate"] = df_stats["game_w"] / df_stats["games_played"]
    df_stats["kda_ratio"] = (df_stats["kills"] + df_stats["assists"]) / df_stats["deaths"]
    df_stats["kills_per_game"] = df_stats["kills"] / df_stats["games_played"]
    df_stats["deaths_per_game"] = df_stats["deaths"] / df_stats["games_played"]
    df_stats["gold_per_game"] = df_stats["gold"] / df_stats["games_played"]
    df_stats["damage_per_game"] = df_stats["damage"] / df_stats["games_played"]
    df_stats["lord_per_game"] = df_stats["lord"] / df_stats["games_played"]
    df_stats["turtle_per_game"] = df_stats["turtle"] / df_stats["games_played"]
    df_stats["tower_per_game"] = df_stats["tower"] / df_stats["games_played"]

    # Chronological Elo across the 56 regular season matches
    elo: Dict[str, float] = {t: 1500.0 for t in df_stats.index}
    k_base = 32.0
    for week, _, team_a, team_b, sa, sb in REGULAR_SEASON_MATCHES:
        k_week = k_base * (1.0 + 0.05 * (week - 1))
        exp_a = 1.0 / (1.0 + 10.0 ** ((elo[team_b] - elo[team_a]) / 400.0))
        actual_a = 1.0 if (sa == 2 and sb == 0) else (0.75 if sa > sb else (0.25 if sa == 1 else 0.0))
        delta = k_week * (actual_a - exp_a)
        elo[team_a] += delta
        elo[team_b] -= delta

    df_stats["final_elo"] = pd.Series(elo)
    bt_ratings = fit_bradley_terry_game_ratings(list(df_stats.index))
    df_stats["bt_strength"] = pd.Series(bt_ratings)
    return df_stats


def build_training_dataset(df_stats: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """
    Construct symmetric match-level training examples (112 directed rows from 56 matches)
    with pre-match dynamic Elo and macro efficiency differentials.
    """
    feature_cols = [
        "pre_elo_diff",
        "kda_diff",
        "kills_pg_diff",
        "deaths_pg_diff",
        "lord_pg_diff",
        "turtle_pg_diff",
        "tower_pg_diff",
        "gold_pg_diff",
    ]
    rows = []
    elo: Dict[str, float] = {t: 1500.0 for t in df_stats.index}

    for match_id, (week, date_str, ta, tb, sa, sb) in enumerate(REGULAR_SEASON_MATCHES, 1):
        for t1, t2, s1, s2 in [(ta, tb, sa, sb), (tb, ta, sb, sa)]:
            st1, st2 = df_stats.loc[t1], df_stats.loc[t2]
            rows.append(
                {
                    "match_id": match_id,
                    "week": week,
                    "date": date_str,
                    "team_a": t1,
                    "team_b": t2,
                    "score_a": s1,
                    "score_b": s2,
                    "target_win": int(s1 > s2),
                    "pre_elo_diff": elo[t1] - elo[t2],
                    "kda_diff": st1["kda_ratio"] - st2["kda_ratio"],
                    "kills_pg_diff": st1["kills_per_game"] - st2["kills_per_game"],
                    "deaths_pg_diff": st1["deaths_per_game"] - st2["deaths_per_game"],
                    "lord_pg_diff": st1["lord_per_game"] - st2["lord_per_game"],
                    "turtle_pg_diff": st1["turtle_per_game"] - st2["turtle_per_game"],
                    "tower_pg_diff": st1["tower_per_game"] - st2["tower_per_game"],
                    "gold_pg_diff": (st1["gold_per_game"] - st2["gold_per_game"]) / 1000.0,
                }
            )
        k_week = 32.0 * (1.0 + 0.05 * (week - 1))
        exp_a = 1.0 / (1.0 + 10.0 ** ((elo[tb] - elo[ta]) / 400.0))
        actual_a = 1.0 if (sa == 2 and sb == 0) else (0.75 if sa > sb else (0.25 if sa == 1 else 0.0))
        delta = k_week * (actual_a - exp_a)
        elo[ta] += delta
        elo[tb] -= delta

    return pd.DataFrame(rows), feature_cols


def compute_h2h_game_winrate(team_a: str, team_b: str) -> Tuple[int, int, int, int, float]:
    """Return (match_w_a, match_w_b, games_a, games_b, shrunken_h2h_game_wr) from Regular Season."""
    m_a = m_b = g_a = g_b = 0
    for _, _, ta, tb, sa, sb in REGULAR_SEASON_MATCHES:
        if ta == team_a and tb == team_b:
            g_a += sa
            g_b += sb
            m_a += int(sa > sb)
            m_b += int(sb > sa)
        elif ta == team_b and tb == team_a:
            g_a += sb
            g_b += sa
            m_a += int(sb > sa)
            m_b += int(sa > sb)
    # Empirical Bayes shrinkage toward 0.50 with prior strength = 4 games
    h2h_wr = (g_a + 2.0) / (g_a + g_b + 4.0)
    return m_a, m_b, g_a, g_b, h2h_wr


def series_win_prob(p_game: float, best_of: int) -> Tuple[float, Dict[str, float]]:
    """
    Compute exact series win probability and exact scoreline distribution
    for a Best-of-N series (Bo3, Bo5, or Bo7) given single-game win probability p_game.
    """
    wins_needed = (best_of // 2) + 1
    q_game = 1.0 - p_game
    total_win_prob = 0.0
    score_dist: Dict[str, float] = {}

    for losses in range(wins_needed):
        ways = math.comb(wins_needed + losses - 1, losses)
        prob_a = ways * (p_game ** wins_needed) * (q_game ** losses)
        total_win_prob += prob_a
        score_dist[f"{wins_needed}-{losses}"] = prob_a

    for wins_a in range(wins_needed):
        ways = math.comb(wins_needed + wins_a - 1, wins_a)
        prob_b = ways * (q_game ** wins_needed) * (p_game ** wins_a)
        score_dist[f"{wins_a}-{wins_needed}"] = prob_b

    return total_win_prob, score_dist


def compute_roc_auc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    pos = y_prob[y_true == 1]
    neg = y_prob[y_true == 0]
    comparisons = (pos[:, None] > neg[None, :]).astype(float) + 0.5 * (pos[:, None] == neg[None, :])
    return float(np.mean(comparisons))


def train_and_evaluate_models(
    df_train: pd.DataFrame, feature_cols: List[str]
) -> Tuple[np.ndarray, np.ndarray, L2LogisticRegression, GradientBoostedStumpEnsemble]:
    """Evaluate via 8-Fold Leave-One-Week-Out Cross-Validation and fit final models."""
    X = df_train[feature_cols].values
    y = df_train["target_win"].values
    weeks = df_train["week"].values

    p_lr_oof = np.zeros(len(y))
    p_gb_oof = np.zeros(len(y))

    for w in range(1, 9):
        val_mask = weeks == w
        tr_mask = ~val_mask
        mu = np.mean(X[tr_mask], axis=0)
        sd = np.std(X[tr_mask], axis=0) + 1e-8
        X_tr = (X[tr_mask] - mu) / sd
        X_va = (X[val_mask] - mu) / sd

        lr_fold = L2LogisticRegression(l2_penalty=2.0).fit(X_tr, y[tr_mask])
        gb_fold = GradientBoostedStumpEnsemble(n_estimators=80, learning_rate=0.06).fit(X_tr, y[tr_mask])
        p_lr_oof[val_mask] = lr_fold.predict_proba(X_va)[:, 1]
        p_gb_oof[val_mask] = gb_fold.predict_proba(X_va)[:, 1]

    p_ens_oof = 0.65 * p_lr_oof + 0.35 * p_gb_oof

    print("=== Leave-One-Week-Out Cross-Validation (56 Matches / 112 Directed Pairs) ===")
    for name, preds in [
        ("L2 Logistic Regression   ", p_lr_oof),
        ("Boosted Decision Ensemble", p_gb_oof),
        ("Blended ML Ensemble      ", p_ens_oof),
    ]:
        acc = float(np.mean((preds >= 0.5).astype(int) == y))
        auc = compute_roc_auc(y, preds)
        brier = float(np.mean((preds - y) ** 2))
        eps = np.clip(preds, 1e-6, 1.0 - 1e-6)
        ll = float(-np.mean(y * np.log(eps) + (1.0 - y) * np.log(1.0 - eps)))
        print(f"  {name}: Accuracy = {acc:.3f} | ROC-AUC = {auc:.3f} | Brier = {brier:.3f} | LogLoss = {ll:.3f}")

    mu_full = np.mean(X, axis=0)
    sd_full = np.std(X, axis=0) + 1e-8
    X_scaled = (X - mu_full) / sd_full

    lr_full = L2LogisticRegression(l2_penalty=2.0).fit(X_scaled, y)
    gb_full = GradientBoostedStumpEnsemble(n_estimators=80, learning_rate=0.06).fit(X_scaled, y)

    print("\n=== Standardized Feature Weights (L2 Logistic Regression) ===")
    for fname, weight in sorted(zip(feature_cols, lr_full.coef_), key=lambda x: abs(x[1]), reverse=True):
        print(f"  {fname:16s}: {weight:+.4f}")

    return mu_full, sd_full, lr_full, gb_full


def build_pairwise_game_prob_matrix(
    df_stats: pd.DataFrame,
    mu: np.ndarray,
    sd: np.ndarray,
    lr: L2LogisticRegression,
    gb: GradientBoostedStumpEnsemble,
) -> Dict[Tuple[str, str], float]:
    """
    Build calibrated single-game win probability p_game(A beats B) for all Playoff pairs,
    combining:
      - Trained Macro ML Ensemble (45%)
      - Game-level Bradley-Terry Maximum Likelihood Strength (30%)
      - Final Chronological Elo Expectation (15%)
      - Regular Season Head-to-Head Bayesian Prior (10%)
    """
    playoff_teams = ["SRG", "TR", "VMS", "iG", "BTRM", "FL"]
    raw_probs: Dict[Tuple[str, str], float] = {}

    for ta in playoff_teams:
        for tb in playoff_teams:
            if ta == tb:
                continue
            st1, st2 = df_stats.loc[ta], df_stats.loc[tb]
            feat_vec = np.array(
                [
                    [
                        st1["final_elo"] - st2["final_elo"],
                        st1["kda_ratio"] - st2["kda_ratio"],
                        st1["kills_per_game"] - st2["kills_per_game"],
                        st1["deaths_per_game"] - st2["deaths_per_game"],
                        st1["lord_per_game"] - st2["lord_per_game"],
                        st1["turtle_per_game"] - st2["turtle_per_game"],
                        st1["tower_per_game"] - st2["tower_per_game"],
                        (st1["gold_per_game"] - st2["gold_per_game"]) / 1000.0,
                    ]
                ]
            )
            x_sc = (feat_vec - mu) / sd
            ml_match_prob = 0.65 * lr.predict_proba(x_sc)[0, 1] + 0.35 * gb.predict_proba(x_sc)[0, 1]
            # Convert Bo3 match win probability into single-game win probability scale
            ml_game_prob = 0.5 + 0.68 * (ml_match_prob - 0.5)

            bt_prob = st1["bt_strength"] / (st1["bt_strength"] + st2["bt_strength"])
            elo_prob = 1.0 / (1.0 + 10.0 ** ((st2["final_elo"] - st1["final_elo"]) / 400.0))
            _, _, _, _, h2h_wr = compute_h2h_game_winrate(ta, tb)

            raw_probs[(ta, tb)] = (
                0.45 * ml_game_prob + 0.30 * bt_prob + 0.15 * elo_prob + 0.10 * h2h_wr
            )

    sym_probs: Dict[Tuple[str, str], float] = {}
    for ta in playoff_teams:
        for tb in playoff_teams:
            if ta == tb:
                continue
            p_ab = raw_probs[(ta, tb)]
            p_ba = raw_probs[(tb, ta)]
            sym_probs[(ta, tb)] = float(p_ab / (p_ab + p_ba))

    return sym_probs


def simulate_playoffs_monte_carlo(
    p_game_matrix: Dict[Tuple[str, str], float], n_sims: int = 50_000, seed: int = 20261028
) -> pd.DataFrame:
    """
    Simulate the exact MPL Malaysia S18 Hybrid Elimination Playoffs Bracket (50,000 runs):
      - Upper Bracket Quarterfinals (Oct 28, Bo5, Single Elimination):
          UBQF1: #3 VMS vs #6 FL
          UBQF2: #4 iG  vs #5 BTRM
      - Upper Bracket Semifinals (Oct 29, Bo5, Double Elimination starts):
          UBSF1: #2 TR  vs Winner(UBQF1)
          UBSF2: #1 SRG vs Winner(UBQF2)
      - Lower Bracket Semifinal (Oct 30, Bo5):
          LBSF:  Loser(UBSF1) vs Loser(UBSF2)
      - Upper Bracket Final (Oct 30, Bo5):
          UBF:   Winner(UBSF1) vs Winner(UBSF2)
      - Lower Bracket Final (Oct 31, Bo7):
          LBF:   Loser(UBF) vs Winner(LBSF)
      - Grand Final (Nov 1, Bo7):
          GF:    Winner(UBF) vs Winner(LBF)
    """
    rng = np.random.default_rng(seed)
    teams = ["SRG", "TR", "VMS", "iG", "BTRM", "FL"]

    counts = {
        t: {
            "reach_ubsf": 0,
            "reach_ubf": 0,
            "reach_lbf": 0,
            "reach_gf_m8": 0,
            "champion": 0,
            "place_1": 0,
            "place_2": 0,
            "place_3": 0,
            "place_4": 0,
            "place_5_6": 0,
        }
        for t in teams
    }
    gf_matchups: Dict[Tuple[str, str], int] = {}

    bo5_prob = {pair: series_win_prob(p, 5)[0] for pair, p in p_game_matrix.items()}
    bo7_prob = {pair: series_win_prob(p, 7)[0] for pair, p in p_game_matrix.items()}

    def play_series(ta: str, tb: str, is_bo7: bool = False) -> Tuple[str, str]:
        prob_a = bo7_prob[(ta, tb)] if is_bo7 else bo5_prob[(ta, tb)]
        if rng.random() < prob_a:
            return ta, tb
        return tb, ta

    for _ in range(n_sims):
        # Round 1: UB Quarterfinals (Single Elimination, Bo5)
        w_qf1, l_qf1 = play_series("VMS", "FL", is_bo7=False)
        w_qf2, l_qf2 = play_series("iG", "BTRM", is_bo7=False)
        counts[l_qf1]["place_5_6"] += 1
        counts[l_qf2]["place_5_6"] += 1

        for t in ("SRG", "TR", w_qf1, w_qf2):
            counts[t]["reach_ubsf"] += 1

        # Round 2: UB Semifinals (Bo5)
        w_sf1, l_sf1 = play_series("TR", w_qf1, is_bo7=False)
        w_sf2, l_sf2 = play_series("SRG", w_qf2, is_bo7=False)
        counts[w_sf1]["reach_ubf"] += 1
        counts[w_sf2]["reach_ubf"] += 1

        # Lower Bracket Semifinal (Bo5)
        w_lbsf, l_lbsf = play_series(l_sf1, l_sf2, is_bo7=False)
        counts[l_lbsf]["place_4"] += 1

        # Upper Bracket Final (Bo5)
        w_ubf, l_ubf = play_series(w_sf1, w_sf2, is_bo7=False)
        counts[w_ubf]["reach_gf_m8"] += 1
        counts[l_ubf]["reach_lbf"] += 1
        counts[w_lbsf]["reach_lbf"] += 1

        # Lower Bracket Final (Bo7)
        w_lbf, l_lbf = play_series(l_ubf, w_lbsf, is_bo7=True)
        counts[l_lbf]["place_3"] += 1
        counts[w_lbf]["reach_gf_m8"] += 1

        # Grand Final (Bo7)
        pair_key = tuple(sorted((w_ubf, w_lbf)))
        gf_matchups[pair_key] = gf_matchups.get(pair_key, 0) + 1
        champ, runner_up = play_series(w_ubf, w_lbf, is_bo7=True)
        counts[champ]["champion"] += 1
        counts[champ]["place_1"] += 1
        counts[runner_up]["place_2"] += 1

    rows = []
    for t in teams:
        c = counts[t]
        rows.append(
            {
                "Team": t,
                "Reach_UBSF_%": 100.0 * c["reach_ubsf"] / n_sims,
                "Reach_UBF_%": 100.0 * c["reach_ubf"] / n_sims,
                "Reach_LBF_%": 100.0 * c["reach_lbf"] / n_sims,
                "M8_Qual_GrandFinal_%": 100.0 * c["reach_gf_m8"] / n_sims,
                "Champion_%": 100.0 * c["champion"] / n_sims,
                "RunnerUp_2nd_%": 100.0 * c["place_2"] / n_sims,
                "3rd_Place_%": 100.0 * c["place_3"] / n_sims,
                "4th_Place_%": 100.0 * c["place_4"] / n_sims,
                "5th_6th_Place_%": 100.0 * c["place_5_6"] / n_sims,
            }
        )
    df_sim = pd.DataFrame(rows).sort_values("Champion_%", ascending=False)

    print("\n=== Top Grand Final Matchup Probabilities (50,000 Simulations) ===")
    for (t1, t2), cnt in sorted(gf_matchups.items(), key=lambda x: x[1], reverse=True)[:6]:
        p_t1_bo7, scores_bo7 = series_win_prob(p_game_matrix[(t1, t2)], 7)
        most_likely_score = max(scores_bo7.items(), key=lambda x: x[1])
        print(
            f"  {t1:4s} vs {t2:4s}: {100.0 * cnt / n_sims:5.2f}% of simulations | "
            f"Bo7 Win Prob: {t1} {100*p_t1_bo7:.1f}% - {100*(1-p_t1_bo7):.1f}% {t2} "
            f"(Top scoreline: {most_likely_score[0]} at {100*most_likely_score[1]:.1f}%)"
        )

    return df_sim


FEATURE_LABELS = {
    "pre_elo_diff": "Pre-Match Elo Diff",
    "kda_diff": "Team KDA Diff",
    "kills_pg_diff": "Kills / Game Diff",
    "deaths_pg_diff": "Deaths / Game Diff",
    "lord_pg_diff": "Lords / Game Diff",
    "turtle_pg_diff": "Turtles / Game Diff",
    "tower_pg_diff": "Towers / Game Diff",
    "gold_pg_diff": "Gold / Game Diff",
}


def plot_forecast(
    df_sim: pd.DataFrame, feature_cols: List[str], lr: L2LogisticRegression, out_path: Path
) -> None:
    """Two-panel chart: playoff forecast (left) and full-fit logistic weights (right)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    bg, panel, grid, text = "#0f172a", "#1e293b", "#334155", "#cbd5e1"
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8), facecolor=bg)
    for ax in (ax1, ax2):
        ax.set_facecolor(panel)
        ax.tick_params(colors=text, labelsize=13)
        for spine in ax.spines.values():
            spine.set_color(grid)
        ax.grid(axis="x", color=grid, linestyle="--", alpha=0.8)
        ax.set_axisbelow(True)

    teams = df_sim["Team"].tolist()
    y = np.arange(len(teams))
    gf = df_sim["M8_Qual_GrandFinal_%"].to_numpy()
    champ = df_sim["Champion_%"].to_numpy()
    ax1.barh(y - 0.18, gf, height=0.35, color="#38bdf8", label="Reach Grand Final / M8 Slot (%)")
    ax1.barh(y + 0.18, champ, height=0.35, color="#f59e0b", label="Win S18 Final / Champion (%)")
    for yi, (g, c) in enumerate(zip(gf, champ)):
        ax1.text(g + 1.2, yi - 0.18, f"{g:.1f}%", va="center", color="#38bdf8", fontsize=12, fontweight="bold")
        ax1.text(c + 1.2, yi + 0.18, f"{c:.1f}%", va="center", color="#f59e0b", fontsize=12, fontweight="bold")
    ax1.set_yticks(y, teams, fontsize=15, fontweight="bold")
    ax1.invert_yaxis()
    ax1.set_xlim(0, 102)
    ax1.set_xlabel("Probability (%)", color=text, fontsize=13)
    ax1.set_title("MPL MY S18 Playoffs Forecast (50,000 Simulations)", color="white", fontsize=16, fontweight="bold", pad=14)
    ax1.legend(loc="lower right", facecolor=bg, edgecolor=grid, labelcolor="white", fontsize=12)

    weights = sorted(zip(feature_cols, lr.coef_), key=lambda x: abs(x[1]), reverse=True)
    labels = [FEATURE_LABELS.get(f, f) for f, _ in weights]
    vals = np.array([w for _, w in weights])
    yw = np.arange(len(vals))
    ax2.barh(yw, vals, height=0.55, color=np.where(vals >= 0, "#10b981", "#f43f5e"))
    lim = max(1.4, float(np.abs(vals).max()) * 1.3)
    for yi, v in enumerate(vals):
        ax2.text(v + (0.05 if v >= 0 else -0.05) * lim / 1.4, yi, f"{v:+.2f}", va="center",
                 ha="left" if v >= 0 else "right", color="white", fontsize=12, fontweight="bold")
    ax2.axvline(0, color="#94a3b8", linewidth=1.2)
    ax2.set_yticks(yw, labels)
    ax2.invert_yaxis()
    ax2.set_xlim(-lim, lim)
    ax2.set_xlabel("Standardized L2 Logistic Weight", color=text, fontsize=13)
    ax2.set_title("Logistic Regression Weights (Full-Data Fit)", color="white", fontsize=16, fontweight="bold", pad=14)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=117, facecolor=bg)
    plt.close(fig)


def main() -> None:
    df_stats = build_feature_table()
    df_train, feature_cols = build_training_dataset(df_stats)

    # Export clean CSV datasets for GitHub portfolio use
    DATA_DIR.mkdir(exist_ok=True)
    df_stats.to_csv(DATA_DIR / "mpl_my_s18_team_stats.csv")
    df_train[df_train["team_a"] < df_train["team_b"]].to_csv(
        DATA_DIR / "mpl_my_s18_regular_season_matches.csv", index=False
    )

    print("=== MPL Malaysia S18 Regular Season Per-Game Macro, Elo & Bradley-Terry Table ===")
    cols_show = [
        "seed",
        "match_w",
        "match_l",
        "game_w",
        "game_l",
        "kda_ratio",
        "kills_per_game",
        "lord_per_game",
        "tower_per_game",
        "final_elo",
        "bt_strength",
    ]
    print(df_stats[cols_show].round(2).to_string())
    print()

    mu, sd, lr, gb = train_and_evaluate_models(df_train, feature_cols)
    p_game_matrix = build_pairwise_game_prob_matrix(df_stats, mu, sd, lr, gb)

    print("\n=== Key Playoffs Series Win Probabilities & Most Likely Scorelines ===")
    key_matchups = [
        ("UBQF1 (Oct 28, Bo5)", "VMS", "FL", 5),
        ("UBQF2 (Oct 28, Bo5)", "iG", "BTRM", 5),
        ("Potential UBSF1 (Oct 29, Bo5)", "TR", "VMS", 5),
        ("Potential UBSF1 (Oct 29, Bo5)", "TR", "FL", 5),
        ("Potential UBSF2 (Oct 29, Bo5)", "SRG", "iG", 5),
        ("Potential UBSF2 (Oct 29, Bo5)", "SRG", "BTRM", 5),
        ("Potential UBF (Oct 30, Bo5)", "SRG", "TR", 5),
        ("Potential UBF (Oct 30, Bo5)", "SRG", "VMS", 5),
        ("Potential Grand Final (Nov 1, Bo7)", "SRG", "VMS", 7),
        ("Potential Grand Final (Nov 1, Bo7)", "SRG", "TR", 7),
        ("Potential Grand Final (Nov 1, Bo7)", "TR", "VMS", 7),
    ]
    for stage, ta, tb, bo in key_matchups:
        p_g = p_game_matrix[(ta, tb)]
        p_s, s_dist = series_win_prob(p_g, bo)
        best_sc = max(s_dist.items(), key=lambda x: x[1])
        m_a, m_b, g_a, g_b, _ = compute_h2h_game_winrate(ta, tb)
        print(
            f"  {stage:34s} | {ta:4s} vs {tb:4s} | RS H2H: {m_a}-{m_b} ({g_a}-{g_b} games) | "
            f"p(Game)={100*p_g:5.1f}% | p(Bo{bo})={100*p_s:5.1f}% {ta} | Top Score: {best_sc[0]} ({100*best_sc[1]:.1f}%)"
        )

    df_sim = simulate_playoffs_monte_carlo(p_game_matrix, n_sims=50_000)
    print("\n=== 50,000-Run Monte Carlo Playoffs Placement & Championship Forecast ===")
    print(df_sim.round(2).to_string(index=False))

    df_sim.round(2).to_csv(DATA_DIR / "mpl_my_s18_playoffs_forecast.csv", index=False)
    plot_forecast(df_sim, feature_cols, lr, ASSETS_DIR / "mpl_my_s18_championship_forecast.png")


if __name__ == "__main__":
    main()