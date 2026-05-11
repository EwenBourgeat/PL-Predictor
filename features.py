"""Feature engineering pour la prédiction de matchs de Premier League.

Toutes les features sont calculées en n'utilisant que les matchs passés
par rapport au match en cours — aucun data leakage possible.
"""

import pandas as pd
import numpy as np
from collections import defaultdict

# Colonnes de features partagées entre train.py et app.py
FEATURE_COLS = [
    # Forme glissante — fenêtre 5 matchs
    "home_form",
    "away_form",
    # Forme glissante — fenêtre 3 matchs (momentum court terme)
    "home_form_3",
    "away_form_3",
    # Taux de victoire domicile/extérieur dans la saison en cours
    "home_win_rate",
    "away_win_rate",
    # Buts marqués et encaissés (moyenne 5 derniers matchs)
    "home_goals_scored_avg",
    "away_goals_scored_avg",
    "home_goals_conceded_avg",
    "away_goals_conceded_avg",
    # Différence de buts moyenne (5 derniers matchs)
    "home_goal_diff_avg",
    "away_goal_diff_avg",
    # Points cumulés dans la saison (proxy du classement)
    "home_season_pts",
    "away_season_pts",
    # Force relative : écart de forme entre les deux équipes
    "strength_diff",
    # Confrontation attaque vs défense adverse (proxy expected goals)
    "attack_vs_defense_h",
    "attack_vs_defense_a",
    # Head-to-head (5 dernières confrontations directes)
    "h2h_home_wins",
    "h2h_draws",
    "h2h_away_wins",
]


# ------------------------------------------------------------------
# Fonctions de calcul (privées)
# ------------------------------------------------------------------

def _calc_form(history, n=5):
    """Points moyens sur les n derniers matchs (victoire=3, nul=1, défaite=0)."""
    recent = history[-n:]
    if not recent:
        return 0.0
    return sum(m["pts"] for m in recent) / len(recent)


def _calc_goals_avg(history, n=5):
    """Moyenne de buts marqués et encaissés sur les n derniers matchs."""
    recent = history[-n:]
    if not recent:
        return 0.0, 0.0
    scored   = sum(m["scored"]   for m in recent)
    conceded = sum(m["conceded"] for m in recent)
    return scored / len(recent), conceded / len(recent)


def _calc_home_win_rate(history, season):
    """Taux de victoires à domicile depuis le début de la saison en cours."""
    season_home = [m for m in history if m["season"] == season and m["is_home"]]
    if not season_home:
        return 0.0
    return sum(1 for m in season_home if m["result"] == "W") / len(season_home)


def _calc_away_win_rate(history, season):
    """Taux de victoires à l'extérieur depuis le début de la saison en cours."""
    season_away = [m for m in history if m["season"] == season and not m["is_home"]]
    if not season_away:
        return 0.0
    return sum(1 for m in season_away if m["result"] == "W") / len(season_away)


def _calc_season_pts(history, season):
    """Points cumulés dans la saison en cours (proxy du classement)."""
    return float(sum(m["pts"] for m in history if m["season"] == season))


def _calc_h2h(home_hist, away_team, n=5):
    """
    Résultats des n dernières confrontations directes (du point de vue de home_team).
    Retourne (victoires home_team, nuls, victoires away_team).
    """
    h2h = [m for m in home_hist if m["opponent"] == away_team][-n:]
    if not h2h:
        return 0, 0, 0
    hw = sum(1 for m in h2h if m["result"] == "W")
    d  = sum(1 for m in h2h if m["result"] == "D")
    aw = sum(1 for m in h2h if m["result"] == "L")
    return hw, d, aw


def _extract_features(home_team, away_team, season, team_history):
    """Construit le vecteur de 20 features depuis les historiques actuels."""
    home_hist = team_history[home_team]
    away_hist = team_history[away_team]

    home_form   = _calc_form(home_hist, n=5)
    away_form   = _calc_form(away_hist, n=5)
    home_form_3 = _calc_form(home_hist, n=3)
    away_form_3 = _calc_form(away_hist, n=3)

    home_scored, home_conceded = _calc_goals_avg(home_hist, n=5)
    away_scored, away_conceded = _calc_goals_avg(away_hist, n=5)

    h2h_hw, h2h_d, h2h_aw = _calc_h2h(home_hist, away_team)

    return {
        "home_form":   home_form,
        "away_form":   away_form,
        "home_form_3": home_form_3,
        "away_form_3": away_form_3,
        "home_win_rate":  _calc_home_win_rate(home_hist, season),
        "away_win_rate":  _calc_away_win_rate(away_hist, season),
        "home_goals_scored_avg":   home_scored,
        "away_goals_scored_avg":   away_scored,
        "home_goals_conceded_avg": home_conceded,
        "away_goals_conceded_avg": away_conceded,
        # Différence de buts (buts marqués - buts encaissés)
        "home_goal_diff_avg": home_scored - home_conceded,
        "away_goal_diff_avg": away_scored - away_conceded,
        # Points de saison cumulés
        "home_season_pts": _calc_season_pts(home_hist, season),
        "away_season_pts": _calc_season_pts(away_hist, season),
        # Forme relative entre les deux équipes
        "strength_diff": home_form - away_form,
        # Attaque de l'équipe X face à la défense de l'équipe Y
        "attack_vs_defense_h": home_scored - away_conceded,
        "attack_vs_defense_a": away_scored - home_conceded,
        "h2h_home_wins": h2h_hw,
        "h2h_draws":     h2h_d,
        "h2h_away_wins": h2h_aw,
    }


def _update_history(team_history, row):
    """Ajoute le match terminé dans l'historique des deux équipes."""
    home = row["HomeTeam"]
    away = row["AwayTeam"]
    ftr  = row["FTR"]
    fthg = int(float(row["FTHG"]))
    ftag = int(float(row["FTAG"]))
    season = row["Season"]
    date   = row["Date"]

    if ftr == "H":
        home_result, away_result = "W", "L"
        home_pts, away_pts = 3, 0
    elif ftr == "D":
        home_result = away_result = "D"
        home_pts = away_pts = 1
    else:
        home_result, away_result = "L", "W"
        home_pts, away_pts = 0, 3

    team_history[home].append({
        "date": date, "season": season, "is_home": True, "opponent": away,
        "scored": fthg, "conceded": ftag, "result": home_result, "pts": home_pts,
    })
    team_history[away].append({
        "date": date, "season": season, "is_home": False, "opponent": home,
        "scored": ftag, "conceded": fthg, "result": away_result, "pts": away_pts,
    })


# ------------------------------------------------------------------
# Fonctions publiques
# ------------------------------------------------------------------

def compute_features(df):
    """
    Calcule les features pour tous les matchs sans data leakage.

    Itère chronologiquement: pour chaque match, les features sont calculées
    depuis l'historique disponible AVANT ce match, puis l'historique est mis
    à jour avec le résultat du match.
    """
    df = df.copy().reset_index(drop=True)
    team_history  = defaultdict(list)
    feature_records = []

    total = len(df)
    print(f"Calcul des features pour {total} matchs...")

    for idx in range(total):
        if idx % 500 == 0 and idx > 0:
            print(f"  {idx}/{total} matchs traités...")

        row = df.iloc[idx]
        record = _extract_features(row["HomeTeam"], row["AwayTeam"], row["Season"], team_history)
        feature_records.append(record)
        _update_history(team_history, row)

    features_df = pd.DataFrame(feature_records, index=df.index)
    result = pd.concat([df, features_df], axis=1)
    result["target"] = result["FTR"].map({"H": 0, "D": 1, "A": 2})
    result.dropna(subset=FEATURE_COLS + ["target"], inplace=True)
    result.reset_index(drop=True, inplace=True)

    print(f"Features calculées pour {len(result)} matchs ({len(FEATURE_COLS)} features).")
    return result


def compute_match_features(home_team, away_team, df):
    """
    Calcule les features pour un match hypothétique (utilisé dans l'app Streamlit).
    Utilise l'intégralité de l'historique disponible comme contexte passé.
    """
    all_teams = set(df["HomeTeam"].unique()) | set(df["AwayTeam"].unique())

    if home_team not in all_teams:
        raise ValueError(f"Équipe introuvable dans les données : '{home_team}'")
    if away_team not in all_teams:
        raise ValueError(f"Équipe introuvable dans les données : '{away_team}'")

    team_history = defaultdict(list)
    for _, row in df.iterrows():
        _update_history(team_history, row)

    latest_season = df.sort_values("Date")["Season"].iloc[-1]
    return _extract_features(home_team, away_team, latest_season, team_history)


def get_team_recent_matches(team, df, n=5):
    """Retourne les n derniers matchs d'une équipe."""
    mask   = (df["HomeTeam"] == team) | (df["AwayTeam"] == team)
    recent = df[mask].tail(n).copy()
    return recent[["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR"]]


if __name__ == "__main__":
    from data_loader import load_data

    print("=== Feature Engineering ===")
    df = load_data("data/")
    df_feat = compute_features(df)

    print(f"\nDataset avec features : {df_feat.shape}")
    print(f"Nombre de features : {len(FEATURE_COLS)}")
    print("\nStatistiques descriptives :")
    print(df_feat[FEATURE_COLS].describe().round(3))

    missing = df_feat[FEATURE_COLS].isnull().sum()
    print("\nValeurs manquantes :")
    print(missing[missing > 0] if missing.any() else "  Aucune.")
