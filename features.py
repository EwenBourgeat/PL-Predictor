"""Feature engineering pour la prédiction de matchs de Premier League.

Toutes les features sont calculées en n'utilisant que les matchs passés
par rapport au match en cours — aucun data leakage possible.
"""

import pandas as pd
import numpy as np
from collections import defaultdict

# Colonnes de features partagées entre train.py et app.py
FEATURE_COLS = [
    "home_form",
    "away_form",
    "home_win_rate",
    "away_win_rate",
    "home_goals_scored_avg",
    "away_goals_scored_avg",
    "home_goals_conceded_avg",
    "away_goals_conceded_avg",
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
    scored = sum(m["scored"] for m in recent)
    conceded = sum(m["conceded"] for m in recent)
    return scored / len(recent), conceded / len(recent)


def _calc_home_win_rate(history, season):
    """Taux de victoires à domicile depuis le début de la saison en cours."""
    season_home = [m for m in history if m["season"] == season and m["is_home"]]
    if not season_home:
        return 0.0
    wins = sum(1 for m in season_home if m["result"] == "W")
    return wins / len(season_home)


def _calc_away_win_rate(history, season):
    """Taux de victoires à l'extérieur depuis le début de la saison en cours."""
    season_away = [m for m in history if m["season"] == season and not m["is_home"]]
    if not season_away:
        return 0.0
    wins = sum(1 for m in season_away if m["result"] == "W")
    return wins / len(season_away)


def _calc_h2h(home_hist, away_team, n=5):
    """
    Résultats des n dernières confrontations directes (du point de vue de home_team).
    Retourne (victoires home_team, nuls, victoires away_team).
    """
    h2h = [m for m in home_hist if m["opponent"] == away_team][-n:]
    if not h2h:
        return 0, 0, 0
    hw = sum(1 for m in h2h if m["result"] == "W")
    d = sum(1 for m in h2h if m["result"] == "D")
    aw = sum(1 for m in h2h if m["result"] == "L")
    return hw, d, aw


def _extract_features(home_team, away_team, season, team_history):
    """Construit le vecteur de features depuis les historiques actuels."""
    home_hist = team_history[home_team]
    away_hist = team_history[away_team]

    home_scored, home_conceded = _calc_goals_avg(home_hist)
    away_scored, away_conceded = _calc_goals_avg(away_hist)
    h2h_hw, h2h_d, h2h_aw = _calc_h2h(home_hist, away_team)

    return {
        "home_form": _calc_form(home_hist),
        "away_form": _calc_form(away_hist),
        "home_win_rate": _calc_home_win_rate(home_hist, season),
        "away_win_rate": _calc_away_win_rate(away_hist, season),
        "home_goals_scored_avg": home_scored,
        "away_goals_scored_avg": away_scored,
        "home_goals_conceded_avg": home_conceded,
        "away_goals_conceded_avg": away_conceded,
        "h2h_home_wins": h2h_hw,
        "h2h_draws": h2h_d,
        "h2h_away_wins": h2h_aw,
    }


def _update_history(team_history, row):
    """Ajoute le match terminé dans l'historique des deux équipes."""
    home = row["HomeTeam"]
    away = row["AwayTeam"]
    ftr = row["FTR"]
    fthg = int(float(row["FTHG"]))
    ftag = int(float(row["FTAG"]))
    season = row["Season"]
    date = row["Date"]

    if ftr == "H":
        home_result, away_result = "W", "L"
        home_pts, away_pts = 3, 0
    elif ftr == "D":
        home_result = away_result = "D"
        home_pts = away_pts = 1
    else:  # "A"
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

    Args:
        df: DataFrame avec colonnes Date, Season, HomeTeam, AwayTeam, FTHG, FTAG, FTR

    Returns:
        DataFrame enrichi avec les features et la colonne 'target' (H=0, D=1, A=2)
    """
    df = df.copy().reset_index(drop=True)
    team_history = defaultdict(list)
    feature_records = []

    total = len(df)
    print(f"Calcul des features pour {total} matchs...")

    for idx in range(total):
        if idx % 500 == 0 and idx > 0:
            print(f"  {idx}/{total} matchs traités...")

        row = df.iloc[idx]

        # Extraire les features depuis l'historique ACTUEL (avant ce match)
        record = _extract_features(row["HomeTeam"], row["AwayTeam"], row["Season"], team_history)
        feature_records.append(record)

        # Mettre à jour l'historique APRÈS le calcul — jamais avant
        _update_history(team_history, row)

    # Fusionner les features avec le DataFrame original
    features_df = pd.DataFrame(feature_records, index=df.index)
    result = pd.concat([df, features_df], axis=1)

    # Encoder la variable cible : H=0, D=1, A=2
    result["target"] = result["FTR"].map({"H": 0, "D": 1, "A": 2})

    # Supprimer les lignes avec valeurs manquantes
    result.dropna(subset=FEATURE_COLS + ["target"], inplace=True)
    result.reset_index(drop=True, inplace=True)

    print(f"Features calculées pour {len(result)} matchs.")
    return result


def compute_match_features(home_team, away_team, df):
    """
    Calcule les features pour un match hypothétique (utilisé dans l'app Streamlit).

    Utilise l'intégralité de l'historique disponible comme contexte passé.

    Args:
        home_team: Nom de l'équipe domicile
        away_team: Nom de l'équipe extérieure
        df: DataFrame complet des matchs historiques

    Returns:
        dict avec les features du match (clés = FEATURE_COLS)

    Raises:
        ValueError: Si une équipe n'apparaît pas dans les données
    """
    all_teams = set(df["HomeTeam"].unique()) | set(df["AwayTeam"].unique())

    if home_team not in all_teams:
        raise ValueError(f"Équipe introuvable dans les données : '{home_team}'")
    if away_team not in all_teams:
        raise ValueError(f"Équipe introuvable dans les données : '{away_team}'")

    # Construire l'historique complet de toutes les équipes
    team_history = defaultdict(list)
    for _, row in df.iterrows():
        _update_history(team_history, row)

    # Pour les taux de victoire, on utilise la saison la plus récente
    latest_season = df.sort_values("Date")["Season"].iloc[-1]

    return _extract_features(home_team, away_team, latest_season, team_history)


def get_team_recent_matches(team, df, n=5):
    """Retourne les n derniers matchs d'une équipe (pour l'affichage dans l'app)."""
    mask = (df["HomeTeam"] == team) | (df["AwayTeam"] == team)
    recent = df[mask].tail(n).copy()
    return recent[["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR"]]


if __name__ == "__main__":
    from data_loader import load_data

    print("=== Feature Engineering ===")
    df = load_data("data/")
    df_feat = compute_features(df)

    print(f"\nDataset avec features : {df_feat.shape}")
    print("\nStatistiques descriptives :")
    print(df_feat[FEATURE_COLS].describe().round(3))

    print("\nValeurs manquantes :")
    missing = df_feat[FEATURE_COLS].isnull().sum()
    print(missing[missing > 0] if missing.any() else "  Aucune valeur manquante.")

    print("\nDistribution de la variable cible :")
    dist = df_feat["FTR"].value_counts(normalize=True).round(3)
    print(dist)
