"""Chargement et fusion des données de football-data.co.uk"""

import os
import re
import glob
import pandas as pd


def load_data(data_dir="data/"):
    """
    Charge et fusionne tous les fichiers CSV du dossier data/.

    Args:
        data_dir: Chemin vers le dossier contenant les CSV

    Returns:
        DataFrame trié par date avec colonnes :
        Date, Season, HomeTeam, AwayTeam, FTHG, FTAG, FTR
    """
    pattern = os.path.join(data_dir, "*.csv")
    files = sorted(glob.glob(pattern))

    if not files:
        raise FileNotFoundError(
            f"Aucun fichier CSV trouvé dans '{data_dir}'. "
            "Vérifiez le chemin du dossier de données."
        )

    dfs = []
    for filepath in files:
        filename = os.path.basename(filepath)

        # Extraire la saison depuis le nom de fichier (ex: E0_2018-2019.csv -> 18-19)
        match = re.search(r"(\d{4})-(\d{4})", filename)
        if match:
            y1, y2 = match.group(1), match.group(2)
            season = f"{y1[-2:]}-{y2[-2:]}"
        else:
            season = filename.replace(".csv", "")

        try:
            df = pd.read_csv(filepath, low_memory=False)
        except Exception as e:
            print(f"  Attention: impossible de charger {filename}: {e}")
            continue

        df["Season"] = season
        dfs.append(df)
        print(f"  Chargé : {filename} — {len(df)} matchs (saison {season})")

    if not dfs:
        raise ValueError("Aucun fichier CSV n'a pu être chargé.")

    full_df = pd.concat(dfs, ignore_index=True)

    # Vérifier la présence des colonnes obligatoires
    required = ["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR"]
    missing = [c for c in required if c not in full_df.columns]
    if missing:
        raise ValueError(
            f"Colonnes manquantes : {missing}. "
            f"Colonnes disponibles : {list(full_df.columns)}"
        )

    # Garder uniquement les colonnes utiles
    full_df = full_df[["Date", "Season", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR"]].copy()

    # Convertir la date (format DD/MM/YYYY sur football-data.co.uk)
    full_df["Date"] = pd.to_datetime(full_df["Date"], dayfirst=True, errors="coerce")

    # Convertir les buts en numérique (les CSV peuvent contenir des floats type 2.0)
    full_df["FTHG"] = pd.to_numeric(full_df["FTHG"], errors="coerce")
    full_df["FTAG"] = pd.to_numeric(full_df["FTAG"], errors="coerce")

    # Supprimer les lignes incomplètes et les résultats invalides
    full_df.dropna(subset=required, inplace=True)
    full_df = full_df[full_df["FTR"].isin(["H", "D", "A"])].copy()

    # Trier par date croissante
    full_df.sort_values("Date", inplace=True)
    full_df.reset_index(drop=True, inplace=True)

    return full_df


if __name__ == "__main__":
    print("=== Chargement des données ===")
    df = load_data("data/")
    print(f"\nTotal : {len(df)} matchs chargés")
    print(f"Saisons : {', '.join(sorted(df['Season'].unique()))}")
    print(f"Période : {df['Date'].min().date()} → {df['Date'].max().date()}")
    print(f"\nAperçu :")
    print(df.head())
    print(f"\nTypes de colonnes :")
    print(df.dtypes)
