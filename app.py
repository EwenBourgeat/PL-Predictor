"""Application Streamlit — PL Match Predictor"""

import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

from data_loader import load_data
from features import FEATURE_COLS, compute_match_features

# ------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------

st.set_page_config(
    page_title="PL Match Predictor",
    page_icon=None,
    layout="centered",
)

st.markdown("""
<style>
    .block-container { padding-top: 2rem; padding-bottom: 2rem; }
    h1 { font-size: 1.6rem; font-weight: 700; margin-bottom: 0.2rem; }
    .stButton > button {
        width: 100%;
        background-color: #1a1a2e;
        color: white;
        border: none;
        padding: 0.6rem 1rem;
        font-size: 0.95rem;
        font-weight: 600;
        border-radius: 4px;
        margin-top: 0.5rem;
    }
    .stButton > button:hover { background-color: #16213e; }
    .model-card {
        background: #f8f9fa;
        border-radius: 6px;
        padding: 0.8rem 1rem;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .model-name { font-size: 0.85rem; font-weight: 600; color: #333; }
    .model-acc  { font-size: 0.75rem; color: #888; margin-top: 0.1rem; }
    .model-pred { font-size: 1rem; font-weight: 700; color: #1a1a2e; margin-top: 0.4rem; }
    hr { margin: 1.5rem 0; border: none; border-top: 1px solid #e8e8e8; }
</style>
""", unsafe_allow_html=True)

COLORS     = {"H": "#2563eb", "D": "#94a3b8", "A": "#dc2626"}
PIE_COLORS = [COLORS["H"], COLORS["D"], COLORS["A"]]
FTR_LABELS = {0: "Victoire domicile", 1: "Nul", 2: "Victoire exterieur"}

MODEL_FILES = {
    "Logistic Regression": "model_lr.pkl",
    "Random Forest":       "model_rf.pkl",
    "XGBoost":             "model_xgb.pkl",
}

PERIODS = [
    "Debut de saison",
    "Mi-saison",
    "Fin de saison",
    "Date precise",
]


# ------------------------------------------------------------------
# Chargement (mis en cache)
# ------------------------------------------------------------------

@st.cache_data
def load_dataset():
    return load_data("data/")


@st.cache_resource
def load_all_models():
    try:
        with open("scaler.pkl", "rb") as f:
            scaler = pickle.load(f)
        with open("model_scores.pkl", "rb") as f:
            scores = pickle.load(f)
        models = {}
        for name, path in MODEL_FILES.items():
            with open(path, "rb") as f:
                models[name] = pickle.load(f)
        return models, scaler, scores
    except FileNotFoundError:
        return None, None, None


def cutoff_for_period(period, season_df, custom_date=None):
    """
    Retourne la date de coupure (pd.Timestamp) en fonction de la période choisie.
    On utilise uniquement les matchs joués AVANT cette date pour calculer les features.
    """
    first_date = season_df["Date"].min()
    last_date  = season_df["Date"].max()

    if period == "Debut de saison":
        # ~mi-octobre : environ 7-8 journées jouées
        return pd.Timestamp(year=first_date.year, month=10, day=15)

    if period == "Mi-saison":
        # ~janvier : environ 20 journées jouées (mi-chemin)
        # La 2e année de la saison (ex: 2019 pour 18-19)
        return pd.Timestamp(year=last_date.year, month=1, day=15)

    if period == "Fin de saison":
        # Après le dernier match : toute la saison est disponible
        return last_date + pd.Timedelta(days=1)

    # Date précise choisie par l'utilisateur
    if custom_date is not None:
        return pd.Timestamp(custom_date)

    return last_date + pd.Timedelta(days=1)


# ------------------------------------------------------------------
# Graphiques
# ------------------------------------------------------------------

def make_pie_charts(predictions, scores):
    fig, axes = plt.subplots(1, 3, figsize=(11, 4))
    fig.patch.set_facecolor("white")

    for ax, (name, (proba, _)) in zip(axes, predictions.items()):
        wedges, _, autotexts = ax.pie(
            proba,
            labels=["Dom.", "Nul", "Ext."],
            autopct="%1.1f%%",
            colors=PIE_COLORS,
            startangle=90,
            wedgeprops={"edgecolor": "white", "linewidth": 2},
            textprops={"fontsize": 9},
        )
        for at in autotexts:
            at.set_fontsize(8)
            at.set_color("white")
            at.set_fontweight("bold")

        ax.set_title(f"{name}\n{scores[name]*100:.1f}% accuracy", fontsize=9, pad=10, color="#333")

    plt.tight_layout(pad=2)
    return fig


def make_bar_chart(predictions, home_team, away_team):
    model_names = list(predictions.keys())
    x     = np.arange(len(model_names))
    width = 0.25

    fig, ax = plt.subplots(figsize=(9, 4))
    fig.patch.set_facecolor("white")

    for i, (label, color, idx) in enumerate([
        ("Victoire domicile",  COLORS["H"], 0),
        ("Nul",                COLORS["D"], 1),
        ("Victoire exterieur", COLORS["A"], 2),
    ]):
        values = [predictions[m][0][idx] * 100 for m in model_names]
        bars = ax.bar(x + i * width, values, width, label=label, color=color, alpha=0.85)
        for bar, val in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.5,
                f"{val:.1f}%",
                ha="center", va="bottom", fontsize=7.5, color="#333",
            )

    ax.set_xticks(x + width)
    ax.set_xticklabels([m.replace(" ", "\n") for m in model_names], fontsize=9)
    ax.set_ylabel("Probabilite (%)", fontsize=9)
    ax.set_ylim(0, 105)
    ax.set_title(f"Comparaison des modeles — {home_team} vs {away_team}", fontsize=10, pad=10)
    ax.legend(fontsize=8, loc="upper right")
    ax.spines[["top", "right"]].set_visible(False)
    ax.yaxis.grid(True, linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)

    plt.tight_layout()
    return fig


# ------------------------------------------------------------------
# Interface principale
# ------------------------------------------------------------------

df_full = load_dataset()
all_models, scaler, scores = load_all_models()

st.markdown("# PL Match Predictor")
st.markdown("Prediction de resultats de Premier League.")
st.markdown("<hr>", unsafe_allow_html=True)

if all_models is None:
    st.error("Modeles introuvables. Lancez d'abord : python train.py")
    st.stop()

# --- Saison ---
seasons_available = sorted(df_full["Season"].unique())
selected_season   = st.selectbox("Saison", seasons_available, index=len(seasons_available) - 1)

df_season = df_full[df_full["Season"] == selected_season].copy().reset_index(drop=True)

# --- Période de prédiction ---
st.markdown("**Point de prediction dans la saison**")
period = st.radio(
    "Periode",
    PERIODS,
    horizontal=True,
    label_visibility="collapsed",
    help=(
        "Debut de saison : ~mi-octobre (~7 journees jouees)\n"
        "Mi-saison : ~janvier (~20 journees jouees)\n"
        "Fin de saison : apres le dernier match\n"
        "Date precise : vous choisissez"
    ),
)

custom_date = None
if period == "Date precise":
    season_min = df_season["Date"].min().date()
    season_max = df_season["Date"].max().date()
    custom_date = st.date_input(
        "Date de prediction",
        value=season_min + (df_season["Date"].max() - df_season["Date"].min()) / 2,
        min_value=season_min,
        max_value=season_max,
    )

# Calculer la date de coupure et filtrer
cutoff    = cutoff_for_period(period, df_season, custom_date)
df_active = df_season[df_season["Date"] < cutoff].copy().reset_index(drop=True)

n_matches = len(df_active)
if n_matches == 0:
    st.warning(
        f"Aucun match disponible avant le {cutoff.strftime('%d/%m/%Y')} "
        f"dans la saison {selected_season}. Choisissez une periode plus tardive."
    )
    st.stop()

st.caption(
    f"{n_matches} matchs utilises pour les features "
    f"(avant le {cutoff.strftime('%d/%m/%Y')})"
)

# Equipes présentes dans les matchs filtrés
teams_active = sorted(set(df_active["HomeTeam"]) | set(df_active["AwayTeam"]))

st.markdown("<hr>", unsafe_allow_html=True)

# --- Equipes ---
col1, col2 = st.columns(2)
with col1:
    st.markdown("**Domicile**")
    home_team = st.selectbox("Domicile", teams_active, index=0, label_visibility="collapsed")
with col2:
    st.markdown("**Exterieur**")
    default_away = 1 if len(teams_active) > 1 else 0
    away_team = st.selectbox("Exterieur", teams_active, index=default_away, label_visibility="collapsed")

if home_team == away_team:
    st.warning("Selectionnez deux equipes differentes.")
    st.stop()

predict = st.button("Predire le resultat")

# ------------------------------------------------------------------
# Prédictions
# ------------------------------------------------------------------

if predict:
    try:
        feat_dict = compute_match_features(home_team, away_team, df_active)
        X        = np.array([[feat_dict[col] for col in FEATURE_COLS]])
        X_scaled = scaler.transform(X)

        predictions = {}
        for name, model in all_models.items():
            proba = model.predict_proba(X_scaled)[0]
            pred  = int(model.predict(X_scaled)[0])
            predictions[name] = (proba, pred)

        st.markdown("<hr>", unsafe_allow_html=True)
        st.markdown(
            f"**{home_team} — {away_team}** "
            f"· Saison {selected_season} · {period.lower()} "
            f"({n_matches} matchs de contexte)"
        )

        # Cartes verdict
        cols = st.columns(3)
        for col, (name, (proba, pred)) in zip(cols, predictions.items()):
            with col:
                st.markdown(
                    f'<div class="model-card">'
                    f'<div class="model-name">{name}</div>'
                    f'<div class="model-acc">{scores[name]*100:.1f}% accuracy</div>'
                    f'<div class="model-pred">{FTR_LABELS[pred]}</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

        # Camemberts
        st.markdown("**Repartition des probabilites**")
        fig_pie = make_pie_charts(predictions, scores)
        st.pyplot(fig_pie, use_container_width=True)
        plt.close(fig_pie)

        # Barres comparatives
        st.markdown("**Comparaison des modeles**")
        fig_bar = make_bar_chart(predictions, home_team, away_team)
        st.pyplot(fig_bar, use_container_width=True)
        plt.close(fig_bar)

    except Exception as e:
        st.error(f"Erreur : {e}")

st.markdown("<hr>", unsafe_allow_html=True)
st.caption("Donnees : football-data.co.uk  |  Saisons 2018-19 a 2024-25")


if __name__ == "__main__":
    print("Lancez l'application avec : streamlit run app.py")
