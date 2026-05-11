"""Application Streamlit — PL Match Predictor"""

import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import streamlit as st

from data_loader import load_data
from features import FEATURE_COLS, compute_match_features, compute_features

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
    .metric-card {
        background: #f8f9fa;
        border-radius: 6px;
        padding: 1rem;
        text-align: center;
    }
    .metric-label { font-size: 0.75rem; color: #888; text-transform: uppercase; letter-spacing: 0.05em; }
    .metric-value { font-size: 1.6rem; font-weight: 700; color: #1a1a2e; margin-top: 0.2rem; }
    .metric-sub   { font-size: 0.75rem; color: #666; margin-top: 0.1rem; }
    hr { margin: 1.5rem 0; border: none; border-top: 1px solid #e8e8e8; }
    .tag-train { background:#dbeafe; color:#1e40af; border-radius:4px; padding:2px 7px; font-size:0.72rem; font-weight:600; }
    .tag-test  { background:#dcfce7; color:#166534; border-radius:4px; padding:2px 7px; font-size:0.72rem; font-weight:600; }
</style>
""", unsafe_allow_html=True)

COLORS     = {"H": "#2563eb", "D": "#94a3b8", "A": "#dc2626"}
PIE_COLORS = [COLORS["H"], COLORS["D"], COLORS["A"]]
FTR_LABELS = {0: "Victoire domicile", 1: "Nul", 2: "Victoire exterieur"}
SHORT      = ["H", "D", "A"]

MODEL_FILES = {
    "Logistic Regression": "model_lr.pkl",
    "Random Forest":       "model_rf.pkl",
    "XGBoost":             "model_xgb.pkl",
}
PERIODS = ["Debut de saison", "Mi-saison", "Fin de saison", "Date precise"]

TRAIN_RATIO = 0.8   # doit correspondre à ce qui est dans train.py


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


@st.cache_data
def build_eval_data(_scaler, df_full):
    """
    Calcule les features pour tous les matchs et retourne les prédictions
    de chaque modèle, avec la distinction train/test.
    Mis en cache pour ne pas recalculer à chaque interaction.
    """
    df_feat = compute_features(df_full)
    n_train = int(len(df_feat) * TRAIN_RATIO)

    X_all = _scaler.transform(df_feat[FEATURE_COLS].values)
    y_all = df_feat["target"].astype(int).values

    # Charger les modèles directement (pas via all_models pour éviter
    # un problème de sérialisation avec st.cache_data)
    preds = {}
    probas = {}
    for name, path in MODEL_FILES.items():
        with open(path, "rb") as f:
            m = pickle.load(f)
        preds[name]  = m.predict(X_all)
        probas[name] = m.predict_proba(X_all)

    df_feat = df_feat.copy()
    df_feat["split"] = "Train"
    df_feat.iloc[n_train:, df_feat.columns.get_loc("split")] = "Test"

    return df_feat, X_all, y_all, preds, probas, n_train


# ------------------------------------------------------------------
# Graphiques — onglet Prédiction
# ------------------------------------------------------------------

def cutoff_for_period(period, season_df, custom_date=None):
    first_date = season_df["Date"].min()
    last_date  = season_df["Date"].max()
    if period == "Debut de saison":
        return pd.Timestamp(year=first_date.year, month=10, day=15)
    if period == "Mi-saison":
        return pd.Timestamp(year=last_date.year, month=1, day=15)
    if period == "Fin de saison":
        return last_date + pd.Timedelta(days=1)
    if custom_date is not None:
        return pd.Timestamp(custom_date)
    return last_date + pd.Timedelta(days=1)


def make_pie_charts(predictions, scores):
    fig, axes = plt.subplots(1, 3, figsize=(11, 4))
    fig.patch.set_facecolor("white")
    for ax, (name, (proba, _)) in zip(axes, predictions.items()):
        wedges, _, autotexts = ax.pie(
            proba, labels=["Dom.", "Nul", "Ext."],
            autopct="%1.1f%%", colors=PIE_COLORS, startangle=90,
            wedgeprops={"edgecolor": "white", "linewidth": 2},
            textprops={"fontsize": 9},
        )
        for at in autotexts:
            at.set_fontsize(8); at.set_color("white"); at.set_fontweight("bold")
        ax.set_title(f"{name}\n{scores[name]*100:.1f}% accuracy", fontsize=9, pad=10, color="#333")
    plt.tight_layout(pad=2)
    return fig


def make_bar_chart(predictions, home_team, away_team):
    model_names = list(predictions.keys())
    x, width = np.arange(len(model_names)), 0.25
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
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                    f"{val:.1f}%", ha="center", va="bottom", fontsize=7.5, color="#333")
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
# Graphiques — onglet Analyse
# ------------------------------------------------------------------

def make_accuracy_by_season(df_feat, preds):
    """
    Accuracy de chaque modèle par saison.
    Note : les saisons purement Train ont un biais (le modèle a appris sur ces données).
    La 23-24 est mixte (60% train / 40% test). La 24-25 est purement test.
    """
    seasons     = sorted(df_feat["Season"].unique())
    model_names = list(preds.keys())
    acc_data    = {name: [] for name in model_names}
    zone_colors = []

    for season in seasons:
        mask      = df_feat["Season"] == season
        y_s       = df_feat.loc[mask, "target"].astype(int).values
        n_test    = (df_feat.loc[mask, "split"] == "Test").sum()
        n_total   = mask.sum()
        pct_test  = n_test / n_total

        if pct_test == 0:
            zone_colors.append("#dbeafe")    # bleu clair = full train
        elif pct_test == 1:
            zone_colors.append("#dcfce7")    # vert clair = full test
        else:
            zone_colors.append("#fef9c3")    # jaune clair = mixte

        for name in model_names:
            y_hat = preds[name][mask]
            acc_data[name].append(accuracy_score(y_s, y_hat) * 100)

    fig, ax = plt.subplots(figsize=(10, 4.5))
    fig.patch.set_facecolor("white")

    x, width = np.arange(len(seasons)), 0.22
    palette  = ["#2563eb", "#16a34a", "#d97706"]

    for (name, accs), offset, color in zip(acc_data.items(), [-1, 0, 1], palette):
        ax.bar(x + offset * width, accs, width, label=name, color=color, alpha=0.85)

    for i, bg in enumerate(zone_colors):
        ax.axvspan(i - 0.5, i + 0.5, color=bg, alpha=0.3, zorder=0)

    ax.set_xticks(x)
    ax.set_xticklabels(seasons, fontsize=9)
    ax.set_ylabel("Accuracy (%)", fontsize=9)
    ax.set_ylim(0, 105)
    ax.set_title(
        "Accuracy par saison\n"
        "Attention : RF et XGBoost overfit le train (accuracy artificiellement haute)",
        fontsize=10, pad=10,
    )
    ax.spines[["top", "right"]].set_visible(False)
    ax.yaxis.grid(True, linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)

    from matplotlib.patches import Patch
    model_handles = [
        Patch(color=c, alpha=0.85, label=n)
        for n, c in zip(model_names, palette)
    ]
    zone_handles = [
        Patch(color="#dbeafe", alpha=0.6, label="Train"),
        Patch(color="#fef9c3", alpha=0.6, label="Mixte (23-24)"),
        Patch(color="#dcfce7", alpha=0.6, label="Test"),
    ]
    ax.legend(handles=model_handles + zone_handles, fontsize=8, loc="lower left",
              ncol=2, framealpha=0.8)

    plt.tight_layout()
    return fig


def make_ftr_distribution(df_feat):
    """Proportion H/D/A par saison."""
    seasons = sorted(df_feat["Season"].unique())
    h_vals, d_vals, a_vals = [], [], []
    for season in seasons:
        sub = df_feat[df_feat["Season"] == season]["FTR"]
        total = len(sub)
        h_vals.append((sub == "H").sum() / total * 100)
        d_vals.append((sub == "D").sum() / total * 100)
        a_vals.append((sub == "A").sum() / total * 100)

    x = np.arange(len(seasons))
    fig, ax = plt.subplots(figsize=(10, 4))
    fig.patch.set_facecolor("white")

    ax.bar(x, h_vals, label="Victoire domicile", color=COLORS["H"], alpha=0.85)
    ax.bar(x, d_vals, bottom=h_vals, label="Nul", color=COLORS["D"], alpha=0.85)
    ax.bar(x, a_vals, bottom=[h + d for h, d in zip(h_vals, d_vals)],
           label="Victoire exterieur", color=COLORS["A"], alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(seasons, fontsize=9)
    ax.set_ylabel("Proportion (%)", fontsize=9)
    ax.set_ylim(0, 105)
    ax.set_title("Distribution des resultats par saison (H / D / A)", fontsize=11, pad=10)
    ax.legend(fontsize=8, loc="upper right")
    ax.spines[["top", "right"]].set_visible(False)
    ax.yaxis.grid(True, linestyle="--", alpha=0.3)
    ax.set_axisbelow(True)
    plt.tight_layout()
    return fig


def make_confusion_matrices(df_feat, preds, split="Test"):
    """Matrices de confusion pour les 3 modèles sur le split choisi."""
    mask   = df_feat["split"] == split
    y_true = df_feat.loc[mask, "target"].astype(int).values

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    fig.patch.set_facecolor("white")

    for ax, (name, y_hat_all) in zip(axes, preds.items()):
        y_hat = y_hat_all[mask]
        cm    = confusion_matrix(y_true, y_hat)
        acc   = accuracy_score(y_true, y_hat)
        sns.heatmap(
            cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=SHORT, yticklabels=SHORT,
            ax=ax, cbar=False,
        )
        ax.set_title(f"{name}\nAccuracy {split} = {acc*100:.1f}%", fontsize=9, pad=8)
        ax.set_xlabel("Predit", fontsize=8)
        ax.set_ylabel("Reel", fontsize=8)

    plt.tight_layout(pad=2)
    return fig


def make_f1_radar(df_feat, preds, split="Test"):
    """F1-score par classe (H/D/A) pour chaque modèle — graphique en barres groupées."""
    from sklearn.metrics import f1_score

    mask   = df_feat["split"] == split
    y_true = df_feat.loc[mask, "target"].astype(int).values

    model_names = list(preds.keys())
    classes     = ["H (Dom.)", "D (Nul)", "A (Ext.)"]
    x           = np.arange(len(classes))
    width       = 0.22
    palette     = ["#2563eb", "#16a34a", "#d97706"]

    fig, ax = plt.subplots(figsize=(9, 4))
    fig.patch.set_facecolor("white")

    for i, (name, color) in enumerate(zip(model_names, palette)):
        y_hat = preds[name][mask]
        f1s   = f1_score(y_true, y_hat, average=None, labels=[0, 1, 2], zero_division=0)
        bars  = ax.bar(x + (i - 1) * width, f1s * 100, width,
                       label=name, color=color, alpha=0.85)
        for bar, val in zip(bars, f1s):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                    f"{val*100:.1f}", ha="center", va="bottom", fontsize=7.5, color="#333")

    ax.set_xticks(x)
    ax.set_xticklabels(classes, fontsize=9)
    ax.set_ylabel("F1-score (%)", fontsize=9)
    ax.set_ylim(0, 100)
    ax.set_title(f"F1-score par classe — jeu de {split.lower()}", fontsize=11, pad=10)
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
st.markdown("<hr>", unsafe_allow_html=True)

if all_models is None:
    st.error("Modeles introuvables. Lancez d'abord : python train.py")
    st.stop()

tab_pred, tab_data = st.tabs(["Prediction", "Analyse des modeles"])


# ==================================================================
# ONGLET 1 — Prédiction
# ==================================================================

with tab_pred:

    seasons_available = sorted(df_full["Season"].unique())
    selected_season   = st.selectbox("Saison", seasons_available, index=len(seasons_available) - 1)

    df_season = df_full[df_full["Season"] == selected_season].copy().reset_index(drop=True)

    st.markdown("**Point de prediction dans la saison**")
    period = st.radio(
        "Periode", PERIODS, horizontal=True, label_visibility="collapsed",
        help=(
            "Debut de saison : ~mi-octobre (~7 journees jouees)\n"
            "Mi-saison : ~janvier (~20 journees jouees)\n"
            "Fin de saison : apres le dernier match\n"
            "Date precise : vous choisissez"
        ),
    )

    custom_date = None
    if period == "Date precise":
        season_min  = df_season["Date"].min().date()
        season_max  = df_season["Date"].max().date()
        custom_date = st.date_input(
            "Date de prediction",
            value=season_min + (df_season["Date"].max() - df_season["Date"].min()) / 2,
            min_value=season_min, max_value=season_max,
        )

    cutoff    = cutoff_for_period(period, df_season, custom_date)
    df_active = df_season[df_season["Date"] < cutoff].copy().reset_index(drop=True)

    if len(df_active) == 0:
        st.warning(
            f"Aucun match disponible avant le {cutoff.strftime('%d/%m/%Y')} "
            f"dans la saison {selected_season}."
        )
        st.stop()

    st.caption(f"{len(df_active)} matchs utilises pour les features (avant le {cutoff.strftime('%d/%m/%Y')})")

    teams_active = sorted(set(df_active["HomeTeam"]) | set(df_active["AwayTeam"]))

    st.markdown("<hr>", unsafe_allow_html=True)

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

    if st.button("Predire le resultat"):
        try:
            feat_dict = compute_match_features(home_team, away_team, df_active)
            X         = np.array([[feat_dict[col] for col in FEATURE_COLS]])
            X_scaled  = scaler.transform(X)

            predictions = {}
            for name, model in all_models.items():
                proba = model.predict_proba(X_scaled)[0]
                pred  = int(model.predict(X_scaled)[0])
                predictions[name] = (proba, pred)

            st.markdown("<hr>", unsafe_allow_html=True)
            st.markdown(
                f"**{home_team} — {away_team}** "
                f"· Saison {selected_season} · {period.lower()} "
                f"({len(df_active)} matchs de contexte)"
            )

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

            st.markdown("**Repartition des probabilites**")
            fig_pie = make_pie_charts(predictions, scores)
            st.pyplot(fig_pie, use_container_width=True)
            plt.close(fig_pie)

            st.markdown("**Comparaison des modeles**")
            fig_bar = make_bar_chart(predictions, home_team, away_team)
            st.pyplot(fig_bar, use_container_width=True)
            plt.close(fig_bar)

        except Exception as e:
            st.error(f"Erreur : {e}")

    st.markdown("<hr>", unsafe_allow_html=True)
    st.caption("Donnees : football-data.co.uk  |  Saisons 2018-19 a 2024-25")


# ==================================================================
# ONGLET 2 — Analyse des modèles
# ==================================================================

with tab_data:

    st.markdown("### Performance des modeles")
    st.caption(
        f"Split chronologique 80/20 — Train : 2128 matchs (saisons 18-19 a mi-23-24) "
        f"· Test : 532 matchs (fin 23-24 + 24-25)"
    )

    with st.spinner("Calcul des predictions sur l'ensemble des donnees..."):
        df_feat, X_all, y_all, preds, probas, n_train = build_eval_data(scaler, df_full)

    # --- Métriques globales (test set) ---
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("**Accuracy globale — jeu de test**")

    mask_test  = df_feat["split"] == "Test"
    y_test_all = df_feat.loc[mask_test, "target"].astype(int).values

    cols = st.columns(3)
    for col, name in zip(cols, preds.keys()):
        y_hat = preds[name][mask_test]
        acc   = accuracy_score(y_test_all, y_hat)
        best_marker = " (meilleur)" if acc == max(
            accuracy_score(y_test_all, preds[n][mask_test]) for n in preds
        ) else ""
        with col:
            st.markdown(
                f'<div class="metric-card">'
                f'<div class="metric-label">{name}</div>'
                f'<div class="metric-value">{acc*100:.1f}%</div>'
                f'<div class="metric-sub">accuracy test{best_marker}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

    # --- Tableau accuracy par saison ---
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown(
        "**Accuracy par saison**  "
        '<span class="tag-train">Train</span>&nbsp;'
        '<span class="tag-test">Test</span>',
        unsafe_allow_html=True,
    )

    seasons_list = sorted(df_feat["Season"].unique())
    table_rows   = []
    for season in seasons_list:
        mask  = df_feat["Season"] == season
        y_s   = df_feat.loc[mask, "target"].astype(int).values
        pct_test = (df_feat.loc[mask, "split"] == "Test").mean()
        if pct_test == 0:
            split_label = "Train"
        elif pct_test >= 0.9:
            split_label = "Test"
        else:
            split_label = "Mixte"
        row = {"Saison": season, "Split": split_label}
        for name in preds:
            acc = accuracy_score(y_s, preds[name][mask])
            row[name] = round(acc * 100, 1)
        table_rows.append(row)

    df_table = pd.DataFrame(table_rows).set_index("Saison")

    def color_split(val):
        if val == "Train":
            return "background-color:#dbeafe; color:#1e40af; font-weight:600"
        if val == "Test":
            return "background-color:#dcfce7; color:#166534; font-weight:600"
        return "background-color:#fef9c3; color:#854d0e; font-weight:600"

    def color_acc(val):
        if not isinstance(val, float):
            return ""
        if val >= 60:
            return "color:#166534; font-weight:600"
        if val >= 50:
            return "color:#1a1a2e"
        return "color:#dc2626"

    styled = (
        df_table.style
        .applymap(color_split, subset=["Split"])
        .applymap(color_acc, subset=list(preds.keys()))
        .format({name: "{:.1f}%" for name in preds.keys()})
    )
    st.dataframe(styled, use_container_width=True)
    st.caption(
        "Vert fonce = accuracy >= 60%  ·  Noir = >= 50%  ·  Rouge = < 50%  "
        "·  Les saisons Train ont un biais (modele a appris sur ces donnees)"
    )

    # Graphique accuracy par saison
    fig_acc = make_accuracy_by_season(df_feat, preds)
    st.pyplot(fig_acc, use_container_width=True)
    plt.close(fig_acc)

    # --- F1-score par classe (test) ---
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("**F1-score par classe — jeu de test**")
    st.caption("Le nul (D) est structurellement difficile à prédire en football.")
    fig_f1 = make_f1_radar(df_feat, preds, split="Test")
    st.pyplot(fig_f1, use_container_width=True)
    plt.close(fig_f1)

    # --- Matrices de confusion (test) ---
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("**Matrices de confusion — jeu de test**")
    fig_cm = make_confusion_matrices(df_feat, preds, split="Test")
    st.pyplot(fig_cm, use_container_width=True)
    plt.close(fig_cm)

    # --- Distribution H/D/A par saison ---
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("**Distribution des resultats par saison**")
    st.caption("Contexte pour interpreter les performances : certaines saisons favorisent davantage le domicile.")
    fig_dist = make_ftr_distribution(df_feat)
    st.pyplot(fig_dist, use_container_width=True)
    plt.close(fig_dist)

    st.markdown("<hr>", unsafe_allow_html=True)
    st.caption("Donnees : football-data.co.uk  |  Saisons 2018-19 a 2024-25")


if __name__ == "__main__":
    print("Lancez l'application avec : streamlit run app.py")
