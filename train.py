"""Entraînement, évaluation et sauvegarde des modèles de prédiction"""

import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_curve,
    auc,
)
from xgboost import XGBClassifier

from data_loader import load_data
from features import compute_features, FEATURE_COLS

# Labels courts pour les graphiques
SHORT_LABELS = ["H", "D", "A"]


# ------------------------------------------------------------------
# Préparation des données
# ------------------------------------------------------------------

def prepare_data(df_features, train_ratio=0.8):
    """
    Split temporel (pas aléatoire) et normalisation StandardScaler.

    Le scaler est fitté uniquement sur le train pour éviter toute fuite
    d'information vers le jeu de test.
    """
    n_train = int(len(df_features) * train_ratio)

    train = df_features.iloc[:n_train]
    test = df_features.iloc[n_train:]

    X_train = train[FEATURE_COLS].values
    y_train = train["target"].astype(int).values
    X_test = test[FEATURE_COLS].values
    y_test = test["target"].astype(int).values

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test, scaler, train, test


# ------------------------------------------------------------------
# Entraînement
# ------------------------------------------------------------------

def train_models(X_train, y_train):
    """Entraîne les trois modèles et retourne un dict {nom: modèle}."""
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
        "XGBoost": XGBClassifier(n_estimators=200, random_state=42, eval_metric="mlogloss"),
    }

    trained = {}
    for name, model in models.items():
        print(f"  Entraînement : {name}...")
        model.fit(X_train, y_train)
        trained[name] = model

    return trained


# ------------------------------------------------------------------
# Évaluation
# ------------------------------------------------------------------

def evaluate_model(name, model, X_test, y_test):
    """Affiche accuracy, rapport et matrice de confusion pour un modèle."""
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    print(f"\n{'=' * 55}")
    print(f"Modèle : {name}")
    print(f"Accuracy : {acc:.4f}")
    print("\nRapport de classification :")
    print(classification_report(y_test, y_pred, target_names=SHORT_LABELS, zero_division=0))

    cm = confusion_matrix(y_test, y_pred)
    cm_df = pd.DataFrame(cm, index=SHORT_LABELS, columns=SHORT_LABELS)
    print("Matrice de confusion :")
    print(cm_df)

    return acc


def plot_confusion_matrix(name, model, X_test, y_test, ax):
    """Heatmap de la matrice de confusion sur un axe matplotlib donné."""
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=SHORT_LABELS, yticklabels=SHORT_LABELS, ax=ax
    )
    ax.set_title(f"Confusion — {name}", fontsize=10)
    ax.set_ylabel("Réel")
    ax.set_xlabel("Prédit")


def plot_roc_curves(name, model, X_test, y_test, ax):
    """Courbes ROC one-vs-rest (une courbe par classe) sur un axe matplotlib."""
    if not hasattr(model, "predict_proba"):
        ax.text(0.5, 0.5, "predict_proba non disponible",
                ha="center", va="center", transform=ax.transAxes)
        return

    y_proba = model.predict_proba(X_test)
    y_bin = label_binarize(y_test, classes=[0, 1, 2])

    colors = ["steelblue", "darkorange", "green"]
    for i, (label, color) in enumerate(zip(SHORT_LABELS, colors)):
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_proba[:, i])
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=color, lw=2, label=f"{label} (AUC={roc_auc:.3f})")

    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.05])
    ax.set_xlabel("Taux faux positifs")
    ax.set_ylabel("Taux vrais positifs")
    ax.set_title(f"ROC (OvR) — {name}", fontsize=10)
    ax.legend(loc="lower right", fontsize=8)


def plot_feature_importance(model, model_name):
    """Graphique horizontal des importances triées par ordre décroissant."""
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    elif hasattr(model, "coef_"):
        # Régression logistique multi-classe : moyenne des valeurs absolues
        importances = np.abs(model.coef_).mean(axis=0)
    else:
        print(f"  {model_name} ne fournit pas d'importance de features.")
        return

    indices = np.argsort(importances)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(range(len(indices)), importances[indices], align="center", color="steelblue")
    ax.set_yticks(range(len(indices)))
    ax.set_yticklabels([FEATURE_COLS[i] for i in indices])
    ax.set_xlabel("Importance")
    ax.set_title(f"Importance des features — {model_name}")
    plt.tight_layout()
    plt.savefig("feature_importance.png", dpi=150, bbox_inches="tight")
    print("  Sauvegardé : feature_importance.png")
    plt.show()


# ------------------------------------------------------------------
# Sauvegarde
# ------------------------------------------------------------------

MODEL_FILES = {
    "Logistic Regression": "model_lr.pkl",
    "Random Forest":       "model_rf.pkl",
    "XGBoost":             "model_xgb.pkl",
}

def save_artifacts(trained_models, accuracies, scaler, scaler_path="scaler.pkl"):
    """Sauvegarde les 3 modèles, leurs accuracies et le scaler."""
    # Scaler commun aux trois modèles
    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)
    print(f"  Scaler sauvegardé : {scaler_path}")

    # Un fichier par modèle
    for name, model in trained_models.items():
        path = MODEL_FILES[name]
        with open(path, "wb") as f:
            pickle.dump(model, f)
        print(f"  Modèle sauvegardé : {path}  (accuracy test = {accuracies[name]:.4f})")

    # Accuracies séparées pour les afficher dans l'app sans recharger les modèles
    with open("model_scores.pkl", "wb") as f:
        pickle.dump(accuracies, f)
    print("  Scores sauvegardés : model_scores.pkl")


# ------------------------------------------------------------------
# Pipeline principal
# ------------------------------------------------------------------

if __name__ == "__main__":
    # ---- Étape 1 & 2 : Données + features ----
    print("=== Chargement des données ===")
    df = load_data("data/")
    print(f"Total : {len(df)} matchs\n")

    print("=== Feature Engineering ===")
    df_feat = compute_features(df)

    # ---- Étape 3 : Préparation ----
    print("\n=== Préparation des données ===")
    X_train, X_test, y_train, y_test, scaler, train_df, test_df = prepare_data(df_feat)

    print(f"Entraînement : {len(X_train)} matchs  |  Test : {len(X_test)} matchs")
    for split_name, y in [("Train", y_train), ("Test", y_test)]:
        print(f"\n  Distribution {split_name} :")
        for i, lbl in enumerate(SHORT_LABELS):
            n = (y == i).sum()
            print(f"    {lbl} : {n} ({n / len(y) * 100:.1f}%)")

    # ---- Étape 4 : Entraînement ----
    print("\n=== Entraînement des modèles ===")
    trained_models = train_models(X_train, y_train)

    # ---- Évaluation ----
    print("\n=== Évaluation des modèles ===")
    accuracies = {}
    for name, model in trained_models.items():
        acc = evaluate_model(name, model, X_test, y_test)
        accuracies[name] = acc

    # Tableau récapitulatif
    summary = (
        pd.DataFrame.from_dict(accuracies, orient="index", columns=["Accuracy"])
        .sort_values("Accuracy", ascending=False)
    )
    print("\n=== Récapitulatif ===")
    print(summary.round(4))

    # Trouver le meilleur modèle
    best_name = summary.index[0]
    best_model = trained_models[best_name]
    print(f"\nMeilleur modèle : {best_name} (accuracy = {accuracies[best_name]:.4f})")

    # ---- Visualisations ----
    print("\n=== Génération des graphiques ===")
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    for i, (name, model) in enumerate(trained_models.items()):
        plot_confusion_matrix(name, model, X_test, y_test, ax=axes[0, i])
        plot_roc_curves(name, model, X_test, y_test, ax=axes[1, i])

    plt.suptitle("Évaluation des modèles — Premier League Predictor", fontsize=13, y=1.01)
    plt.tight_layout()
    plt.savefig("evaluation.png", dpi=150, bbox_inches="tight")
    print("  Sauvegardé : evaluation.png")
    plt.show()

    plot_feature_importance(best_model, best_name)

    # ---- Sauvegarde ----
    print("\n=== Sauvegarde ===")
    save_artifacts(trained_models, accuracies, scaler)
    print("\nEntraînement terminé.")
