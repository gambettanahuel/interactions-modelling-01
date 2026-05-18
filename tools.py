
#================================#
#             PLOTS              #
#================================#

# ======= KDE vs evento ========= #

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np


def churn_numeric_analysis(
    data,
    variable,
    target="Churn",
    positive_class="Yes",
    q=10,
    figsize=(10, 7)
):
    """
    Visualización profesional para analizar la relación entre
    una variable numérica y churn.

    Parameters
    ----------
    data : pd.DataFrame
        Dataset.

    variable : str
        Variable numérica a analizar.

    target : str, default="Churn"
        Variable objetivo.

    positive_class : str, default="Yes"
        Clase positiva de churn.

    q : int, default=10
        Cantidad de quantile bins.

    figsize : tuple, default=(12,7)
        Tamaño del gráfico.
    """

    # =========================
    # DATA
    # =========================
    df = data.copy()

    df = df[[variable, target]].dropna()

    df["target_num"] = (
        df[target] == positive_class
    ).astype(int)

    # =========================
    # FIGURE LAYOUT
    # =========================
    fig = plt.figure(figsize=figsize)

    gs = fig.add_gridspec(
        2, 1,
        height_ratios=[3, 1],
        hspace=0.05
    )

    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1], sharex=ax1)

    # =========================
    # KDE DISTRIBUTIONS
    # =========================
    sns.kdeplot(
        data=df[df[target] != positive_class],
        x=variable,
        fill=True,
        alpha=0.35,
        linewidth=2,
        color="lightgray",
        label=f"No",
        ax=ax1
    )

    sns.kdeplot(
        data=df[df[target] == positive_class],
        x=variable,
        fill=True,
        alpha=0.40,
        linewidth=2,
        color="orange",
        label=positive_class,
        ax=ax1
    )

    # =========================
    # QUANTILE CHURN RATE
    # =========================
    df["bin"] = pd.qcut(
        df[variable],
        q=q,
        duplicates="drop"
    )

    summary = (
        df.groupby("bin")
          .agg(
              avg_value=(variable, "mean"),
              churn_rate=("target_num", "mean"),
              count=("target_num", "size")
          )
          .reset_index()
    )

    # Línea de churn rate
    ax2.plot(
        summary["avg_value"],
        summary["churn_rate"],
        marker="o",
        linewidth=2,
        c='orange'
    )

    # =========================
    # STYLING
    # =========================
    ax1.set_title(
        f"{variable} vs {target}",
        fontsize=14
    )

    ax1.set_ylabel("Density", fontsize=12)
    ax1.legend(frameon=False)

    ax2.set_ylabel("Probabilidad", fontsize=11)
    ax2.set_xlabel(variable, fontsize=12)

    # Grid elegante
    for ax in [ax1, ax2]:
        ax.grid(alpha=0.2)

    sns.despine()

    plt.show()

# ======= Interaction numerical vs categorical ========= #
from pygam import LogisticGAM, s, l
from sklearn.metrics import roc_auc_score


def interaction_effect_plot_gam(
    data,
    x,
    group,
    target='Churn',
    positive_class='Yes',
    n_grid=200,
    ci=True,
    figsize=(8,6)
):
    """
    Professional interaction visualization using:
    - Logistic GAM smoothing
    - subgroup conditional probabilities
    - subgroup AUC
    - confidence intervals

    Parameters
    ----------
    data : pd.DataFrame
    x : str
        Continuous feature
    group : str
        Grouping categorical variable
    target : str
    positive_class : str
    n_grid : int
    ci : bool
        Whether to display confidence intervals
    figsize : tuple, default=(12,7)
        Tamaño del gráfico.
    """

    df = data.copy()

    # --- numeric conversion ---
    df[x] = pd.to_numeric(df[x], errors='coerce')

    # remove missing
    df = df[df[x].notna()].copy()

    # binary target
    df['_target_'] = (
        df[target] == positive_class
    ).astype(int)

    groups = sorted(df[group].dropna().unique())

    plt.figure(figsize=figsize)

    auc_dict = {}

    for g in groups:

        sub = df[df[group] == g].copy()

        X = sub[[x]].values
        y = sub['_target_'].values

        # subgroup auc
        auc = roc_auc_score(y, sub[x])
        auc = max(auc, 1 - auc)
        auc_dict[g] = auc

        # --- GAM fit ---
        # gam = LogisticGAM(
        #     s(0, n_splines=5)
        # ).fit(X, y)

        gam = LogisticGAM(
            l(0)
        ).fit(X, y)

        # smooth grid
        XX = gam.generate_X_grid(term=0, n=n_grid)

        preds = gam.predict_proba(XX)

        # confidence intervals
        if ci:
            intervals = gam.confidence_intervals(
                XX,
                width=0.95
            )

        # plot main curve
        plt.plot(
            XX[:, 0],
            preds,
            linewidth=2,
            label=f"{group}={g} | AUC={auc:.3f}"
        )

        # CI band
        if ci:
            plt.fill_between(
                XX[:, 0],
                intervals[:, 0],
                intervals[:, 1],
                alpha=0.15
            )

        # optional raw bin estimates
        bins = pd.qcut(
            sub[x],
            q=15,
            duplicates='drop'
        )

        empirical = (
            sub.groupby(bins, observed=True)
            .agg(
                mean_x=(x, 'mean'),
                churn_rate=('_target_', 'mean')
            )
            .reset_index(drop=True)
        )

        plt.scatter(
            empirical['mean_x'],
            empirical['churn_rate'],
            s=35,
            alpha=0.7
        )

    plt.xlabel(x, fontsize=12)
    plt.ylabel("Probabilidad Observada", fontsize=12)

    plt.title(
        f"Probabilidad Observada de Churn por {group}",
        fontsize=14
    )

    plt.grid(alpha=0.25)

    plt.legend(frameon=True)

    sns.despine()

    plt.tight_layout()

    plt.show()


# ======= Interaction categorical vs categorical ========= #

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def churn_interaction_matrix_marginals(
    data,
    var1,
    var2,
    target='Churn',
    positive_class='Yes',
    figsize=(9,5)
):


    df = data[[var1, var2, target]].dropna().copy()
    df['target_num'] = (df[target] == positive_class).astype(int)

    matrix = df.groupby([var1, var2])['target_num'].mean().unstack()
    counts = df.groupby([var1, var2])['target_num'].size().unstack()
    row_marginals = df.groupby(var1)['target_num'].mean()
    col_marginals = df.groupby(var2)['target_num'].mean()
    overall = df['target_num'].mean()

    expanded = pd.DataFrame(
        index=list(matrix.index) + ['Overall'],
        columns=list(matrix.columns) + ['Overall'],
        dtype=float
    )

    for i in matrix.index:
        for j in matrix.columns:
            expanded.loc[i, j] = matrix.loc[i, j]

    for i in matrix.index:
        expanded.loc[i, 'Overall'] = row_marginals.loc[i]

    for j in matrix.columns:
        expanded.loc['Overall', j] = col_marginals.loc[j]

    expanded.loc['Overall', 'Overall'] = overall

    annot = pd.DataFrame('', index=expanded.index, columns=expanded.columns)

    for i in expanded.index:
        for j in expanded.columns:
            val = expanded.loc[i, j]
            if pd.isna(val):
                annot.loc[i, j] = ''
            elif i != 'Overall' and j != 'Overall':
                n = counts.loc[i, j]
                annot.loc[i, j] = f"{val:.1%}\n(n={int(n)})"
            else:
                annot.loc[i, j] = f"{val:.1%}"

    plt.figure(figsize=figsize)

    ax = sns.heatmap(
        expanded,
        annot=annot,
        fmt='',
        cmap='Oranges',
        linewidths=2,
        linecolor='white',
        mask=expanded.isna(),
        vmin=0,
        vmax=expanded.max().max(),
        cbar_kws={'label': 'Churn Rate'}
    )

    n_rows, n_cols = expanded.shape

    ax.hlines(n_rows - 1, *ax.get_xlim(), colors='black', linewidth=4)
    ax.vlines(n_cols - 1, *ax.get_ylim(), colors='black', linewidth=4)
    ax.add_patch(plt.Rectangle((0, n_rows - 1), n_cols, 1, fill=False, edgecolor='black', lw=4))
    ax.add_patch(plt.Rectangle((n_cols - 1, 0), 1, n_rows, fill=False, edgecolor='black', lw=4))

    plt.title(
        f'Matriz de Interacciones\n{var1} × {var2}',
        fontsize=14,
    )
    plt.xlabel(var2, fontsize=12)
    plt.ylabel(var1, fontsize=12)
    sns.despine(left=True, bottom=True)
    plt.tight_layout()
    plt.show()