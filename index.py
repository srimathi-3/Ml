"""
Partial Dependence Plot (Layered Bar Chart) for California Housing Dataset
Using RandomForestRegressor

Author: Srimathi
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.inspection import partial_dependence


def main():
    # 1) Load dataset
    data = fetch_california_housing(as_frame=True)
    X = data.frame
    y = data.target

    # 2) Train model
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, random_state=0
    )

    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=8,
        random_state=0
    )
    model.fit(X_train, y_train)

    # 3) Choose two features
    feat1 = "MedInc"
    feat2 = "AveRooms"

    # Check if the features exist, otherwise fall back
    columns = X_train.columns.tolist()
    if feat1 not in columns or feat2 not in columns:
        print(f"Warning: '{feat1}' or '{feat2}' not found. Using first two features instead.")
        feat1, feat2 = columns[0], columns[1]

    # 4) Compute partial dependence
    pd_results = partial_dependence(
        model,
        X_train,
        features=[(feat1, feat2)],
        grid_resolution=5
    )

    # Extract grid and average predictions
    f1_vals = pd_results["values"][0]
    f2_vals = pd_results["values"][1]
    avg_preds = pd_results["average"].reshape(len(f1_vals), len(f2_vals))

    # 5) Plot as layered/grouped bars
    bar_width = 0.15
    x = np.arange(len(f1_vals))

    plt.figure(figsize=(10, 6))
    colors = plt.cm.viridis(np.linspace(0, 1, len(f2_vals)))

    for i, (f2, color) in enumerate(zip(f2_vals, colors)):
        plt.bar(
            x + i * bar_width,
            avg_preds[:, i],
            width=bar_width,
            color=color,
            edgecolor="black",
            label=f"{feat2} = {round(f2, 2)}"
        )

    # X-axis ticks in the center of grouped bars
    plt.xticks(x + bar_width * (len(f2_vals) - 1) / 2, np.round(f1_vals, 2))

    plt.xlabel(feat1)
    plt.ylabel("Partial dependence (avg prediction)")
    plt.title(f"Layered PDP (Bar Flow) for '{feat1}' vs '{feat2}'")
    plt.legend(title=feat2)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()