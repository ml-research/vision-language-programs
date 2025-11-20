import os


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
from matplotlib.patches import Patch

# from eval import eval_max_imgs

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.lines import Line2D

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.lines import Line2D

dataset_names = {
    "bongard-hoi-max-img": "Bongard-HOI",
    "cocologic-max-img": "COCOLogic",
    "CLEVR-Hans3-unconfounded": "CLEVR-Hans",
}

model_names = {
    "InternVL3-8B": "InternVL3-8B",
    "InternVL3-14B": "InternVL3-14B",
    "Qwen2.5-VL-7B-Instruct": "Qwen2.5-VL-7B",
    "Ovis2.5-9B": "Ovis2.5-9B",
}


def plot_max_imgs_aggregate_by_method():
    df = pd.read_csv("results/all_results_max_imgs.csv")

    # Rename first column to "model"
    df.rename(columns={df.columns[0]: "model"}, inplace=True)

    # ---------------------------
    # Step 1: Reshape and preprocess
    # ---------------------------
    df["method"] = df["model"].str.split("_").str[-1]
    df["model_name"] = df["model"].str.rsplit("_", n=1).str[0]

    # Melt wide → long
    df_long = df.melt(
        id_vars=["model", "model_name", "method"],
        var_name="dataset_img",
        value_name="score",
    )

    # Split dataset_img into dataset + img_count
    df_long[["dataset", "img_count"]] = df_long["dataset_img"].str.rsplit(
        "_", n=1, expand=True
    )
    df_long["img_count"] = df_long["img_count"].astype(int) * 2

    # Replace nans with 50 (only if baseline in name)
    df_long["score"] = df_long.apply(
        lambda row: (
            50 if pd.isna(row["score"]) and "baseline" in row["model"] else row["score"]
        ),
        axis=1,
    )

    # ---------------------------
    # Step 2: Aggregate across models within each method
    # ---------------------------
    df_agg = df_long.groupby(["method", "dataset", "img_count"], as_index=False).agg(
        mean_score=("score", "mean"), std_score=("score", "std")
    )  # ← CHANGED

    print(df_agg)

    # ---------------------------
    # Step 3: Plot
    # ---------------------------
    datasets = df_agg["dataset"].unique()
    datasets = [ds for ds in datasets if ds != "bongard-op"]
    methods = ["baseline", "vlp"]

    method_linestyles = {
        "baseline": "dashed",
        "vlp": "solid",
        "Random Guessing": "dotted",
    }
    method_colors = {"baseline": "#1f77b4", "vlp": "#ff7f0e", "Random Guessing": "gray"}

    # set general font sizes
    plt.rcParams.update({"font.size": 16, "axes.titlesize": 18, "axes.labelsize": 16})

    fig, axes = plt.subplots(
        1, len(datasets), figsize=(3 * len(datasets), 4), sharey=True
    )
    if len(datasets) == 1:
        axes = [axes]

    # sort datasets: bongard-hoi-max-img, cocologic-max-img, CLEVR-Hans3-unconfounded
    order = [
        "bongard-hoi-max-img\\",
        "cocologic-max-img\\",
        "CLEVR-Hans3-unconfounded\\",
    ]
    datasets = sorted(datasets, key=lambda x: order.index(x))

    for i, ds in enumerate(datasets):
        ax = axes[i]
        subset = df_agg[df_agg["dataset"] == ds]

        for method in methods:
            data = subset[subset["method"] == method]
            # ax.plot(
            #     data["img_count"],
            #     data["score"],
            #     linestyle=method_linestyles[method],
            #     marker="o",
            #     linewidth=2,
            #     color=method_colors[method],
            #     label=method if i == 0 else "",
            # )
            # Line plot of mean
            ax.plot(
                data["img_count"],
                data["mean_score"],  # ← CHANGED
                linestyle=method_linestyles[method],
                marker="o",
                linewidth=2,
                color=method_colors[method],
                label=method if i == 0 else "",
            )

            # Add standard deviation shading
            ax.fill_between(
                data["img_count"],
                data["mean_score"] - data["std_score"],  # ← CHANGED
                data["mean_score"] + data["std_score"],  # ← CHANGED
                color=method_colors[method],
                alpha=0.2,  # ← CHANGED
            )

        # add random guessing baseline
        random_baseline = 50

        ax.axhline(
            y=random_baseline,
            color="gray",
            linestyle=":",
            linewidth=1.5,
            label="Random Guessing" if i == 0 else "",
        )

        dataset_handle = ds.replace("\\", "")
        dataset_handle = dataset_names.get(dataset_handle, dataset_handle)

        ax.set_title(dataset_handle)
        ax.set_xlabel("Images per sample")
        if i == 0:
            ax.set_ylabel("Accuracy (%)")
        ax.set_xticks(sorted(df_agg["img_count"].unique()))
        ax.grid(True, linestyle="--", alpha=0.5)

    method_names = {
        "baseline": "Baseline (All)",
        "vlp": "VLP (All)",
        "Random Guessing": "Random Guessing",
    }

    # Legend
    method_handles = [
        Line2D(
            [0],
            [0],
            color=method_colors[m],
            lw=3,
            linestyle=method_linestyles[m],
            label=method_names[m],
        )
        for m in methods + ["Random Guessing"]
    ]
    # fig.legend(
    #     handles=method_handles,
    #     loc="center left",
    #     bbox_to_anchor=(1.02, 0.5),
    #     frameon=True,
    #     title="Methods",
    # )
    fig.legend(
        handles=method_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.1),
        ncol=len(method_handles),
        frameon=True,
        # title="Methods",
    )

    # y-axis limit
    for ax in axes:
        ax.set_ylim(35, 95)

    plt.tight_layout()
    plt.savefig("results/more_imgs_plot_by_method.png", bbox_inches="tight")
    plt.savefig("results/more_imgs_plot_by_method.pdf", bbox_inches="tight")
    plt.show()


def plot_max_imgs():

    df = pd.read_csv("results/max_imgs_for_plotting.csv")

    # name first column to "model"
    df.rename(columns={df.columns[0]: "model"}, inplace=True)

    # ---------------------------
    # Step 2: Reshape the data
    # ---------------------------
    # Split model into base name and method
    df["method"] = df["model"].str.split("\_").str[-1]
    df["model_name"] = df["model"].str.rsplit("\_", n=1).str[0]

    # Melt wide → long format
    df_long = df.melt(
        id_vars=["model", "model_name", "method"],
        var_name="dataset_img",
        value_name="score",
    )

    # Split dataset_img into dataset + img_count
    df_long[["dataset", "img_count"]] = df_long["dataset_img"].str.rsplit(
        "\_", n=1, expand=True
    )
    df_long["img_count"] = df_long["img_count"].astype(int) * 2

    print(df_long)

    # ---------------------------
    # Step 3: Plot
    # ---------------------------
    datasets = df_long["dataset"].unique()
    # remove dataset "bonggard_op" if present
    datasets = [ds for ds in datasets if ds != "bongard-op"]
    models = df_long["model_name"].unique()
    methods = ["baseline", "vlp"]

    method_linestyles = {"baseline": "dashed", "vlp": "solid"}

    # Assign colors to models
    prop_cycle = plt.rcParams["axes.prop_cycle"].by_key().get("color", [])
    while len(prop_cycle) < len(models):
        prop_cycle += prop_cycle
    # color map set 2
    color_map = sns.color_palette("Set2", n_colors=len(models))
    model_to_color = {m: color_map[i] for i, m in enumerate(models)}

    fig, axes = plt.subplots(
        1, len(datasets), figsize=(3 * len(datasets), 4), sharey=True
    )

    if len(datasets) == 1:
        axes = [axes]

    for i, ds in enumerate(datasets):
        ax = axes[i]
        subset = df_long[df_long["dataset"] == ds]

        for model in models:
            for method in methods:
                data = subset[
                    (subset["model_name"] == model) & (subset["method"] == method)
                ]
                ax.plot(
                    data["img_count"],
                    data["score"],
                    linestyle=method_linestyles[method],
                    marker="o",
                    linewidth=2,
                    color=model_to_color[model],
                    label=f"{model} ({method})" if i == 0 else "",
                )

        ax.set_title(dataset_names[ds])
        ax.set_xlabel("Images per sample")
        if i == 0:
            ax.set_ylabel("Accuracy (%)")
        ax.set_xticks(sorted(df_long["img_count"].unique()))
        ax.grid(True, linestyle="--", alpha=0.5)

    # Custom legends: models (colors), methods (line styles)
    model_handles = [
        Line2D([0], [0], color=model_to_color[m], lw=3, label=model_names[m])
        for m in models
    ]

    method_names = {"baseline": "Baseline (All)", "vlp": "VLP (All)"}

    method_handles = [
        Line2D(
            [0],
            [0],
            color="#646464",
            lw=3,
            linestyle=method_linestyles[k],
            label=method_names[k],
        )
        for k in methods
    ]

    # # Image counts legend
    # fig.legend(
    #     handles=model_handles,
    #     loc="center left",
    #     bbox_to_anchor=(1.02, 0.6),
    #     frameon=True,
    #     # title="Models",
    # )

    # Methods legend
    fig.legend(
        handles=method_handles,
        loc="center left",
        bbox_to_anchor=(1.02, 0.3),
        frameon=True,
        # title="Methods",
    )

    # set y-axis limit to 0-100
    for ax in axes:
        ax.set_ylim(30, 90)

    # fig.suptitle("Baseline vs VLP across image counts per dataset", y=1.02)
    plt.tight_layout()
    plt.savefig("results/more_imgs_plot.png", bbox_inches="tight")
    plt.savefig("results/more_imgs_plot.pdf", bbox_inches="tight")

    plt.show()


def plot_max_imgs_bars():

    df = pd.read_csv("results/all_results_for_plotting.csv")

    # name first column to "model"
    df.rename(columns={df.columns[0]: "model"}, inplace=True)

    # ---------------------------
    # Step 2: Reshape the data
    # ---------------------------
    # Split model into base name and method
    df["method"] = df["model"].str.split("_").str[-1]
    df["model_name"] = df["model"].str.rsplit("_", n=1).str[0]

    # Melt wide → long format
    df_long = df.melt(
        id_vars=["model", "model_name", "method"],
        var_name="dataset_img",
        value_name="score",
    )

    # Split dataset_img into dataset + img_count
    df_long[["dataset", "img_count"]] = df_long["dataset_img"].str.rsplit(
        "_", n=1, expand=True
    )
    df_long["img_count"] = df_long["img_count"].astype(int) * 2

    # ---------------------------
    # Step 3: Plot as grouped bars by model
    # ---------------------------
    datasets = df_long["dataset"].unique()
    datasets = [ds for ds in datasets if ds != "bongard-op"]

    models = list(df_long["model_name"].unique())
    methods = ["baseline", "vlp"]
    img_counts = sorted(df_long["img_count"].unique())

    # Colors for image counts
    prop_cycle = plt.rcParams["axes.prop_cycle"].by_key().get("color", [])
    while len(prop_cycle) < len(img_counts):
        prop_cycle += prop_cycle
    img_to_color = {c: prop_cycle[i] for i, c in enumerate(img_counts)}

    # Hatches for methods
    method_hatches = {"baseline": "//", "vlp": "\\\\"}

    fig, axes = plt.subplots(
        1, len(datasets), figsize=(5 * len(datasets), 5), sharey=True
    )
    if len(datasets) == 1:
        axes = [axes]

    for i, ds in enumerate(datasets):
        ax = axes[i]
        subset = df_long[df_long["dataset"] == ds].copy()

        # group geometry
        n_models = len(models)
        n_series = len(img_counts) * len(methods)
        x = np.arange(n_models)
        total_group_width = 0.82
        bar_width = total_group_width / n_series
        left_edge = -total_group_width / 2

        series_order = [(c, m) for c in img_counts for m in methods]

        for s_idx, (img_c, method) in enumerate(series_order):
            offs = left_edge + (s_idx + 0.5) * bar_width
            xpos = x + offs
            heights = []
            for model in models:
                row = subset[
                    (subset["model_name"] == model)
                    & (subset["method"] == method)
                    & (subset["img_count"] == img_c)
                ]
                heights.append(row["score"].values[0] if not row.empty else np.nan)

            ax.bar(
                xpos,
                heights,
                width=bar_width * 0.95,
                color=img_to_color[img_c],
                hatch=method_hatches[method],
                edgecolor="black",
                linewidth=0.6,
            )

        # cosmetics
        ax.set_title(ds)
        ax.set_xlabel("Model")
        if i == 0:
            ax.set_ylabel("Accuracy (%)")
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=30, ha="right")
        ax.grid(True, axis="y", linestyle="--", alpha=0.5)
        ax.set_ylim(30, 90)

    # Legends: image counts (colors), methods (hatches)
    img_handles = [
        Patch(facecolor=img_to_color[c], edgecolor="black", label=f"{c} imgs")
        for c in img_counts
    ]
    method_handles = [
        Patch(facecolor="white", edgecolor="black", hatch=method_hatches[k], label=k)
        for k in methods
    ]

    fig.legend(
        handles=img_handles,
        loc="lower center",
        ncol=len(img_counts),
        bbox_to_anchor=(0.5, -0.02),
        frameon=False,
        title="Image counts",
    )
    fig.legend(
        handles=method_handles,
        loc="lower center",
        ncol=len(methods),
        bbox_to_anchor=(0.5, -0.12),
        frameon=False,
        title="Methods",
    )

    plt.tight_layout()
    plt.savefig("results/more_imgs_bar_by_model.png", bbox_inches="tight")
    plt.savefig("results/more_imgs_bar_by_model.pdf", bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    # eval_max_imgs("naive_weighted")
    # eval_max_imgs("n_occurrence")
    plot_max_imgs_aggregate_by_method()
