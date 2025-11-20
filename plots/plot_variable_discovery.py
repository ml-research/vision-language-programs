import pandas as pd
import matplotlib.pyplot as plt


# colormap set2
colors = plt.get_cmap("Set2").colors


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset_rename = {
    "bongard-op": "Bongard-OpenWorld",
    "bongard-hoi": "Bongard-HOI",
    "bongard-rwr": "Bongard-RWR",
    "cocologic": "COCOLogic",
    "CLEVR-Hans3-unconfounded": "CLEVR-Hans3",
}


def plot_variable_discovery_results(keyword="objects", type="object"):
    big_type = (
        "Object" if type == "object" else "Property" if type == "property" else "Action"
    )
    big_type_plural = (
        "Objects"
        if type == "object"
        else "Properties" if type == "property" else "Actions"
    )
    # Load CSV file
    file_path = f"results/variable_discovery/variable_discovery_results_{keyword}.csv"
    df = pd.read_csv(file_path, skipinitialspace=True)

    # Clean column names
    df.columns = [col.strip() for col in df.columns]

    # Select relevant columns (copy to avoid SettingWithCopyWarning)
    metric_col = f"Hit {big_type} Ratio"
    needed_cols = ["Dataset", "Model", metric_col]
    missing = [c for c in needed_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in CSV: {missing}")

    data = df[needed_cols].copy()

    # Rename dataset values
    data["Dataset"] = data["Dataset"].map(dataset_rename).fillna(data["Dataset"])

    # Categorize for consistent order
    desired_order = ["Bongard-OpenWorld", "Bongard-HOI", "COCOLogic", "CLEVR-Hans3"]
    data["Dataset"] = pd.Categorical(
        data["Dataset"], categories=desired_order, ordered=True
    )

    # Pivot: rows=datasets, cols=models
    pivot_df = data.pivot(index="Dataset", columns="Model", values=metric_col)

    # Drop models that are entirely NaN
    pivot_df = pivot_df.dropna(axis=1, how="all")

    # Drop datasets where all model values are NaN or 0
    def all_zero_or_nan(row):
        vals = row.values.astype(float)
        # treat NaNs as 0 for the "all-zero" check
        return np.nan_to_num(vals, nan=0.0).sum() == 0.0

    pivot_df = pivot_df[~pivot_df.apply(all_zero_or_nan, axis=1)]

    # If no data remains, bail out gracefully
    if pivot_df.empty:
        print("No non-zero results to plot for the given keyword and type.")
        return

    # Reindex to keep the desired dataset order and drop unused categories
    pivot_df = pivot_df.reindex([d for d in desired_order if d in pivot_df.index])

    # Flexible figure width: ~1.6 inches per dataset, with a floor and ceiling
    n_datasets = len(pivot_df.index)
    per_dataset_width = 1.6
    base_width = 2.0  # room for y-axis labels
    fig_width = max(6.0, min(18.0, base_width + per_dataset_width * n_datasets))
    fig_height = 5.0

    ax = pivot_df.plot(kind="bar", figsize=(fig_width, fig_height), color=colors)

    # Labels and styling
    ax.set_ylabel(f"Hit {big_type} Ratio")
    ax.set_title(f"{big_type_plural}")
    ax.set_ylim(0, 1)
    ax.grid(axis="y", linestyle=":", linewidth=0.8)
    plt.xticks(rotation=30, ha="right")
    plt.legend(title="Model", ncols=2 if pivot_df.shape[1] > 6 else 1, frameon=False)
    plt.tight_layout()

    # Save
    out_base = f"results/variable_discovery/variable_discovery_results_{big_type_plural.lower()}_{keyword}"
    plt.savefig(out_base + ".png", dpi=300, bbox_inches="tight")
    plt.savefig(out_base + ".pdf", bbox_inches="tight")

    plt.show()


def plot_variable_discovery_results_(keyword="objects", type="object"):

    big_type = (
        "Object" if type == "object" else "Property" if type == "property" else "Action"
    )
    big_type_plural = (
        "Objects"
        if type == "object"
        else "Properties" if type == "property" else "Actions"
    )

    # Load CSV file
    file_path = f"results/variable_discovery/variable_discovery_results_{keyword}.csv"  # ← replace this with the actual file path
    df = pd.read_csv(file_path, skipinitialspace=True)

    # Ensure column names are clean
    df.columns = [col.strip() for col in df.columns]

    # Select relevant columns
    data = df[["Dataset", "Model", f"Hit {big_type} Ratio"]]

    # rename dataset values
    data.loc[:, "Dataset"] = data["Dataset"].map(dataset_rename).fillna(data["Dataset"])

    # dataset order: Bongard-OpenWorld, Bongard-HOI,  COCOLogic, CLEVR-Hans3
    data["Dataset"] = pd.Categorical(
        data["Dataset"],
        categories=[
            "Bongard-OpenWorld",
            "Bongard-HOI",
            "COCOLogic",
            "CLEVR-Hans3",
        ],
    )

    # Pivot the table to have models as columns, datasets as rows
    pivot_df = data.pivot(
        index="Dataset", columns="Model", values=f"Hit {big_type} Ratio"
    )

    # Plot grouped bar chart
    ax = pivot_df.plot(kind="bar", figsize=(10, 6), color=colors)

    # Labels and title
    plt.ylabel(f"Hit {big_type} Ratio")
    plt.title(f"{big_type_plural}")
    plt.xticks(rotation=30, ha="right")
    plt.legend(title="Model")
    plt.tight_layout()

    # add horizontal grid lines
    plt.grid(axis="y")

    # set y axis to 0-100
    plt.ylim(0, 1)

    # save plot as png
    plt.savefig(
        f"results/variable_discovery/variable_discovery_results_{big_type_plural.lower()}_{keyword}.png"
    )
    plt.savefig(
        f"results/variable_discovery/variable_discovery_results_{big_type_plural.lower()}_{keyword}.pdf"
    )

    # Show the plot
    plt.show()


def plot_variable_discovery_results_properties(keyword="objects"):

    # Load CSV file
    file_path = f"results/variable_discovery/variable_discovery_results_{keyword}.csv"  # ← replace this with the actual file path
    df = pd.read_csv(file_path, skipinitialspace=True)

    # Ensure column names are clean
    df.columns = [col.strip() for col in df.columns]

    # Select relevant columns
    data = df[["Dataset", "Model", "Hit Property Ratio"]]

    # Pivot the table to have models as columns, datasets as rows
    pivot_df = data.pivot(index="Dataset", columns="Model", values="Hit Property Ratio")

    # Plot grouped bar chart
    ax = pivot_df.plot(kind="bar", figsize=(10, 6), color=colors)

    # Labels and title
    plt.ylabel("Hit Property Ratio")
    plt.title("Hit Property Ratio by Dataset and Model")
    plt.xticks(rotation=30, ha="right")
    plt.legend(title="Model")
    plt.tight_layout()

    # add horizontal grid lines
    plt.grid(axis="y")

    # set y axis to 0-100
    plt.ylim(0, 1)

    # save plot as png
    plt.savefig(
        f"results/variable_discovery/variable_discovery_results_properties_{keyword}.png"
    )
    plt.savefig(
        f"results/variable_discovery/variable_discovery_results_properties_{keyword}.pdf"
    )

    # Show the plot
    plt.show()


def plot_variable_discovery_results_actions(keyword="objects"):

    # Load CSV file
    file_path = f"results/variable_discovery/variable_discovery_results_{keyword}.csv"  # ← replace this with the actual file path
    df = pd.read_csv(file_path, skipinitialspace=True)

    # Ensure column names are clean
    df.columns = [col.strip() for col in df.columns]

    # Select relevant columns
    data = df[["Dataset", "Model", "Hit Action Ratio"]]

    # Pivot the table to have models as columns, datasets as rows
    pivot_df = data.pivot(index="Dataset", columns="Model", values="Hit Action Ratio")

    # Plot grouped bar chart
    ax = pivot_df.plot(kind="bar", figsize=(10, 6), color=colors)

    # Labels and title
    plt.ylabel("Hit Action Ratio")
    plt.title("Hit Action Ratio by Dataset and Model {}".format(keyword))
    plt.xticks(rotation=30, ha="right")
    plt.legend(title="Model")
    plt.tight_layout()

    # add horizontal grid lines
    plt.grid(axis="y")

    # set y axis to 0-100
    plt.ylim(0, 1)

    # save plot as png
    plt.savefig(
        f"results/variable_discovery/variable_discovery_results_actions_{keyword}.png"
    )
    plt.savefig(
        f"results/variable_discovery/variable_discovery_results_actions_{keyword}.pdf"
    )

    # Show the plot
    plt.show()


if __name__ == "__main__":
    for keyword in [
        "combi",
    ]:
        plot_variable_discovery_results(keyword, type="object")
        plot_variable_discovery_results(keyword, type="property")
        plot_variable_discovery_results(keyword, type="action")
