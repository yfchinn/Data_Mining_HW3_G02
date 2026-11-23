import ast
import json
from collections import Counter
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


RAW_RECIPES_PATH = Path("data/RAW_recipes.csv")
RAW_INTERACTIONS_PATH = Path("data/RAW_interactions.csv")
OUTPUT_DIR = Path("outputs/eda")

plt.rcParams.update(
    {
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "figure.dpi": 300,
    }
)


def compute_long_tail_metrics(counts: np.ndarray) -> dict:
    sorted_desc = np.sort(counts)[::-1]
    total_inter = int(sorted_desc.sum())
    n_entities = len(sorted_desc)

    metrics = {
        "total_entities": int(n_entities),
        "total_interactions": total_inter,
        "mean": round(float(sorted_desc.mean()), 2),
        "median": float(np.median(sorted_desc)),
        "p90": float(np.percentile(sorted_desc, 90)),
        "max": int(sorted_desc[0]),
    }

    for pct in (1, 5, 10, 20):
        k = max(1, int(np.ceil(n_entities * pct / 100.0)))
        share = float(sorted_desc[:k].sum()) / total_inter * 100.0
        metrics[f"top_{pct}pct_share_pct"] = round(share, 2)

    for thresh in (2, 5, 10, 20):
        cnt = int((sorted_desc < thresh).sum())
        metrics[f"lt_{thresh}_inter_count"] = cnt
        metrics[f"lt_{thresh}_inter_pct"] = round(cnt / n_entities * 100.0, 2)

    return metrics


def plot_rating_distribution(interactions: pd.DataFrame, out_path: Path) -> None:
    counts = interactions["rating"].value_counts().sort_index()
    total = counts.sum()

    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    bars = ax.bar(counts.index.astype(str), counts.values, color="#2B4C7E", width=0.62)
    for bar, val in zip(bars, counts.values):
        pct = val / total * 100
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{pct:.1f}%",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    ax.set_xlabel("Rating Score (Stars)")
    ax.set_ylabel("Number of Interactions (Count)")
    ax.set_title("Rating Distribution in RAW_interactions.csv")
    ax.set_ylim(0, counts.max() * 1.13)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def plot_long_tail_and_lorenz(item_counts: np.ndarray, out_path: Path) -> None:
    sorted_desc = np.sort(item_counts)[::-1]
    ranks = np.arange(1, len(sorted_desc) + 1)

    sorted_asc = np.sort(item_counts)
    cum_items_pct = np.arange(1, len(sorted_asc) + 1) / len(sorted_asc) * 100.0
    cum_inter_pct = np.cumsum(sorted_asc) / sorted_asc.sum() * 100.0

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.3))

    # Left: Log-log rank-frequency plot
    axes[0].loglog(ranks, sorted_desc, color="#C0392B", linewidth=1.8)
    axes[0].set_xlabel("Recipe Popularity Rank (Log Scale)")
    axes[0].set_ylabel("Positive Interactions per Recipe (Log Scale)")
    axes[0].set_title("Recipe Interaction Long-Tail Distribution")
    axes[0].grid(True, which="both", linestyle="--", alpha=0.35)

    # Right: Lorenz / cumulative interaction curve
    axes[1].plot(
        cum_items_pct,
        cum_inter_pct,
        color="#2B4C7E",
        linewidth=2.0,
        label="Observed Distribution",
    )
    axes[1].plot(
        [0, 100],
        [0, 100],
        color="#7F8C8D",
        linestyle="--",
        linewidth=1.2,
        label="Uniform Equality",
    )
    axes[1].set_xlabel("Cumulative Share of Recipes, Least to Most Popular (%)")
    axes[1].set_ylabel("Cumulative Share of Positive Interactions (%)")
    axes[1].set_title("Cumulative Interaction Concentration (Lorenz Curve)")
    axes[1].legend(frameon=True)
    axes[1].grid(True, linestyle="--", alpha=0.35)

    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def plot_user_and_ingredient_dist(
    user_counts: np.ndarray, ingr_lengths: np.ndarray, out_path: Path
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.3))

    # Left: User interaction histogram (clipped at 50 for readability)
    clipped_u = np.clip(user_counts, 1, 50)
    axes[0].hist(clipped_u, bins=25, color="#2E86AB", edgecolor="white", alpha=0.9)
    axes[0].set_xlabel("Positive Interactions per User (Count, Capped at 50+)")
    axes[0].set_ylabel("Number of Users (Count)")
    axes[0].set_title("User Activity Distribution")
    axes[0].grid(axis="y", linestyle="--", alpha=0.35)

    # Right: Ingredients per recipe histogram (clipped at 30)
    clipped_i = np.clip(ingr_lengths, 0, 30)
    axes[1].hist(clipped_i, bins=30, color="#D97706", edgecolor="white", alpha=0.9)
    axes[1].set_xlabel("Number of Ingredients per Recipe (Count, Capped at 30+)")
    axes[1].set_ylabel("Number of Recipes (Count)")
    axes[1].set_title("Ingredients per Recipe Distribution")
    axes[1].grid(axis="y", linestyle="--", alpha=0.35)

    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading {RAW_RECIPES_PATH} and {RAW_INTERACTIONS_PATH}...")
    recipes = pd.read_csv(RAW_RECIPES_PATH)
    interactions = pd.read_csv(RAW_INTERACTIONS_PATH)
    interactions["date"] = pd.to_datetime(interactions["date"], errors="coerce")

    valid_recipe_ids = set(recipes["id"].dropna().astype(int))
    dedup = (
        interactions[interactions["recipe_id"].isin(valid_recipe_ids)]
        .sort_values("date")
        .drop_duplicates(subset=["user_id", "recipe_id"], keep="last")
    )
    positives = dedup[dedup["rating"] >= 4]

    item_counts = positives["recipe_id"].value_counts().values
    user_counts = positives["user_id"].value_counts().values

    item_lt = compute_long_tail_metrics(item_counts)
    user_lt = compute_long_tail_metrics(user_counts)

    # Ingredient statistics
    ingr_counter: Counter = Counter()
    ingr_lengths = []
    for raw_val in recipes["ingredients"]:
        try:
            parsed = ast.literal_eval(raw_val) if isinstance(raw_val, str) else []
            if isinstance(parsed, list):
                ingr_lengths.append(len(parsed))
                for ing in parsed:
                    ingr_counter[str(ing).strip().lower()] += 1
        except (ValueError, SyntaxError):
            continue

    ingr_arr = np.array(ingr_lengths, dtype=int)
    ingr_stats = {
        "unique_raw_ingredients": int(len(ingr_counter)),
        "mean_ingredients_per_recipe": round(float(ingr_arr.mean()), 2),
        "median_ingredients_per_recipe": float(np.median(ingr_arr)),
        "max_ingredients_per_recipe": int(ingr_arr.max()),
        "top_15_ingredients": ingr_counter.most_common(15),
    }

    # Generate plots
    plot_rating_distribution(interactions, OUTPUT_DIR / "rating_distribution.png")
    plot_long_tail_and_lorenz(item_counts, OUTPUT_DIR / "recipe_long_tail.png")
    plot_user_and_ingredient_dist(
        user_counts, ingr_arr, OUTPUT_DIR / "user_and_ingredient_dist.png"
    )

    stats = {
        "item_long_tail_positives_ge4": item_lt,
        "user_activity_positives_ge4": user_lt,
        "ingredient_statistics": ingr_stats,
    }
    out_json = OUTPUT_DIR / "characteristics_stats.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    print("\n=== Item Long-Tail Summary (dedup, rating >= 4) ===")
    for k, v in item_lt.items():
        print(f"  {k}: {v}")

    print("\n=== User Activity Summary (dedup, rating >= 4) ===")
    for k, v in user_lt.items():
        print(f"  {k}: {v}")

    print("\n=== Ingredient Summary ===")
    print(f"  Unique raw ingredient strings: {ingr_stats['unique_raw_ingredients']:,}")
    print(
        f"  Mean / Median ingredients per recipe: "
        f"{ingr_stats['mean_ingredients_per_recipe']} / "
        f"{ingr_stats['median_ingredients_per_recipe']}"
    )
    print(f"\nSaved plots and stats to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()