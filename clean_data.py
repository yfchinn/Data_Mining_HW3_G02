import argparse
import ast
import json
from pathlib import Path
import pandas as pd


def calc_sparsity(n_users: int, n_items: int, n_inter: int) -> float:
    if n_users == 0 or n_items == 0:
        return 1.0
    return 1.0 - (n_inter / (n_users * n_items))


def summarize_step(df: pd.DataFrame, step_name: str) -> dict:
    n_users = int(df["user_id"].nunique())
    n_items = int(df["recipe_id"].nunique())
    n_inter = int(len(df))
    sparsity = calc_sparsity(n_users, n_items, n_inter)
    date_min = str(df["date"].min().date()) if n_inter > 0 else None
    date_max = str(df["date"].max().date()) if n_inter > 0 else None
    return {
        "step": step_name,
        "users": n_users,
        "items": n_items,
        "interactions": n_inter,
        "sparsity_pct": round(sparsity * 100, 4),
        "avg_inter_per_user": round(n_inter / n_users, 2) if n_users else 0.0,
        "avg_inter_per_item": round(n_inter / n_items, 2) if n_items else 0.0,
        "date_min": date_min,
        "date_max": date_max,
    }


def has_valid_ingredients(val) -> bool:
    try:
        parsed = ast.literal_eval(val) if isinstance(val, str) else val
        return isinstance(parsed, list) and len(parsed) > 0
    except (ValueError, SyntaxError):
        return False


def iterative_k_core(
    df: pd.DataFrame, min_user: int, min_item: int
) -> tuple[pd.DataFrame, int]:
    cur = df.copy()
    iterations = 0
    while True:
        prev_len = len(cur)
        iterations += 1
        if min_user > 0:
            u_cnt = cur["user_id"].value_counts()
            cur = cur[cur["user_id"].isin(u_cnt[u_cnt >= min_user].index)]
        if min_item > 0:
            i_cnt = cur["recipe_id"].value_counts()
            cur = cur[cur["recipe_id"].isin(i_cnt[i_cnt >= min_item].index)]
        if len(cur) == prev_len or len(cur) == 0:
            break
    return cur, iterations


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Clean Food.com RAW_recipes.csv and RAW_interactions.csv."
    )
    parser.add_argument(
        "--raw-recipes",
        type=Path,
        default=Path("data/RAW_recipes.csv"),
        help="Path to RAW_recipes.csv",
    )
    parser.add_argument(
        "--raw-interactions",
        type=Path,
        default=Path("data/RAW_interactions.csv"),
        help="Path to RAW_interactions.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed"),
        help="Directory to save cleaned CSV files and stats JSON",
    )
    parser.add_argument(
        "--min-rating",
        type=int,
        default=4,
        help="Minimum rating threshold for positive interactions (default: 4)",
    )
    parser.add_argument(
        "--min-user-inter",
        type=int,
        default=5,
        help="Minimum positive interactions per user for k-core filtering (default: 5)",
    )
    parser.add_argument(
        "--min-item-inter",
        type=int,
        default=5,
        help="Minimum positive interactions per recipe for k-core filtering (default: 5)",
    )
    parser.add_argument(
        "--keep-review",
        action="store_true",
        help="Keep raw review text column in interactions_clean.csv (default: drop to save space)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading {args.raw_recipes} and {args.raw_interactions}...")
    recipes = pd.read_csv(args.raw_recipes)
    interactions = pd.read_csv(args.raw_interactions)
    interactions["date"] = pd.to_datetime(interactions["date"], errors="coerce")

    stats_steps = [summarize_step(interactions, "0. Raw interactions")]

    # Step 1: Clean recipes (drop missing name, n_steps <= 0, invalid/empty ingredients)
    n_raw_recipes = len(recipes)
    valid_name_mask = recipes["name"].notna() & (
        recipes["name"].astype(str).str.strip() != ""
    )
    valid_steps_mask = recipes["n_steps"] > 0
    valid_ingr_mask = recipes["ingredients"].apply(has_valid_ingredients)

    recipes_valid = recipes[valid_name_mask & valid_steps_mask & valid_ingr_mask].copy()
    valid_recipe_ids = set(recipes_valid["id"].astype(int))

    # Align interactions with valid recipes and valid dates
    inter = interactions[
        interactions["recipe_id"].isin(valid_recipe_ids)
        & interactions["date"].notna()
    ].copy()
    stats_steps.append(
        summarize_step(inter, "1. Align with valid recipes & non-null dates")
    )

    # Step 2: Deduplicate (user_id, recipe_id), keeping the latest interaction
    inter = (
        inter.sort_values("date")
        .drop_duplicates(subset=["user_id", "recipe_id"], keep="last")
        .copy()
    )
    stats_steps.append(
        summarize_step(inter, "2. Deduplicate (user_id, recipe_id) keep latest")
    )

    # Step 3: Filter positive interactions
    inter = inter[inter["rating"] >= args.min_rating].copy()
    stats_steps.append(
        summarize_step(inter, f"3. Positive interactions (rating >= {args.min_rating})")
    )

    # Step 4: Iterative k-core filtering
    inter, kcore_iters = iterative_k_core(
        inter, min_user=args.min_user_inter, min_item=args.min_item_inter
    )
    stats_steps.append(
        summarize_step(
            inter,
            f"4. Iterative k-core (user >= {args.min_user_inter}, item >= {args.min_item_inter})",
        )
    )

    # Finalize cleaned interactions
    inter = inter.sort_values(["user_id", "date", "recipe_id"]).reset_index(drop=True)
    inter["date"] = inter["date"].dt.strftime("%Y-%m-%d")

    keep_cols = ["user_id", "recipe_id", "date", "rating"]
    if args.keep_review and "review" in inter.columns:
        keep_cols.append("review")
    inter_out = inter[keep_cols]

    # Align cleaned recipes to the final remaining recipe_ids
    final_recipe_ids = set(inter_out["recipe_id"].unique())
    recipes_out = (
        recipes_valid[recipes_valid["id"].isin(final_recipe_ids)]
        .sort_values("id")
        .reset_index(drop=True)
    )

    # Verify strict consistency
    assert set(inter_out["recipe_id"].unique()) == set(recipes_out["id"].unique())
    assert inter_out.duplicated(subset=["user_id", "recipe_id"]).sum() == 0
    assert (inter_out["user_id"].value_counts() >= args.min_user_inter).all()
    assert (inter_out["recipe_id"].value_counts() >= args.min_item_inter).all()

    # Yearly distribution of cleaned interactions (helpful for temporal split design)
    yearly_counts = (
        pd.to_datetime(inter_out["date"])
        .dt.year.value_counts()
        .sort_index()
        .to_dict()
    )
    yearly_counts = {str(int(k)): int(v) for k, v in yearly_counts.items()}

    # Save CSV outputs
    inter_path = args.output_dir / "interactions_clean.csv"
    recipes_path = args.output_dir / "recipes_clean.csv"
    stats_path = args.output_dir / "clean_stats.json"

    inter_out.to_csv(inter_path, index=False)
    recipes_out.to_csv(recipes_path, index=False)

    summary = {
        "config": {
            "min_rating": args.min_rating,
            "min_user_inter": args.min_user_inter,
            "min_item_inter": args.min_item_inter,
            "keep_review": args.keep_review,
            "kcore_iterations_to_converge": kcore_iters,
        },
        "recipe_cleaning": {
            "raw_recipes": int(n_raw_recipes),
            "valid_recipes_before_kcore": int(len(recipes_valid)),
            "final_recipes_after_kcore": int(len(recipes_out)),
        },
        "step_by_step_stats": stats_steps,
        "cleaned_yearly_interaction_counts": yearly_counts,
    }
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("\n=== Cleaning Pipeline Summary ===")
    print(pd.DataFrame(stats_steps).to_string(index=False))
    print(f"\nk-core converged in {kcore_iters} iterations.")
    print(
        f"Saved:\n  - {inter_path} ({len(inter_out):,} rows)\n"
        f"  - {recipes_path} ({len(recipes_out):,} rows)\n"
        f"  - {stats_path}"
    )


if __name__ == "__main__":
    main()