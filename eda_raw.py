import ast
import json
from pathlib import Path
import numpy as np
import pandas as pd


RAW_RECIPES_PATH = Path("data/RAW_recipes.csv")
RAW_INTERACTIONS_PATH = Path("data/RAW_interactions.csv")
OUTPUT_DIR = Path("outputs/eda")


def calc_sparsity(n_users: int, n_items: int, n_inter: int) -> float:
    if n_users == 0 or n_items == 0:
        return 1.0
    return 1.0 - (n_inter / (n_users * n_items))


def summarize_df(df: pd.DataFrame, name: str) -> dict:
    n_users = int(df["user_id"].nunique())
    n_items = int(df["recipe_id"].nunique())
    n_inter = int(len(df))
    sparsity = calc_sparsity(n_users, n_items, n_inter)
    return {
        "setting": name,
        "users": n_users,
        "items": n_items,
        "interactions": n_inter,
        "sparsity_pct": round(sparsity * 100, 4),
        "avg_inter_per_user": round(n_inter / n_users, 2) if n_users else 0.0,
        "avg_inter_per_item": round(n_inter / n_items, 2) if n_items else 0.0,
    }


def k_core_filter(df: pd.DataFrame, min_user: int, min_item: int) -> pd.DataFrame:
    cur = df.copy()
    while True:
        prev_len = len(cur)
        if min_user > 0:
            u_cnt = cur["user_id"].value_counts()
            cur = cur[cur["user_id"].isin(u_cnt[u_cnt >= min_user].index)]
        if min_item > 0:
            i_cnt = cur["recipe_id"].value_counts()
            cur = cur[cur["recipe_id"].isin(i_cnt[i_cnt >= min_item].index)]
        if len(cur) == prev_len or len(cur) == 0:
            break
    return cur


def one_shot_filter(df: pd.DataFrame, min_user: int, min_item: int) -> pd.DataFrame:
    cur = df.copy()
    if min_user > 0:
        u_cnt = cur["user_id"].value_counts()
        cur = cur[cur["user_id"].isin(u_cnt[u_cnt >= min_user].index)]
    if min_item > 0:
        i_cnt = cur["recipe_id"].value_counts()
        cur = cur[cur["recipe_id"].isin(i_cnt[i_cnt >= min_item].index)]
    return cur


def check_recipes(recipes: pd.DataFrame) -> dict:
    empty_ingr = 0
    invalid_ingr = 0
    for val in recipes["ingredients"]:
        try:
            parsed = ast.literal_eval(val) if isinstance(val, str) else val
            if not isinstance(parsed, list) or len(parsed) == 0:
                empty_ingr += 1
        except (ValueError, SyntaxError):
            invalid_ingr += 1

    return {
        "total_raw_recipes": int(len(recipes)),
        "unique_recipe_ids": int(recipes["id"].nunique()),
        "missing_name": int(recipes["name"].isna().sum()),
        "empty_ingredients": int(empty_ingr),
        "invalid_ingredients_format": int(invalid_ingr),
        "minutes_le_zero": int((recipes["minutes"] <= 0).sum()),
        "minutes_gt_1440": int((recipes["minutes"] > 1440).sum()),
        "n_steps_le_zero": int((recipes["n_steps"] <= 0).sum()),
    }


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading {RAW_RECIPES_PATH} and {RAW_INTERACTIONS_PATH}...")
    recipes = pd.read_csv(RAW_RECIPES_PATH)
    interactions = pd.read_csv(RAW_INTERACTIONS_PATH)
    interactions["date"] = pd.to_datetime(interactions["date"], errors="coerce")

    # 1. Raw recipe diagnostics
    recipe_diag = check_recipes(recipes)
    valid_recipe_ids = set(recipes["id"].dropna().astype(int))

    # 2. Raw interaction diagnostics
    n_raw_inter = len(interactions)
    n_invalid_dates = int(interactions["date"].isna().sum())
    n_orphan_inter = int((~interactions["recipe_id"].isin(valid_recipe_ids)).sum())
    dup_mask = interactions.duplicated(subset=["user_id", "recipe_id"], keep=False)
    n_dup_rows = int(dup_mask.sum())
    n_dup_pairs = int(
        interactions[dup_mask][["user_id", "recipe_id"]].drop_duplicates().shape[0]
    )

    rating_counts = (
        interactions["rating"]
        .value_counts()
        .sort_index()
        .to_dict()
    )
    rating_dist = {
        str(int(k)): {
            "count": int(v),
            "pct": round(v / n_raw_inter * 100, 2),
        }
        for k, v in rating_counts.items()
    }

    raw_diag = {
        "date_min": str(interactions["date"].min().date()),
        "date_max": str(interactions["date"].max().date()),
        "invalid_dates": n_invalid_dates,
        "orphan_interactions_not_in_recipes": n_orphan_inter,
        "duplicate_user_recipe_rows": n_dup_rows,
        "duplicate_user_recipe_pairs": n_dup_pairs,
        "rating_distribution": rating_dist,
    }

    # 3. Filtering threshold sensitivity table
    # Base cleaning: valid recipe + deduplicate (keep latest by date)
    dedup = (
        interactions[interactions["recipe_id"].isin(valid_recipe_ids)]
        .sort_values("date")
        .drop_duplicates(subset=["user_id", "recipe_id"], keep="last")
    )
    pos_4 = dedup[dedup["rating"] >= 4]
    pos_5 = dedup[dedup["rating"] == 5]

    sensitivity_rows = [
        summarize_df(interactions, "0. Raw (all interactions)"),
        summarize_df(dedup, "1. Valid recipe + dedup (keep latest)"),
        summarize_df(pos_4, "2a. Positives (rating >= 4)"),
        summarize_df(pos_5, "2b. Positives (rating == 5)"),
        summarize_df(
            one_shot_filter(pos_4, min_user=5, min_item=0),
            "3a. rating>=4, user>=5 (one-shot, item>=1)",
        ),
        summarize_df(
            one_shot_filter(pos_4, min_user=5, min_item=2),
            "3b. rating>=4, user>=5, item>=2 (one-shot)",
        ),
        summarize_df(
            one_shot_filter(pos_4, min_user=10, min_item=0),
            "3c. rating>=4, user>=10 (one-shot, item>=1)",
        ),
        summarize_df(
            k_core_filter(pos_4, min_user=5, min_item=5),
            "4a. rating>=4, 5-core (iterative)",
        ),
        summarize_df(
            k_core_filter(pos_4, min_user=10, min_item=10),
            "4b. rating>=4, 10-core (iterative)",
        ),
        summarize_df(
            k_core_filter(pos_4, min_user=20, min_item=20),
            "4c. rating>=4, 20-core (iterative)",
        ),
    ]

    # Print formatted outputs
    print("\n=== Recipe Diagnostics ===")
    for k, v in recipe_diag.items():
        print(f"  {k}: {v}")

    print("\n=== Raw Interaction Diagnostics ===")
    print(f"  Date range: {raw_diag['date_min']} to {raw_diag['date_max']}")
    print(f"  Orphan interactions: {n_orphan_inter:,}")
    print(
        f"  Duplicate (user, recipe) pairs: {n_dup_pairs:,} "
        f"({n_dup_rows:,} total rows)"
    )
    print("  Rating distribution:")
    for r, info in rating_dist.items():
        print(f"    rating={r}: {info['count']:,} ({info['pct']}%)")

    print("\n=== Filtering Threshold Sensitivity Table ===")
    sens_df = pd.DataFrame(sensitivity_rows)
    print(sens_df.to_string(index=False))

    # Save JSON and CSV
    out_json = OUTPUT_DIR / "raw_eda_stats.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(
            {
                "recipe_diagnostics": recipe_diag,
                "raw_interaction_diagnostics": raw_diag,
                "sensitivity_table": sensitivity_rows,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )
    sens_df.to_csv(OUTPUT_DIR / "threshold_sensitivity.csv", index=False)
    print(f"\nSaved stats to {out_json} and {OUTPUT_DIR / 'threshold_sensitivity.csv'}")


if __name__ == "__main__":
    main()