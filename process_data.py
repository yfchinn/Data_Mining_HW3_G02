"""
§4 資料處理：切分、負樣本、ID 編號、食材特徵（目前設定LightGCN baseline / LightFM 主方法）

【執行順序】
  1. 前一位組員已執行（預設參數）：python clean_data.py
     -> data/processed/interactions_clean.csv、recipes_clean.csv
  2. 再執行本程式：python process_data.py
     -> 輸出到 data/processed_sec4/，不會動到 data/processed/

本程式的處理（所有模型共用同一份結果）：
  1. 依時間切分（leave-one-out）：每位使用者最新一筆 -> test、
     次新一筆 -> valid、其餘 -> train
  2. 負樣本：每筆 valid / test 抽 100 道該使用者從未互動過的食譜
     （任何評分都算互動，所以會另外讀原始互動檔；seed 固定，所有模型共用）
  3. 使用者與食譜 ID 重新編號為 0 ~ N-1

各方法的處理：
  LightGCN（baseline）：train 中的「使用者-食譜」互動
  LightFM（主方法）：  食譜特徵矩陣 = 食譜本身 ID + 正規化後的食材
                        （小寫、去標點；只保留出現在 >= 5 道食譜的食材）
"""
import argparse, ast, json, re
from collections import Counter
from pathlib import Path
import numpy as np
import pandas as pd
import scipy.sparse as sp


def normalize_ingredient(s: str) -> str:
    s = str(s).lower().strip()
    s = re.sub(r"[^a-z0-9\s\-&']", " ", s)   # 去除標點（保留 - & '）
    s = re.sub(r"\s+", " ", s).strip()
    return s


def parse_ingredients(val):
    try:
        lst = ast.literal_eval(val) if isinstance(val, str) else val
    except (ValueError, SyntaxError):
        return []
    if not isinstance(lst, list):
        return []
    out = [normalize_ingredient(x) for x in lst]
    return sorted(set(x for x in out if x))   # 同一道食譜內去重


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clean-dir", type=Path, default=Path("data/processed"),
                    help="clean_data.py 的輸出資料夾（interactions_clean.csv, recipes_clean.csv）")
    ap.add_argument("--raw-interactions", type=Path, default=Path("data/RAW_interactions.csv"),
                    help="只用來排除負樣本中使用者互動過的食譜")
    ap.add_argument("--out-dir", type=Path, default=Path("data/processed_sec4"))
    ap.add_argument("--min-ingr-freq", type=int, default=5)
    ap.add_argument("--n-neg", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    pos = pd.read_csv(args.clean_dir / "interactions_clean.csv",
                      usecols=["user_id", "recipe_id", "date", "rating"])
    pos["date"] = pd.to_datetime(pos["date"])
    recipes = pd.read_csv(args.clean_dir / "recipes_clean.csv", usecols=["id", "name", "ingredients"])

    # 檢查 clean_data.py 的過濾結果
    min_u = int(pos["user_id"].value_counts().min())
    min_i = int(pos["recipe_id"].value_counts().min())
    print(f"清洗後資料：{pos.user_id.nunique():,} 使用者、{pos.recipe_id.nunique():,} 食譜、"
          f"{len(pos):,} 筆；每人最少 {min_u} 筆、每道食譜最少 {min_i} 筆")

    raw = pd.read_csv(args.raw_interactions, usecols=["user_id", "recipe_id"])
    all_seen = raw.groupby("user_id")["recipe_id"].apply(set).to_dict()

    # 1. leave-one-out 依時間切分（同一天以 recipe_id 排序，確保可重現）
    pos = pos.sort_values(["user_id", "date", "recipe_id"]).copy()
    pos["rank_from_end"] = pos.groupby("user_id").cumcount(ascending=False)
    test = pos[pos["rank_from_end"] == 0].copy()
    valid = pos[pos["rank_from_end"] == 1].copy()
    train = pos[pos["rank_from_end"] >= 2].copy()

    # 3. ID 重新編號
    users = np.sort(pos["user_id"].unique())
    items = np.sort(pos["recipe_id"].unique())
    u2i = {int(u): k for k, u in enumerate(users)}
    r2i = {int(r): k for k, r in enumerate(items)}
    for df in (train, valid, test):
        df["u"] = df["user_id"].map(u2i)
        df["i"] = df["recipe_id"].map(r2i)

    train_items = set(train["recipe_id"])
    cold_valid = int((~valid["recipe_id"].isin(train_items)).sum())
    cold_test = int((~test["recipe_id"].isin(train_items)).sum())

    # 2. 負樣本
    def sample_negs(df):
        rows = []
        for uid, rid in zip(df["user_id"].values, df["recipe_id"].values):
            seen = all_seen[uid]
            negs = []
            while len(negs) < args.n_neg:
                cand = rng.choice(items, size=args.n_neg * 2)
                negs.extend(int(c) for c in cand if c not in seen and c not in negs)
            rows.append({"u": u2i[int(uid)], "pos_i": r2i[int(rid)],
                         "neg_i": [r2i[c] for c in negs[:args.n_neg]]})
        return rows
    neg_valid, neg_test = sample_negs(valid), sample_negs(test)

    # 食材詞彙（只看食譜內容，不用互動資料，不會洩漏）
    recipes = recipes[recipes["id"].isin(r2i)].copy()
    recipes["ingr_list"] = recipes["ingredients"].apply(parse_ingredients)
    raw_unique = len(Counter(str(x).strip().lower()
                             for v in recipes["ingredients"] for x in ast.literal_eval(v)))
    freq = Counter(x for lst in recipes["ingr_list"] for x in lst)
    vocab = sorted(k for k, v in freq.items() if v >= args.min_ingr_freq)
    ing2i = {k: j for j, k in enumerate(vocab)}
    recipes["ingr_kept"] = recipes["ingr_list"].apply(lambda l: [x for x in l if x in ing2i])

    # LightFM 食譜特徵：[食譜 ID | 食材]
    n_items, n_ing = len(items), len(vocab)
    rows, cols = [], []
    for rid, lst in zip(recipes["id"].values, recipes["ingr_kept"].values):
        k = r2i[int(rid)]
        rows.append(k); cols.append(k)
        for x in lst:
            rows.append(k); cols.append(n_items + ing2i[x])
    item_feat = sp.csr_matrix((np.ones(len(rows), dtype=np.float32), (rows, cols)),
                              shape=(n_items, n_items + n_ing))

    # 輸出
    o = args.out_dir
    cols_out = ["u", "i", "user_id", "recipe_id", "date", "rating"]
    for name, df in (("train", train), ("valid", valid), ("test", test)):
        df[cols_out].assign(date=df["date"].dt.strftime("%Y-%m-%d")).to_csv(o / f"{name}.csv", index=False)
    pd.DataFrame(neg_valid).to_json(o / "valid_negatives.jsonl", orient="records", lines=True)
    pd.DataFrame(neg_test).to_json(o / "test_negatives.jsonl", orient="records", lines=True)
    sp.save_npz(o / "lightfm_item_features.npz", item_feat)
    json.dump({"user_id_to_index": {str(k): v for k, v in u2i.items()},
               "recipe_id_to_index": {str(k): v for k, v in r2i.items()},
               "ingredient_vocab": vocab}, open(o / "id_maps.json", "w"))

    stats = {
        "config": {"clean_dir": str(args.clean_dir), "out_dir": str(args.out_dir),
                   "min_ingr_freq": args.min_ingr_freq, "n_neg": args.n_neg, "seed": args.seed},
        "input": {"users": len(users), "recipes": n_items, "interactions": len(pos),
                  "min_per_user": min_u, "min_per_recipe": min_i},
        "split": {"train": len(train), "valid": len(valid), "test": len(test),
                  "min_train_per_user": int(train.groupby("user_id").size().min()),
                  "valid_items_unseen_in_train": cold_valid,
                  "test_items_unseen_in_train": cold_test},
        "ingredients": {"raw_unique_lowercased": raw_unique,
                        "unique_after_normalization": len(freq),
                        f"vocab_size_freq_ge_{args.min_ingr_freq}": n_ing,
                        "recipes_with_no_kept_ingredient": int((recipes["ingr_kept"].str.len() == 0).sum()),
                        "mean_kept_ingredients_per_recipe":
                            round(float(recipes["ingr_kept"].str.len().mean()), 2)},
        "lightfm_item_features_shape": list(item_feat.shape),
    }
    json.dump(stats, open(o / "process_stats.json", "w"), indent=2)
    print(json.dumps(stats, indent=2))

    # 處理後範例（挑訓練筆數最少的使用者，方便閱讀）
    cnt = train.groupby("user_id").size()
    uid = int(cnt[cnt == cnt.min()].index[0])
    ex_r = int(train[train.user_id == uid].recipe_id.iloc[0])
    row = recipes[recipes.id == ex_r].iloc[0]
    k = r2i[ex_r]
    nz = item_feat[k].indices
    ex = {
        "user_id": uid, "u": u2i[uid],
        "train": train[train.user_id == uid][["recipe_id", "i", "date", "rating"]]
                 .assign(date=lambda d: d.date.dt.strftime("%Y-%m-%d")).to_dict("records"),
        "valid": valid[valid.user_id == uid][["recipe_id", "i"]].to_dict("records"),
        "test": test[test.user_id == uid][["recipe_id", "i"]].to_dict("records"),
        "example_recipe": {"recipe_id": ex_r, "i": k, "name": row["name"],
                           "raw_ingredients": ast.literal_eval(row["ingredients"]),
                           "lightfm_feature_columns": [int(c) for c in nz],
                           "lightfm_features": ["recipe_" + str(ex_r) if c < n_items
                                                else "ingr:" + vocab[c - n_items] for c in nz]},
    }
    json.dump(ex, open(o / "processed_example.json", "w"), indent=2, default=int)
    print(json.dumps(ex, indent=2, default=int))


if __name__ == "__main__":
    main()
