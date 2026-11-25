"""
LightFM 訓練 + validation 評估（HW3 G02，§5 Method / §6 Experimental Setup）

只在 validation set 上評估；test set 等所有設定定案後再另外跑一次。

用法（在 repo 根目錄執行）：
  # 主方法：食譜 ID + 食材
  python train_lightfm.py --data-dir data/processed_sec4 --variant full
  # Baseline B1（Same model, without our processing）：只有食譜 ID
  python train_lightfm.py --data-dir data/processed_sec4 --variant id_only

輸入（process_data.py 的輸出）：
  train.csv, valid_negatives.jsonl, id_maps.json, lightfm_item_features.npz
輸出（--out-dir，預設 outputs/lightfm）：
  <tag>.json            設定、最佳 epoch、validation 指標、冷門程度分層結果、耗時
  <tag>_valid_ranks.csv 每位使用者正解的排名（給 bootstrap / 案例分析用）
"""
import argparse
import copy
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

K = 10
# 依「正解食譜在 train 中的正向互動數」分層，用來看冷門食譜上的差異
POP_BUCKETS = [(0, 0, "0 (unseen)"), (1, 2, "1-2"), (3, 5, "3-5"),
               (6, 20, "6-20"), (21, 10**9, "21+")]


# ---------------------------------------------------------------- data
def load_data(d: Path):
    maps = json.load(open(d / "id_maps.json", encoding="utf-8"))
    n_users = len(maps["user_id_to_index"])
    n_items = len(maps["recipe_id_to_index"])
    train = pd.read_csv(d / "train.csv", usecols=["u", "i"])
    inter = sp.coo_matrix(
        (np.ones(len(train), dtype=np.float32), (train["u"].values, train["i"].values)),
        shape=(n_users, n_items))
    item_feat = sp.load_npz(d / "lightfm_item_features.npz").tocsr()
    assert item_feat.shape[0] == n_items, "特徵矩陣列數和食譜數不一致"
    return train, inter, item_feat, n_users, n_items


def build_item_features(item_feat, variant: str, normalize: bool):
    """full = [ID | 食材]；id_only = None（LightFM 會自動用 identity，等於純 MF）"""
    if variant == "id_only":
        return None
    X = item_feat.astype(np.float32)
    if normalize:  # 每列總和 = 1（LightFM 官方 Dataset 的預設做法）
        row_sum = np.asarray(X.sum(axis=1)).ravel()
        row_sum[row_sum == 0] = 1.0
        X = sp.diags(1.0 / row_sum) @ X
    return sp.csr_matrix(X, dtype=np.float32)


def load_candidates(path: Path):
    users, cands = [], []
    with open(path, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            users.append(r["u"])
            cands.append([r["pos_i"]] + list(r["neg_i"]))  # 第 0 欄 = 正解
    return np.asarray(users, dtype=np.int32), np.asarray(cands, dtype=np.int32)


# ---------------------------------------------------------------- eval
def score_candidates(model, users, cands, item_features, threads):
    n, c = cands.shape
    s = model.predict(np.repeat(users, c), cands.ravel(),
                      item_features=item_features, num_threads=threads)
    return np.asarray(s).reshape(n, c)


def positive_ranks(scores):
    """正解的名次（1 = 第一名）。同分時算輸（保守），避免給全部同分的模型高分。"""
    return 1 + (scores[:, 1:] >= scores[:, :1]).sum(axis=1)


def summarize(ranks, k=K):
    ranks = np.asarray(ranks, dtype=float)
    hit = ranks <= k
    return {
        f"HR@{k}": round(float(hit.mean()), 4),
        f"NDCG@{k}": round(float(np.where(hit, 1.0 / np.log2(ranks + 1), 0.0).mean()), 4),
        "MRR": round(float((1.0 / ranks).mean()), 4),
        "n": int(len(ranks)),
    }


def topk_coverage(scores, cands, n_items, k=K):
    top = np.argsort(-scores, axis=1)[:, :k]
    rec = np.take_along_axis(cands, top, axis=1)
    return round(len(np.unique(rec)) / n_items, 4)


def slice_by_popularity(ranks, pos_items, train_pop):
    pop = train_pop[pos_items]
    out = {}
    for lo, hi, name in POP_BUCKETS:
        m = (pop >= lo) & (pop <= hi)
        if m.any():
            out[name] = summarize(ranks[m])
    return out


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, default=Path("data/processed_sec4"))
    ap.add_argument("--variant", choices=["full", "id_only"], default="full")
    ap.add_argument("--tag", default=None, help="輸出檔名；預設 <data-dir 名稱>_<variant>")
    ap.add_argument("--out-dir", type=Path, default=Path("outputs/lightfm"))
    ap.add_argument("--loss", choices=["warp", "bpr", "logistic", "warp-kos"], default="warp")
    ap.add_argument("--no-components", type=int, default=64)
    ap.add_argument("--learning-rate", type=float, default=0.05)
    ap.add_argument("--item-alpha", type=float, default=0.0)
    ap.add_argument("--user-alpha", type=float, default=0.0)
    ap.add_argument("--normalize", action="store_true", help="特徵每列正規化成總和 1")
    ap.add_argument("--max-epochs", type=int, default=50)
    ap.add_argument("--eval-every", type=int, default=5)
    ap.add_argument("--patience", type=int, default=2, help="連續幾次評估沒進步就停止")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    from lightfm import LightFM  # 放在這裡，沒裝 lightfm 時 --help 仍可用

    tag = args.tag or f"{args.data_dir.name}_{args.variant}"
    args.out_dir.mkdir(parents=True, exist_ok=True)

    train, inter, item_feat_raw, n_users, n_items = load_data(args.data_dir)
    X = build_item_features(item_feat_raw, args.variant, args.normalize)
    v_users, v_cands = load_candidates(args.data_dir / "valid_negatives.jsonl")
    train_pop = np.bincount(train["i"].values, minlength=n_items)
    feat_desc = "identity (ID only)" if X is None else f"{X.shape[1]:,}"
    print(f"[{tag}] users={n_users:,} items={n_items:,} train={len(train):,} "
          f"valid={len(v_users):,} features={feat_desc}")

    model = LightFM(no_components=args.no_components, loss=args.loss,
                    learning_rate=args.learning_rate, item_alpha=args.item_alpha,
                    user_alpha=args.user_alpha, random_state=args.seed)

    best = {"ndcg": -1.0, "epoch": 0, "model": None}
    history, bad, train_sec = [], 0, 0.0
    for epoch in range(1, args.max_epochs + 1):
        t0 = time.time()
        model.fit_partial(inter, item_features=X, epochs=1, num_threads=args.threads)
        train_sec += time.time() - t0
        if epoch % args.eval_every and epoch != args.max_epochs:
            continue
        ranks = positive_ranks(score_candidates(model, v_users, v_cands, X, args.threads))
        m = summarize(ranks)
        history.append({"epoch": epoch, **m})
        print(f"  epoch {epoch:3d}  HR@10={m['HR@10']:.4f}  NDCG@10={m['NDCG@10']:.4f}  "
              f"MRR={m['MRR']:.4f}  ({train_sec:.0f}s)")
        if m[f"NDCG@{K}"] > best["ndcg"]:
            best = {"ndcg": m[f"NDCG@{K}"], "epoch": epoch, "model": copy.deepcopy(model)}
            bad = 0
        else:
            bad += 1
            if bad >= args.patience:
                print("  early stop")
                break

    # 用最佳 epoch 的模型重新評估並記錄細節
    model = best["model"]
    t0 = time.time()
    scores = score_candidates(model, v_users, v_cands, X, args.threads)
    infer_sec = time.time() - t0
    ranks = positive_ranks(scores)

    result = {
        "tag": tag,
        "config": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
        "data": {"users": n_users, "items": n_items, "train_interactions": int(len(train)),
                 "valid_users": int(len(v_users)), "candidates_per_user": int(v_cands.shape[1]),
                 "n_item_features": None if X is None else int(X.shape[1])},
        "best_epoch": best["epoch"],
        "valid": {**summarize(ranks), f"Coverage@{K}": topk_coverage(scores, v_cands, n_items)},
        "valid_by_item_popularity": slice_by_popularity(ranks, v_cands[:, 0], train_pop),
        "cost": {"train_seconds_total": round(train_sec, 1),
                 "inference_ms_per_user": round(infer_sec / len(v_users) * 1000, 3)},
        "history": history,
    }
    with open(args.out_dir / f"{tag}.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    pd.DataFrame({"u": v_users, "pos_i": v_cands[:, 0], "rank": ranks,
                  "pos_train_pop": train_pop[v_cands[:, 0]]}
                 ).to_csv(args.out_dir / f"{tag}_valid_ranks.csv", index=False)

    print(f"[{tag}] best epoch {best['epoch']}: {result['valid']}")
    for name, m in result["valid_by_item_popularity"].items():
        print(f"    pos item train-pop {name:>10}: n={m['n']:6d}  HR@10={m['HR@10']:.4f}  "
              f"NDCG@10={m['NDCG@10']:.4f}")
    print(f"  saved {args.out_dir / (tag + '.json')}")


if __name__ == "__main__":
    main()
