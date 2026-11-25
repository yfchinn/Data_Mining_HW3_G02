"""
一鍵比較食譜端門檻 --min-item-inter = 5 vs 2 的 LightFM 結果（只看 validation）

對每個門檻 k：
  1. clean_data.py   --min-item-inter k  -> data/processed_item{k}/
  2. process_data.py                     -> data/processed_sec4_item{k}/
  3. train_lightfm.py --variant full     （主方法：ID + 食材）
     train_lightfm.py --variant id_only  （B1：同模型、不加食材）
最後輸出比較表 outputs/lightfm/compare_item_threshold.csv

不會覆蓋原本的 data/processed/ 和 data/processed_sec4/。

用法（repo 根目錄）：
  python run_compare_item_threshold.py
  python run_compare_item_threshold.py --thresholds 5 2 --max-epochs 30 --threads 8
  python run_compare_item_threshold.py --skip-prep   # 資料已產生過，只重跑訓練
其他參數（--no-components、--learning-rate、--loss、--normalize ...）會原封不動傳給 train_lightfm.py
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd


def run(cmd):
    print("\n$ " + " ".join(str(c) for c in cmd), flush=True)
    subprocess.run([str(c) for c in cmd], check=True)


def check_raw_files():
    for p in (Path("data/RAW_recipes.csv"), Path("data/RAW_interactions.csv")):
        if not p.exists() or p.stat().st_size < 1024:
            sys.exit(f"{p} 不存在或只是 Git LFS 指標檔。\n"
                     "請先執行 `git lfs install && git lfs pull`，"
                     "或從 Kaggle 下載 Food.com 資料集放到 data/。")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--thresholds", type=int, nargs="+", default=[5, 2])
    ap.add_argument("--variants", nargs="+", default=["full", "id_only"])
    ap.add_argument("--skip-prep", action="store_true")
    ap.add_argument("--out-dir", type=Path, default=Path("outputs/lightfm"))
    args, train_args = ap.parse_known_args()

    py = sys.executable
    if not args.skip_prep:
        check_raw_files()

    rows = []
    for k in args.thresholds:
        clean_dir = Path(f"data/processed_item{k}")
        sec4_dir = Path(f"data/processed_sec4_item{k}")
        if not args.skip_prep:
            run([py, "clean_data.py", "--min-item-inter", k, "--output-dir", clean_dir])
            run([py, "process_data.py", "--clean-dir", clean_dir, "--out-dir", sec4_dir])
        pstats = json.load(open(sec4_dir / "process_stats.json", encoding="utf-8"))

        for v in args.variants:
            tag = f"item{k}_{v}"
            run([py, "train_lightfm.py", "--data-dir", sec4_dir, "--variant", v,
                 "--tag", tag, "--out-dir", args.out_dir, *train_args])
            r = json.load(open(args.out_dir / f"{tag}.json", encoding="utf-8"))
            row = {"min_item_inter": k, "variant": v,
                   "users": r["data"]["users"], "items": r["data"]["items"],
                   "valid_pos_unseen_in_train": pstats["split"]["valid_items_unseen_in_train"],
                   "best_epoch": r["best_epoch"], **r["valid"],
                   "train_sec": r["cost"]["train_seconds_total"]}
            for name, m in r["valid_by_item_popularity"].items():
                row[f"NDCG@10[pop {name}]"] = m["NDCG@10"]
                row[f"n[pop {name}]"] = m["n"]
            rows.append(row)

    df = pd.DataFrame(rows)
    out = args.out_dir / "compare_item_threshold.csv"
    df.to_csv(out, index=False)
    pd.set_option("display.width", 200)
    print("\n=== Validation 比較 ===")
    print(df[["min_item_inter", "variant", "users", "items", "valid_pos_unseen_in_train",
              "best_epoch", "HR@10", "NDCG@10", "MRR", "Coverage@10"]].to_string(index=False))
    print("\n=== 依正解食譜的 train 互動數分層（NDCG@10）===")
    print(df[["min_item_inter", "variant"] +
             [c for c in df.columns if c.startswith("NDCG@10[")]].to_string(index=False))
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
