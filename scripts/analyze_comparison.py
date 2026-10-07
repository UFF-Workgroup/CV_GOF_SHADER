"""Analise estatistica par a par das metricas por vista produzidas por compare_preds.py."""
import itertools
import json
import os
from argparse import ArgumentParser

import numpy as np
from scipy import stats

METRICS = ("PSNR", "SSIM", "LPIPS")


def paired_stats(a, b, rng, n_boot=10000):
    d = np.asarray(b) - np.asarray(a)
    boots = rng.choice(d, size=(n_boot, len(d)), replace=True).mean(axis=1)
    return {
        "delta_medio": float(d.mean()),
        "ic95_bootstrap": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))],
        "wilcoxon_p": float(stats.wilcoxon(d).pvalue) if np.any(d != 0) else 1.0,
        "cohen_dz": float(d.mean() / d.std(ddof=1)) if d.std(ddof=1) > 0 else 0.0,
        "vistas_b_melhor": int((d > 0).sum()),
        "n": int(len(d)),
    }


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--per_view", required=True, help="per_view.json do compare_preds.py")
    parser.add_argument("--out", required=True, help="pasta de saida")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    with open(args.per_view) as f:
        per_view = json.load(f)
    methods = list(per_view)
    keys = sorted(next(iter(per_view.values())))
    rng = np.random.default_rng(args.seed)

    result = {"metodos": {}, "pares": {}}
    for m in methods:
        result["metodos"][m] = {
            metric: {
                "media": float(np.mean([per_view[m][k][metric] for k in keys])),
                "desvio": float(np.std([per_view[m][k][metric] for k in keys], ddof=1)),
                "min": float(np.min([per_view[m][k][metric] for k in keys])),
                "max": float(np.max([per_view[m][k][metric] for k in keys])),
            } for metric in METRICS
        }
    for a, b in itertools.combinations(methods, 2):
        pair = {}
        for metric in METRICS:
            va = [per_view[a][k][metric] for k in keys]
            vb = [per_view[b][k][metric] for k in keys]
            s = paired_stats(va, vb, rng)
            if metric == "LPIPS":
                s["vistas_b_melhor"] = int(sum(y < x for x, y in zip(va, vb)))
            pair[metric] = s
        result["pares"][f"{b} vs {a}"] = pair

    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "analysis.json"), "w") as f:
        json.dump(result, f, indent=2)

    lines = ["| Par (B vs A) | Metrica | Delta medio | IC95% | Wilcoxon p | dz | Vistas B melhor |",
             "|---|---|---|---|---|---|---|"]
    for pair, res in result["pares"].items():
        for metric, s in res.items():
            lines.append(f"| {pair} | {metric} | {s['delta_medio']:+.4f} | "
                         f"[{s['ic95_bootstrap'][0]:+.4f}, {s['ic95_bootstrap'][1]:+.4f}] | "
                         f"{s['wilcoxon_p']:.1e} | {s['cohen_dz']:+.2f} | {s['vistas_b_melhor']}/{s['n']} |")
    with open(os.path.join(args.out, "analysis.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
