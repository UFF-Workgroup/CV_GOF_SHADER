"""Copia e deriva, a partir das saidas dos treinos, todos os dados que o manuscrito usa (dados/ e figuras/envmaps)."""
import csv
import json
import os
import shutil
import sys

import numpy as np
import torch
from PIL import Image
from plyfile import PlyData

CENAS = ["bell", "cat", "teapot"]
METODOS = ["gof", "gof_brdf", "gof_brdf_ajustado", "gaussianshader"]


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def material(ply):
    v = PlyData.read(ply)["vertex"]
    op = sigmoid(np.asarray(v["opacity"]))
    rough = sigmoid(np.asarray(v["roughness_0"]))
    tint = sigmoid(np.stack([np.asarray(v[f"specular_tint_{i}"]) for i in range(3)], 1)).mean(1)
    w = op / op.sum()

    def wq(x, q):
        o = np.argsort(x)
        c = np.cumsum(w[o])
        return float(x[o][np.searchsorted(c, q)])

    bins = np.linspace(0, 1, 41)
    return {
        "n_gaussianas": int(len(op)),
        "rugosidade_mediana_pond": wq(rough, 0.5),
        "rugosidade_q25_pond": wq(rough, 0.25),
        "rugosidade_q75_pond": wq(rough, 0.75),
        "tom_mediano_pond": wq(tint, 0.5),
        "frac_tom_maior_0_1_pond": float(w[tint > 0.1].sum()),
        "hist_rugosidade": np.histogram(rough, bins=bins, weights=w)[0].tolist(),
        "hist_tom": np.histogram(tint, bins=bins, weights=w)[0].tolist(),
        "bins": bins.tolist(),
    }


def envmap_png(pth, out):
    base = torch.load(pth, map_location="cpu")["state_dict"]["base"].clamp(min=0).numpy().transpose(1, 2, 0)
    escala = np.percentile(base, 99.5)
    img = np.clip(base / escala, 0, 1) ** (1 / 2.2)
    Image.fromarray((img * 255).astype(np.uint8)).save(out)
    return {"altura": base.shape[0], "largura": base.shape[1], "p99_5": float(escala)}


def potencia(gpu_csv, max_pontos=400):
    vals = []
    with open(gpu_csv) as f:
        for r in csv.reader(f):
            try:
                vals.append(float(r[3]))
            except (ValueError, IndexError):
                pass
    passo = max(1, len(vals) // max_pontos)
    ys = [float(np.mean(vals[i:i + passo])) for i in range(0, len(vals), passo)]
    return {"horas": [i * passo * 2 / 3600 for i in range(len(ys))], "watts": ys}


def main(repo, dst):
    nero, truck = os.path.join(repo, "output", "nero"), os.path.join(repo, "docs", "experiments", "COMPARACAO-truck")
    os.makedirs(os.path.join(dst, "dados", "nero"), exist_ok=True)
    os.makedirs(os.path.join(dst, "dados", "truck"), exist_ok=True)
    os.makedirs(os.path.join(dst, "figuras", "envmaps"), exist_ok=True)
    for f in ("per_view.json", "results.json", "analysis.json"):
        shutil.copy(os.path.join(truck, f), os.path.join(dst, "dados", "truck", f))
    shutil.copy(os.path.join(nero, "relatorio", "resumo.csv"), os.path.join(dst, "dados", "nero", "resumo.csv"))

    perfis, mats, envs, pots = {}, {}, {}, {}
    for s in CENAS:
        for f in ("per_view.json", "results.json", "analysis.json"):
            shutil.copy(os.path.join(nero, s, "metricas", f), os.path.join(dst, "dados", "nero", f"{s}_{f}"))
        for m in METODOS:
            d = os.path.join(nero, s, m)
            p = json.load(open(os.path.join(d, "perfil.json")))
            p["render"] = json.load(open(os.path.join(d, "render_tempo.json")))
            p["n_gaussianas"] = PlyData.read(os.path.join(d, "point_cloud", "iteration_30000", "point_cloud.ply"))["vertex"].count
            perfis[f"{s}/{m}"] = p
            pots[f"{s}/{m}"] = potencia(os.path.join(d, "gpu.csv"))
            if m.startswith("gof_brdf"):
                mats[f"{s}/{m}"] = material(os.path.join(d, "point_cloud", "iteration_30000", "point_cloud.ply"))
                envs[f"{s}/{m}"] = envmap_png(os.path.join(d, "point_cloud", "iteration_30000", "lighting.pth"),
                                              os.path.join(dst, "figuras", "envmaps", f"{s}_{m}.png"))
            print(s, m, "ok", flush=True)
    for nome, obj in (("perfis", perfis), ("material", mats), ("envmaps", envs), ("potencia", pots)):
        json.dump(obj, open(os.path.join(dst, "dados", "nero", nome + ".json"), "w"), indent=1)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
