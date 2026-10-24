"""Imagens dos mapas de ambiente aprendidos (insumo da Fig. 7), tiradas do lighting.pth de cada treino GOF+BRDF.

Entradas: output/nero/<cena>/<gof_brdf|gof_brdf_ajustado>/point_cloud/iteration_30000/lighting.pth (precisa de torch).
Saidas: figuras/envmaps/<cena>_<metodo>.png e dados/nero/envmaps.json (resolucao e escala usada).
"""
import json

import numpy as np
import torch
from PIL import Image

from comum import CENAS, DADOS, FIGURAS, REPO

# --- AJUSTES -------------------------------------------------------------------------------------
PERCENTIL = 99.5   # radiancia neste percentil vira branco; o resto satura
GAMA = 2.2         # so para visualizacao, o mapa e linear
METODOS_BRDF = ["gof_brdf", "gof_brdf_ajustado"]
# -------------------------------------------------------------------------------------------------


def png(pth, out):
    base = torch.load(pth, map_location="cpu")["state_dict"]["base"].clamp(min=0).numpy().transpose(1, 2, 0)
    escala = np.percentile(base, PERCENTIL)
    im = np.clip(base / escala, 0, 1) ** (1 / GAMA)
    Image.fromarray((im * 255).astype(np.uint8)).save(out)
    print("imagem", out)
    return {"altura": base.shape[0], "largura": base.shape[1], "p99_5": float(escala)}


def main():
    (FIGURAS / "envmaps").mkdir(exist_ok=True)
    meta = {}
    for s in CENAS:
        for m in METODOS_BRDF:
            pth = REPO / "output" / "nero" / s / m / "point_cloud" / "iteration_30000" / "lighting.pth"
            meta[f"{s}/{m}"] = png(pth, FIGURAS / "envmaps" / f"{s}_{m}.png")
    with open(DADOS / "nero" / "envmaps.json", "w") as f:
        json.dump(meta, f, indent=1)


if __name__ == "__main__":
    main()
