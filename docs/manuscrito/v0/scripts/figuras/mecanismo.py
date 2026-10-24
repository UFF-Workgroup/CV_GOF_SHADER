"""Fig. 7 (fig:mecanismo): distribuicoes acumuladas do tom especular e da rugosidade, e os mapas de ambiente da bell.

Entradas: dados/nero/material.json e figuras/envmaps/bell_*.png (gerados por envmaps.py).
Saida: figuras/mecanismo.pdf.
"""
import numpy as np
from PIL import Image

from comum import CENAS, COR, FIGURAS, INK, LARG_PAG, grade, ler, plt, salvar

# --- AJUSTES -------------------------------------------------------------------------------------
CENA_ENVMAP = "bell"   # cena cujos mapas de ambiente aparecem na linha de baixo
ROTULO_ENVMAP = {"gof_brdf": "config. Truck (128×256)", "gof_brdf_ajustado": "ajustado (512×1024)"}
TAMANHO = (LARG_PAG, 4.6)
# -------------------------------------------------------------------------------------------------

PARES = [("gof_brdf", "config. Truck", COR["gof_brdf"]), ("gof_brdf_ajustado", "ajustado", COR["gof_brdf_ajustado"])]
ATRIBUTOS = [("hist_tom", "tom especular s (média RGB)"), ("hist_rugosidade", "rugosidade ρ")]


def main():
    mat = ler("nero", "material.json")
    fig = plt.figure(figsize=TAMANHO)
    gs = fig.add_gridspec(3, 6, height_ratios=[1, 1, 1.25], hspace=0.62, wspace=0.9)
    for r, (attr, nome) in enumerate(ATRIBUTOS):
        for c, s in enumerate(CENAS):
            ax = fig.add_subplot(gs[r, 2 * c:2 * c + 2])
            for m, rot, cor in PARES:
                d = mat[f"{s}/{m}"]
                # Histograma ponderado pela opacidade virando fracao acumulada.
                ax.plot(d["bins"], np.concatenate([[0], np.cumsum(d[attr])]), color=cor, linewidth=1.2, label=rot)
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.set_title(f"{s}: {nome}", loc="left", color=INK)
            if c == 0:
                ax.set_ylabel("fração acumulada\n(pond. pela opacidade)")
            grade(ax, "both")
            if r == 0 and c == 2:
                ax.legend(frameon=False, loc="lower right")
    for c, (m, _, _) in enumerate(PARES):
        ax = fig.add_subplot(gs[2, 3 * c:3 * c + 3])
        ax.imshow(Image.open(FIGURAS / "envmaps" / f"{CENA_ENVMAP}_{m}.png"), interpolation="nearest")
        ax.set_title(f"Mapa de ambiente aprendido, {CENA_ENVMAP}, {ROTULO_ENVMAP[m]}", loc="left", color=INK)
        ax.set_xticks([])
        ax.set_yticks([])
    salvar(fig, "mecanismo.pdf")


if __name__ == "__main__":
    main()
