"""Fig. 2 (fig:truck): uma vista do Truck com GOF e GOF+BRDF, o mapa de diferenca de erro e o ganho de PSNR por vista.

Entradas: renders em output/EXP-20260811-02-b1 e output/EXP-20260901-02-e1, e dados/truck/per_view.json.
Saida: figuras/truck.pdf.
"""
import numpy as np
from matplotlib.colors import TwoSlopeNorm

from comum import COR, INK, LARG_PAG, MUTED, REPO, grade, img, ler, plt, salvar, virgula

# --- AJUSTES -------------------------------------------------------------------------------------
VISTA = 24                       # vista de teste mostrada na linha de cima
RUN_GOF = "EXP-20260811-02-b1"   # B1, GOF puro
RUN_BRDF = "EXP-20260901-02-e1"  # E1, GOF+BRDF
LIM_DIF = 0.1                    # limite simetrico da escala do mapa de diferenca de erro
CMAP_DIF = "RdBu_r"              # vermelho = GOF+BRDF pior, azul = melhor
TAMANHO = (LARG_PAG, 3.2)        # polegadas (largura, altura)
# -------------------------------------------------------------------------------------------------


def main():
    pv = ler("truck", "per_view.json")
    base = REPO / "output"
    gt = img(base / RUN_BRDF / "test" / "ours_30000" / "gt_2" / f"{VISTA:05d}.png")
    b1 = img(base / RUN_GOF / "test" / "ours_30000" / "test_preds_2" / f"{VISTA:05d}.png")
    e1 = img(base / RUN_BRDF / "test" / "ours_30000" / "test_preds_2" / f"{VISTA:05d}.png")
    dif = np.abs(e1 - gt).mean(-1) - np.abs(b1 - gt).mean(-1)

    fig = plt.figure(figsize=TAMANHO)
    gs = fig.add_gridspec(2, 4, height_ratios=[1, 0.9], hspace=0.22, wspace=0.04)
    kb, ke = "GOF-B1(nosso)", "GOF+BRDF-E1(nosso)"
    titulos = ["Referência", f"GOF ({pv[kb][f'{VISTA:05d}']['PSNR']:.2f} dB)",
               f"GOF+BRDF ({pv[ke][f'{VISTA:05d}']['PSNR']:.2f} dB)", "Δ erro (GOF+BRDF − GOF)"]
    for c, (im, t) in enumerate(zip([gt, b1, e1, dif], titulos)):
        ax = fig.add_subplot(gs[0, c])
        if c < 3:
            ax.imshow(im)
        else:
            h = ax.imshow(np.clip(im, -LIM_DIF, LIM_DIF), cmap=CMAP_DIF, norm=TwoSlopeNorm(0, -LIM_DIF, LIM_DIF))
            cb = fig.colorbar(h, ax=ax, fraction=0.035, pad=0.02)
            cb.ax.tick_params(labelsize=6)
        ax.set_title(virgula(t), fontsize=7.5, color=INK)
        ax.set_xticks([])
        ax.set_yticks([])

    ax = fig.add_subplot(gs[1, :])
    vistas = sorted(pv[kb])
    d = np.array(sorted(pv[ke][v]["PSNR"] - pv[kb][v]["PSNR"] for v in vistas))
    ax.bar(range(len(d)), d, color=[COR["gof_brdf"] if x >= 0 else COR["gof"] for x in d], width=0.8)
    ax.axhline(0, color=MUTED, linewidth=0.7)
    ax.axhline(d.mean(), color=INK, linewidth=0.7, linestyle=(0, (3, 2)))
    ax.text(1, d.mean() + 0.05, virgula(f"média {d.mean():+.2f} dB"), ha="left", va="bottom", fontsize=7)
    ax.set_xlabel("vistas de teste em ordem crescente de Δ")
    ax.set_ylabel("Δ PSNR (dB)")
    ax.set_xticks([])
    grade(ax)
    salvar(fig, "truck.pdf")


if __name__ == "__main__":
    main()
