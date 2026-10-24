"""Fig. 3 (fig:nero_qualitativo): referencia e renders dos quatro metodos em bell, cat e teapot, com mapas de erro absoluto.

Entradas: renders de teste em output/nero/<cena>/<metodo>/test/ours_30000 e dados/nero/<cena>_per_view.json.
Saida: figuras/nero_qualitativo.pdf.
"""
import os

import numpy as np

from comum import CENAS, INK, LARG_PAG, METODOS, REPO, arquivo, img, por_cena, plt, salvar, virgula

# --- AJUSTES -------------------------------------------------------------------------------------
VISTA = 5              # vista de teste mostrada (a mesma nas tres cenas)
VMAX_ERRO = 0.25       # topo da escala do mapa de erro absoluto medio (0 a 1)
CMAP_ERRO = "inferno"
MARGEM = 24            # pixels em volta do objeto no recorte
LIMIAR_FUNDO = 0.97    # pixel com todos os canais acima disto conta como fundo branco
TAMANHO = (LARG_PAG, 8.9)
# -------------------------------------------------------------------------------------------------


def pasta_pred(s, m):
    sub = "renders" if m == "gaussianshader" else "test_preds_-1"
    return os.path.join(REPO, "output", "nero", s, m, "test", "ours_30000", sub)


def caixa(gt):
    """Recorte quadrado em volta do objeto, achado pelos pixels que nao sao fundo na referencia."""
    obj = np.where((gt < LIMIAR_FUNDO).any(-1))
    y0, y1, x0, x1 = obj[0].min(), obj[0].max(), obj[1].min(), obj[1].max()
    lado = max(y1 - y0, x1 - x0) // 2 + MARGEM
    cy, cx = (y0 + y1) // 2, (x0 + x1) // 2
    h, w = gt.shape[:2]
    cy, cx = np.clip(cy, lado, h - lado), np.clip(cx, lado, w - lado)
    return cy - lado, cy + lado, cx - lado, cx + lado


def main():
    pv = por_cena("per_view")
    fig, axes = plt.subplots(6, 5, figsize=TAMANHO, gridspec_kw={"hspace": 0.04, "wspace": 0.03})
    for i, s in enumerate(CENAS):
        gt = img(arquivo(os.path.join(REPO, "output", "nero", s, "gof_brdf", "test", "ours_30000", "gt_-1"), VISTA))
        y0, y1, x0, x1 = caixa(gt)
        linha_r, linha_e = axes[2 * i], axes[2 * i + 1]
        linha_r[0].imshow(gt[y0:y1, x0:x1])
        linha_e[0].axis("off")
        cax = linha_e[0].inset_axes([0.42, 0.1, 0.12, 0.8])
        cb = fig.colorbar(plt.cm.ScalarMappable(cmap=CMAP_ERRO, norm=plt.Normalize(0, VMAX_ERRO)), cax=cax)
        cb.ax.tick_params(labelsize=6)
        cb.set_label("erro abs. médio", fontsize=6.5)
        for j, (m, nome, key, _) in enumerate(METODOS, start=1):
            pred = img(arquivo(pasta_pred(s, m), VISTA))
            linha_r[j].imshow(pred[y0:y1, x0:x1])
            psnr = pv[s][key][f"{VISTA:05d}"]["PSNR"]
            linha_r[j].text(0.97, 0.03, virgula(f"{psnr:.2f} dB"), ha="right", va="bottom", fontsize=7, color="white",
                            transform=linha_r[j].transAxes, bbox=dict(boxstyle="square,pad=0.2", fc=INK, ec="none"))
            erro = np.abs(pred - gt).mean(-1)[y0:y1, x0:x1]
            linha_e[j].imshow(erro, cmap=CMAP_ERRO, vmin=0, vmax=VMAX_ERRO)
        linha_r[0].set_ylabel(s, fontsize=8.5, color=INK)
    for ax in axes.ravel():
        ax.set_xticks([])
        ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)
    for j, t in enumerate(["Referência"] + [n for _, n, _, _ in METODOS]):
        axes[0, j].set_title(t.replace(" (config. Truck)", "\n(config. Truck)"), fontsize=7.5, color=INK)
    salvar(fig, "nero_qualitativo.pdf")


if __name__ == "__main__":
    main()
