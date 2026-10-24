"""Fig. S2 (fig:por_vista): diferenca de PSNR por vista, GOF+BRDF ajustado menos GaussianShader, em ordem crescente.

Entradas: dados/nero/<cena>_per_view.json.
Saida: figuras/por_vista_ajustado_gs.pdf.
"""
import numpy as np

from comum import CENAS, COR, INK, LARG_PAG, MUTED, grade, por_cena, plt, salvar

# --- AJUSTES -------------------------------------------------------------------------------------
A, B = "GOF+BRDF ajustado", "GaussianShader"   # diferenca A - B; chaves dos JSON de metricas
METRICA = "PSNR"
TAMANHO = (LARG_PAG, 1.9)
# -------------------------------------------------------------------------------------------------


def main():
    pv = por_cena("per_view")
    fig, axes = plt.subplots(1, 3, figsize=TAMANHO, sharey=True)
    for ax, s in zip(axes, CENAS):
        a, b = pv[s][A], pv[s][B]
        d = np.array(sorted(a[k][METRICA] - b[k][METRICA] for k in a))
        ax.bar(range(len(d)), d, color=[COR["gof_brdf_ajustado"] if x >= 0 else COR["gaussianshader"] for x in d], width=0.8)
        ax.axhline(0, color=MUTED, linewidth=0.7)
        ax.set_title(s, loc="left", color=INK)
        ax.set_xticks([])
        ax.set_xlabel("vistas em ordem crescente")
        grade(ax)
    axes[0].set_ylabel(f"Δ {METRICA}" + (" (dB)" if METRICA == "PSNR" else ""))
    fig.tight_layout()
    salvar(fig, "por_vista_ajustado_gs.pdf")


if __name__ == "__main__":
    main()
