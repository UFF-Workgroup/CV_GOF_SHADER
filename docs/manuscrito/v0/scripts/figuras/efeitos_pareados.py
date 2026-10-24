"""Fig. 5 (fig:efeitos): diferencas pareadas por vista com IC 95% bootstrap, para PSNR, SSIM e LPIPS (grafico de floresta).

Entradas: dados/truck/analysis.json e dados/nero/<cena>_analysis.json.
Saida: figuras/efeitos_pareados.pdf.
"""
import numpy as np

from comum import CENAS, COR, GRID, INK, LARG_PAG, MUTED, grade, ler, por_cena, plt, salvar

# --- AJUSTES -------------------------------------------------------------------------------------
TITULOS = {"PSNR": "Δ PSNR (dB) ↑", "SSIM": "Δ SSIM ↑", "LPIPS": "Δ LPIPS ↓"}
TAMANHO = (LARG_PAG, 2.6)
# -------------------------------------------------------------------------------------------------


def inverter(par):
    """Troca o sentido da diferenca (B - A vira A - B), invertendo tambem o intervalo."""
    return {k: {"delta_medio": -v["delta_medio"], "ic95_bootstrap": [-v["ic95_bootstrap"][1], -v["ic95_bootstrap"][0]]}
            for k, v in par.items()}


def main():
    an_truck, an = ler("truck", "analysis.json"), por_cena("analysis")
    # (cena, contraste, estatisticas, cor) de cima para baixo
    linhas = [("Truck", "GOF+BRDF−GOF", an_truck["pares"]["GOF+BRDF-E1(nosso) vs GOF-B1(nosso)"], COR["gof_brdf"])]
    for s in CENAS:
        a = an[s]["pares"]
        linhas += [(s, "GOF+BRDF−GOF", a["GOF+BRDF vs GOF"], COR["gof_brdf"]),
                   (s, "ajustado−GOF+BRDF", a["GOF+BRDF ajustado vs GOF+BRDF"], COR["gof_brdf_ajustado"]),
                   (s, "ajustado−GaussianShader", inverter(a["GaussianShader vs GOF+BRDF ajustado"]), COR["gaussianshader"])]

    fig, axes = plt.subplots(1, 3, figsize=TAMANHO, sharey=True)
    ys = np.arange(len(linhas))[::-1].astype(float)
    for c, (ax, met) in enumerate(zip(axes, ["PSNR", "SSIM", "LPIPS"])):
        for y, (_, _, st, cor) in zip(ys, linhas):
            m, (lo, hi) = st[met]["delta_medio"], st[met]["ic95_bootstrap"]
            ax.plot([lo, hi], [y, y], color=cor, linewidth=1.6, solid_capstyle="round")
            ax.plot(m, y, "o", color=cor, markersize=3.2)
        ax.axvline(0, color=MUTED, linewidth=0.7)
        ax.set_title(TITULOS[met], loc="left", color=INK)
        grade(ax, "x")
        if c == 0:
            ax.set_yticks(ys, [f"{cena}: {rot}" for cena, rot, _, _ in linhas])
    # Separadores entre Truck e cada cena do NeRO.
    for y in (ys[0] - 0.5, ys[3] - 0.5, ys[6] - 0.5):
        for ax in axes:
            ax.axhline(y, color=GRID, linewidth=0.6)
    fig.tight_layout()
    salvar(fig, "efeitos_pareados.pdf")


if __name__ == "__main__":
    main()
