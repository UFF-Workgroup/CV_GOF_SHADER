"""Fig. 4 (fig:nero_metricas): PSNR, SSIM e LPIPS medios dos quatro metodos no NeRO, com o GaussianShader publicado.

Entradas: dados/nero/<cena>_results.json e GS_PUBLICADO (comum.py).
Saida: figuras/nero_metricas.pdf.
"""
from comum import CENAS, GRID, GS_PUBLICADO, INK, LARG_PAG, METODOS, grade, por_cena, plt, salvar

# --- AJUSTES -------------------------------------------------------------------------------------
# Limites dos eixos y; nao comecam em zero, por isso a figura usa pontos e nao barras.
YLIM = {"PSNR": (15, 36), "SSIM": (0.84, 0.99), "LPIPS": (0, 0.135)}
TITULOS = {"PSNR": "PSNR (dB) ↑", "SSIM": "SSIM ↑", "LPIPS": "LPIPS ↓"}
MOSTRAR_PUBLICADO = True   # losangos vazados com os numeros da Tabela 5 do GaussianShader
TAMANHO = (LARG_PAG, 2.35)
# -------------------------------------------------------------------------------------------------


def pontos(ax, res, metrica, publicado):
    w = 0.84 / len(METODOS)
    for j, (_, nome, key, cor) in enumerate(METODOS):
        xs = [i + (j - (len(METODOS) - 1) / 2) * w for i in range(len(CENAS))]
        vals = [res[s]["medias"][key][metrica] for s in CENAS]
        ax.scatter(xs, vals, s=30, color=cor, edgecolor="white", linewidth=0.6, zorder=4, label=nome)
    if MOSTRAR_PUBLICADO:
        xs = [i + 1.5 * w for i in range(len(CENAS))]
        ax.scatter(xs, [GS_PUBLICADO[s][publicado] for s in CENAS], marker="D", s=22, facecolor="none",
                   edgecolor=INK, linewidth=0.8, zorder=5, label="GaussianShader (publicado)")
    ax.set_xticks(range(len(CENAS)), CENAS)
    for i in range(len(CENAS) - 1):
        ax.axvline(i + 0.5, color=GRID, linewidth=0.6)
    ax.set_title(TITULOS[metrica], loc="left", color=INK)
    ax.set_ylim(*YLIM[metrica])
    grade(ax)


def main():
    res = por_cena("results")
    fig, axes = plt.subplots(1, 3, figsize=TAMANHO)
    for i, (ax, metrica) in enumerate(zip(axes, ["PSNR", "SSIM", "LPIPS"])):
        pontos(ax, res, metrica, i)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=5, frameon=False, bbox_to_anchor=(0.5, 1.08), handlelength=1.2, columnspacing=1.0)
    fig.tight_layout()
    salvar(fig, "nero_metricas.pdf")


if __name__ == "__main__":
    main()
