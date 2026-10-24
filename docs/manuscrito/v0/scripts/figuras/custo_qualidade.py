"""Fig. 6 (fig:custo): PSNR contra energia da GPU no treino, por cena e metodo.

Entradas: dados/nero/<cena>_results.json e dados/nero/perfis.json.
Saida: figuras/custo_qualidade.pdf.
"""
from comum import CENAS, INK, LARG_PAG, METODOS, grade, ler, por_cena, plt, salvar

# --- AJUSTES -------------------------------------------------------------------------------------
XLIM = (0, 480)   # eixo de energia (Wh), igual nas tres cenas para comparar
TAMANHO = (LARG_PAG, 2.2)
# -------------------------------------------------------------------------------------------------


def main():
    res, perfis = por_cena("results"), ler("nero", "perfis.json")
    fig, axes = plt.subplots(1, 3, figsize=TAMANHO)
    for ax, s in zip(axes, CENAS):
        for m, nome, key, cor in METODOS:
            e, q = perfis[f"{s}/{m}"]["energia_wh"], res[s]["medias"][key]["PSNR"]
            ax.scatter(e, q, s=26, color=cor, edgecolor="white", linewidth=0.8, zorder=3, label=nome)
        ax.set_title(s, loc="left", color=INK)
        ax.set_xlabel("energia da GPU no treino (Wh)")
        ax.set_xlim(*XLIM)
        grade(ax, "both")
    axes[0].set_ylabel("PSNR (dB)")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 1.1), handlelength=1.0)
    fig.tight_layout()
    salvar(fig, "custo_qualidade.pdf")


if __name__ == "__main__":
    main()
