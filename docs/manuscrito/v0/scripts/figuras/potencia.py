"""Fig. S1 (fig:potencia): potencia da GPU ao longo do treino, por cena e metodo.

Entradas: dados/nero/potencia.json (amostras do nvidia-smi a cada 2 s, ja reamostradas por extrair_dados.py).
Saida: figuras/potencia.pdf.
"""
from comum import CENAS, INK, LARG_PAG, METODOS, grade, ler, plt, salvar

# --- AJUSTES -------------------------------------------------------------------------------------
LARGURA_LINHA = 0.9
TAMANHO = (LARG_PAG, 2.0)
# -------------------------------------------------------------------------------------------------


def main():
    pot = ler("nero", "potencia.json")
    fig, axes = plt.subplots(1, 3, figsize=TAMANHO, sharey=True)
    for ax, s in zip(axes, CENAS):
        for m, nome, _, cor in METODOS:
            p = pot[f"{s}/{m}"]
            ax.plot(p["horas"], p["watts"], color=cor, linewidth=LARGURA_LINHA, label=nome)
        ax.set_title(s, loc="left", color=INK)
        ax.set_xlabel("horas de treino")
        grade(ax)
    axes[0].set_ylabel("potência da GPU (W)")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 1.12))
    fig.tight_layout()
    salvar(fig, "potencia.pdf")


if __name__ == "__main__":
    main()
