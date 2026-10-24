"""Estilo, paleta, caminhos e leitores de dados compartilhados por todos os scripts de figura."""
import glob
import json
import locale
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

try:
    locale.setlocale(locale.LC_NUMERIC, "pt_BR.UTF-8")  # virgula decimal nos eixos, como no texto
except locale.Error:
    pass

# Caminhos: v0/ do manuscrito e raiz do repositorio (GOF_REPO sobrescreve, para rodar com output/ em outro lugar).
V0 = Path(__file__).resolve().parents[2]
REPO = Path(os.environ.get("GOF_REPO", V0.parents[2]))
DADOS, FIGURAS = V0 / "dados", V0 / "figuras"

# Fonte TrueType embutida (pdf.fonttype 42) e tamanhos pensados para a largura final no PDF.
plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Liberation Serif", "Nimbus Roman"], "mathtext.fontset": "stix",
    "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8, "legend.fontsize": 7.5,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6, "pdf.fonttype": 42, "axes.formatter.use_locale": True,
})

# Tinta do texto e da grade; nunca usadas como cor de serie.
INK, MUTED, GRID = "#14202E", "#4A5568", "#DDD8CB"
# Uma cor fixa por metodo em todas as figuras (paleta validada para daltonismo).
COR = {"gof": "#7B6CB5", "gof_brdf": "#C98A2E", "gof_brdf_ajustado": "#A1440C", "gaussianshader": "#1F7AA0"}
CENAS = ["bell", "cat", "teapot"]
# (pasta em output/nero, rotulo nas figuras, chave nos JSON de metricas, cor)
METODOS = [("gof", "GOF", "GOF", COR["gof"]), ("gof_brdf", "GOF+BRDF (config. Truck)", "GOF+BRDF", COR["gof_brdf"]),
           ("gof_brdf_ajustado", "GOF+BRDF ajustado", "GOF+BRDF ajustado", COR["gof_brdf_ajustado"]),
           ("gaussianshader", "GaussianShader", "GaussianShader", COR["gaussianshader"])]
# Tabela 5 de Jiang et al. (arXiv:2311.17977v1), PSNR/SSIM/LPIPS publicados para o GaussianShader.
GS_PUBLICADO = {"bell": (28.08, 0.920, 0.097), "cat": (31.81, 0.961, 0.056), "teapot": (23.57, 0.899, 0.090)}
# Largura de desenho em polegadas; o \textwidth do artigo e 6,5 pol., entao o LaTeX ajusta a escala ao incluir.
LARG_PAG = 7.2


def ler(*partes):
    with open(DADOS.joinpath(*partes)) as f:
        return json.load(f)


def por_cena(sufixo):
    return {s: ler("nero", f"{s}_{sufixo}.json") for s in CENAS}


def grade(ax, eixo="y"):
    ax.grid(True, axis=eixo, color=GRID, linewidth=0.5)
    ax.set_axisbelow(True)


def virgula(texto):
    return texto.replace(".", ",")


def salvar(fig, nome):
    out = FIGURAS / nome
    # Sem data no PDF: rodar de novo sem mudar nada gera o mesmo arquivo, e o git so acusa mudanca real.
    fig.savefig(out, bbox_inches="tight", pad_inches=0.02, metadata={"CreationDate": None})
    plt.close(fig)
    print("figura", out)


def img(p):
    return np.asarray(Image.open(p).convert("RGB"), dtype=np.float32) / 255.0


def arquivo(pasta, idx):
    achados = sorted(glob.glob(os.path.join(pasta, f"{idx:05d}*")))
    if not achados:
        raise FileNotFoundError(f"vista {idx} nao encontrada em {pasta}; esta figura precisa das saidas dos treinos em output/")
    return achados[0]
