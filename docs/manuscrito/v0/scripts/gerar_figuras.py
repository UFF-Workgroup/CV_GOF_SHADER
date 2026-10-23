"""Gera as figuras do manuscrito a partir de dados/ e dos renders em output/ (uso: gerar_figuras.py <repo> <v0>)."""
import glob
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import locale

try:
    locale.setlocale(locale.LC_NUMERIC, "pt_BR.UTF-8")  # virgula decimal nos eixos, como no texto
except locale.Error:
    pass
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TwoSlopeNorm
from PIL import Image

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Liberation Serif", "Nimbus Roman"], "mathtext.fontset": "stix",
    "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8, "legend.fontsize": 7.5,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6, "pdf.fonttype": 42, "axes.formatter.use_locale": True,
})
INK, MUTED, GRID = "#14202E", "#4A5568", "#DDD8CB"
CENAS = ["bell", "cat", "teapot"]
# (pasta, rotulo, chave nas metricas, cor)
METODOS = [("gof", "GOF", "GOF", "#7B6CB5"), ("gof_brdf", "GOF+BRDF (config. Truck)", "GOF+BRDF", "#C98A2E"),
           ("gof_brdf_ajustado", "GOF+BRDF ajustado", "GOF+BRDF ajustado", "#A1440C"),
           ("gaussianshader", "GaussianShader", "GaussianShader", "#1F7AA0")]
# Tabela 5 de Jiang et al. (arXiv:2311.17977v1), PSNR/SSIM/LPIPS publicados para o GaussianShader.
GS_PUBLICADO = {"bell": (28.08, 0.920, 0.097), "cat": (31.81, 0.961, 0.056), "teapot": (23.57, 0.899, 0.090)}
VISTA_NERO, VISTA_TRUCK = 5, 24
LARG_COL, LARG_PAG = 3.5, 7.2


def grade(ax, eixo="y"):
    ax.grid(True, axis=eixo, color=GRID, linewidth=0.5)
    ax.set_axisbelow(True)


def salvar(fig, out):
    fig.savefig(out, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print("figura", out)


def img(p):
    return np.asarray(Image.open(p).convert("RGB"), dtype=np.float32) / 255.0


def arquivo(pasta, idx):
    return sorted(glob.glob(os.path.join(pasta, f"{idx:05d}*")))[0]


def pasta_pred(repo, s, m):
    sub = "renders" if m == "gaussianshader" else "test_preds_-1"
    return os.path.join(repo, "output", "nero", s, m, "test", "ours_30000", sub)


def caixa(gt, margem=24):
    obj = np.where((gt < 0.97).any(-1))
    y0, y1, x0, x1 = obj[0].min(), obj[0].max(), obj[1].min(), obj[1].max()
    lado = max(y1 - y0, x1 - x0) // 2 + margem
    cy, cx = (y0 + y1) // 2, (x0 + x1) // 2
    h, w = gt.shape[:2]
    cy, cx = np.clip(cy, lado, h - lado), np.clip(cx, lado, w - lado)
    return cy - lado, cy + lado, cx - lado, cx + lado


def fig_nero_qualitativo(repo, v0, pv):
    fig, axes = plt.subplots(6, 5, figsize=(LARG_PAG, 8.9), gridspec_kw={"hspace": 0.04, "wspace": 0.03})
    vmax = 0.25
    for i, s in enumerate(CENAS):
        gt = img(arquivo(os.path.join(repo, "output", "nero", s, "gof_brdf", "test", "ours_30000", "gt_-1"), VISTA_NERO))
        y0, y1, x0, x1 = caixa(gt)
        linha_r, linha_e = axes[2 * i], axes[2 * i + 1]
        linha_r[0].imshow(gt[y0:y1, x0:x1])
        linha_e[0].axis("off")
        cax = linha_e[0].inset_axes([0.42, 0.1, 0.12, 0.8])
        cb = fig.colorbar(plt.cm.ScalarMappable(cmap="inferno", norm=plt.Normalize(0, vmax)), cax=cax)
        cb.ax.tick_params(labelsize=6)
        cb.set_label("erro abs. médio", fontsize=6.5)
        for j, (m, nome, key, _) in enumerate(METODOS, start=1):
            pred = img(arquivo(pasta_pred(repo, s, m), VISTA_NERO))
            linha_r[j].imshow(pred[y0:y1, x0:x1])
            psnr = pv[s][key][f"{VISTA_NERO:05d}"]["PSNR"]
            linha_r[j].text(0.97, 0.03, f"{psnr:.2f} dB".replace(".", ","), ha="right", va="bottom", fontsize=7, color="white",
                            transform=linha_r[j].transAxes, bbox=dict(boxstyle="square,pad=0.2", fc=INK, ec="none"))
            erro = np.abs(pred - gt).mean(-1)[y0:y1, x0:x1]
            linha_e[j].imshow(erro, cmap="inferno", vmin=0, vmax=vmax)
        linha_r[0].set_ylabel(s, fontsize=8.5, color=INK)
    for ax in axes.ravel():
        ax.set_xticks([])
        ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)
    for j, t in enumerate(["Referência"] + [n for _, n, _, _ in METODOS]):
        axes[0, j].set_title(t.replace(" (config. Truck)", "\n(config. Truck)"), fontsize=7.5, color=INK)
    salvar(fig, os.path.join(v0, "figuras", "nero_qualitativo.pdf"))


def barras(ax, res, metrica, titulo, publicado=None):
    w = 0.84 / len(METODOS)
    for j, (_, nome, key, cor) in enumerate(METODOS):
        xs = [i + (j - (len(METODOS) - 1) / 2) * w for i in range(len(CENAS))]
        vals = [res[s]["medias"][key][metrica] for s in CENAS]
        # Pontos em vez de barras: o eixo nao comeca em zero, o que distorceria o comprimento das barras.
        ax.scatter(xs, vals, s=30, color=cor, edgecolor="white", linewidth=0.6, zorder=4, label=nome)
    if publicado is not None:
        xs = [i + 1.5 * w for i in range(len(CENAS))]
        ax.scatter(xs, [GS_PUBLICADO[s][publicado] for s in CENAS], marker="D", s=22, facecolor="none",
                   edgecolor=INK, linewidth=0.8, zorder=5, label="GaussianShader (publicado)")
    ax.set_xticks(range(len(CENAS)), CENAS)
    for i in range(len(CENAS) - 1):
        ax.axvline(i + 0.5, color=GRID, linewidth=0.6)
    ax.set_title(titulo, loc="left", color=INK)
    grade(ax)


def fig_nero_metricas(v0, res):
    fig, axes = plt.subplots(1, 3, figsize=(LARG_PAG, 2.35))
    barras(axes[0], res, "PSNR", "PSNR (dB) ↑", 0)
    barras(axes[1], res, "SSIM", "SSIM ↑", 1)
    barras(axes[2], res, "LPIPS", "LPIPS ↓", 2)
    axes[0].set_ylim(15, 36)
    axes[1].set_ylim(0.84, 0.99)
    axes[2].set_ylim(0, 0.135)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=5, frameon=False, bbox_to_anchor=(0.5, 1.08), handlelength=1.2, columnspacing=1.0)
    fig.tight_layout()
    salvar(fig, os.path.join(v0, "figuras", "nero_metricas.pdf"))


def fig_efeitos(v0, an_nero, an_truck):
    linhas = [("Truck", "GOF+BRDF−GOF", an_truck["pares"]["GOF+BRDF-E1(nosso) vs GOF-B1(nosso)"], "#C98A2E")]
    for s in CENAS:
        a = an_nero[s]["pares"]
        linhas += [(s, "GOF+BRDF−GOF", a["GOF+BRDF vs GOF"], "#C98A2E"),
                   (s, "ajustado−GOF+BRDF", a["GOF+BRDF ajustado vs GOF+BRDF"], "#A1440C"),
                   (s, "ajustado−GaussianShader", {k: {"delta_medio": -v["delta_medio"], "ic95_bootstrap": [-v["ic95_bootstrap"][1], -v["ic95_bootstrap"][0]]}
                                                   for k, v in a["GaussianShader vs GOF+BRDF ajustado"].items()}, "#1F7AA0")]
    fig, axes = plt.subplots(1, 3, figsize=(LARG_PAG, 2.6), sharey=True)
    ys = np.arange(len(linhas))[::-1].astype(float)
    for c, (ax, met, tit) in enumerate(zip(axes, ["PSNR", "SSIM", "LPIPS"], ["Δ PSNR (dB) ↑", "Δ SSIM ↑", "Δ LPIPS ↓"])):
        for y, (cena, rot, st, cor) in zip(ys, linhas):
            m, (lo, hi) = st[met]["delta_medio"], st[met]["ic95_bootstrap"]
            ax.plot([lo, hi], [y, y], color=cor, linewidth=1.6, solid_capstyle="round")
            ax.plot(m, y, "o", color=cor, markersize=3.2)
        ax.axvline(0, color=MUTED, linewidth=0.7)
        ax.set_title(tit, loc="left", color=INK)
        grade(ax, "x")
        if c == 0:
            ax.set_yticks(ys, [f"{cena}: {rot}" for cena, rot, _, _ in linhas])
    for y in (ys[0] - 0.5, ys[3] - 0.5, ys[6] - 0.5):
        for ax in axes:
            ax.axhline(y, color=GRID, linewidth=0.6)
    fig.tight_layout()
    salvar(fig, os.path.join(v0, "figuras", "efeitos_pareados.pdf"))


def fig_custo_qualidade(v0, res, perfis):
    fig, axes = plt.subplots(1, 3, figsize=(LARG_PAG, 2.2))
    for ax, s in zip(axes, CENAS):
        for _, nome, key, cor in METODOS:
            m = [k for k, n, kk, _ in METODOS if kk == key][0]
            e, q = perfis[f"{s}/{m}"]["energia_wh"], res[s]["medias"][key]["PSNR"]
            ax.scatter(e, q, s=26, color=cor, edgecolor="white", linewidth=0.8, zorder=3, label=nome)
        ax.set_title(s, loc="left", color=INK)
        ax.set_xlabel("energia da GPU no treino (Wh)")
        ax.set_xlim(0, 480)
        grade(ax, "both")
    axes[0].set_ylabel("PSNR (dB)")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 1.1), handlelength=1.0)
    fig.tight_layout()
    salvar(fig, os.path.join(v0, "figuras", "custo_qualidade.pdf"))


def fig_mecanismo(v0, mat):
    fig = plt.figure(figsize=(LARG_PAG, 4.6))
    gs = fig.add_gridspec(3, 6, height_ratios=[1, 1, 1.25], hspace=0.62, wspace=0.9)
    pares = [("gof_brdf", "config. Truck", "#C98A2E"), ("gof_brdf_ajustado", "ajustado", "#A1440C")]
    for r, (attr, nome) in enumerate([("hist_tom", "tom especular s (média RGB)"), ("hist_rugosidade", "rugosidade ρ")]):
        for c, s in enumerate(CENAS):
            ax = fig.add_subplot(gs[r, 2 * c:2 * c + 2])
            for m, rot, cor in pares:
                d = mat[f"{s}/{m}"]
                ax.plot(d["bins"], np.concatenate([[0], np.cumsum(d[attr])]), color=cor, linewidth=1.2, label=rot)
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.set_title(f"{s}: {nome}", loc="left", color=INK)
            if c == 0:
                ax.set_ylabel("fração acumulada\n(pond. pela opacidade)")
            grade(ax, "both")
            if r == 0 and c == 2:
                ax.legend(frameon=False, loc="lower right")
    for c, (m, rot) in enumerate([("gof_brdf", "bell, config. Truck (128×256)"), ("gof_brdf_ajustado", "bell, ajustado (512×1024)")]):
        ax = fig.add_subplot(gs[2, 3 * c:3 * c + 3])
        ax.imshow(Image.open(os.path.join(v0, "figuras", "envmaps", f"bell_{m}.png")), interpolation="nearest")
        ax.set_title(f"Mapa de ambiente aprendido, {rot}", loc="left", color=INK)
        ax.set_xticks([])
        ax.set_yticks([])
    salvar(fig, os.path.join(v0, "figuras", "mecanismo.pdf"))


def fig_truck(repo, v0, pv_truck):
    base = os.path.join(repo, "output")
    gt = img(os.path.join(base, "EXP-20260901-02-e1", "test", "ours_30000", "gt_2", f"{VISTA_TRUCK:05d}.png"))
    b1 = img(os.path.join(base, "EXP-20260811-02-b1", "test", "ours_30000", "test_preds_2", f"{VISTA_TRUCK:05d}.png"))
    e1 = img(os.path.join(base, "EXP-20260901-02-e1", "test", "ours_30000", "test_preds_2", f"{VISTA_TRUCK:05d}.png"))
    dif = np.abs(e1 - gt).mean(-1) - np.abs(b1 - gt).mean(-1)
    fig = plt.figure(figsize=(LARG_PAG, 3.2))
    gs = fig.add_gridspec(2, 4, height_ratios=[1, 0.9], hspace=0.22, wspace=0.04)
    kb, ke = "GOF-B1(nosso)", "GOF+BRDF-E1(nosso)"
    titulos = ["Referência", f"GOF ({pv_truck[kb][f'{VISTA_TRUCK:05d}']['PSNR']:.2f} dB)",
               f"GOF+BRDF ({pv_truck[ke][f'{VISTA_TRUCK:05d}']['PSNR']:.2f} dB)", "Δ erro (GOF+BRDF − GOF)"]
    for c, (im, t) in enumerate(zip([gt, b1, e1, dif], titulos)):
        ax = fig.add_subplot(gs[0, c])
        if c < 3:
            ax.imshow(im)
        else:
            lim = 0.1
            h = ax.imshow(np.clip(im, -lim, lim), cmap="RdBu_r", norm=TwoSlopeNorm(0, -lim, lim))
            cb = fig.colorbar(h, ax=ax, fraction=0.035, pad=0.02)
            cb.ax.tick_params(labelsize=6)
        ax.set_title(t.replace(".", ","), fontsize=7.5, color=INK)
        ax.set_xticks([])
        ax.set_yticks([])
    ax = fig.add_subplot(gs[1, :])
    vistas = sorted(pv_truck[kb])
    d = np.array(sorted(pv_truck[ke][v]["PSNR"] - pv_truck[kb][v]["PSNR"] for v in vistas))
    ax.bar(range(len(d)), d, color=["#C98A2E" if x >= 0 else "#7B6CB5" for x in d], width=0.8)
    ax.axhline(0, color=MUTED, linewidth=0.7)
    ax.axhline(d.mean(), color=INK, linewidth=0.7, linestyle=(0, (3, 2)))
    ax.text(1, d.mean() + 0.05, f"média {d.mean():+.2f} dB".replace(".", ","), ha="left", va="bottom", fontsize=7)
    ax.set_xlabel("vistas de teste em ordem crescente de Δ")
    ax.set_ylabel("Δ PSNR (dB)")
    ax.set_xticks([])
    grade(ax)
    salvar(fig, os.path.join(v0, "figuras", "truck.pdf"))


def fig_potencia(v0, pot):
    fig, axes = plt.subplots(1, 3, figsize=(LARG_PAG, 2.0), sharey=True)
    for ax, s in zip(axes, CENAS):
        for m, nome, _, cor in METODOS:
            p = pot[f"{s}/{m}"]
            ax.plot(p["horas"], p["watts"], color=cor, linewidth=0.9, label=nome)
        ax.set_title(s, loc="left", color=INK)
        ax.set_xlabel("horas de treino")
        grade(ax)
    axes[0].set_ylabel("potência da GPU (W)")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 1.12))
    fig.tight_layout()
    salvar(fig, os.path.join(v0, "figuras", "potencia.pdf"))


def fig_por_vista(v0, pv):
    fig, axes = plt.subplots(1, 3, figsize=(LARG_PAG, 1.9), sharey=True)
    for ax, s in zip(axes, CENAS):
        a, g = pv[s]["GOF+BRDF ajustado"], pv[s]["GaussianShader"]
        d = np.array(sorted(a[k]["PSNR"] - g[k]["PSNR"] for k in a))
        ax.bar(range(len(d)), d, color=["#A1440C" if x >= 0 else "#1F7AA0" for x in d], width=0.8)
        ax.axhline(0, color=MUTED, linewidth=0.7)
        ax.set_title(s, loc="left", color=INK)
        ax.set_xticks([])
        ax.set_xlabel("vistas em ordem crescente")
        grade(ax)
    axes[0].set_ylabel("Δ PSNR (dB)")
    fig.tight_layout()
    salvar(fig, os.path.join(v0, "figuras", "por_vista_ajustado_gs.pdf"))


def main(repo, v0):
    dn = os.path.join(v0, "dados", "nero")
    res = {s: json.load(open(os.path.join(dn, f"{s}_results.json"))) for s in CENAS}
    pv = {s: json.load(open(os.path.join(dn, f"{s}_per_view.json"))) for s in CENAS}
    an = {s: json.load(open(os.path.join(dn, f"{s}_analysis.json"))) for s in CENAS}
    an_truck = json.load(open(os.path.join(v0, "dados", "truck", "analysis.json")))
    pv_truck = json.load(open(os.path.join(v0, "dados", "truck", "per_view.json")))
    perfis = json.load(open(os.path.join(dn, "perfis.json")))
    fig_nero_qualitativo(repo, v0, pv)
    fig_nero_metricas(v0, res)
    fig_efeitos(v0, an, an_truck)
    fig_custo_qualidade(v0, res, perfis)
    fig_mecanismo(v0, json.load(open(os.path.join(dn, "material.json"))))
    fig_truck(repo, v0, pv_truck)
    fig_potencia(v0, json.load(open(os.path.join(dn, "potencia.json"))))
    fig_por_vista(v0, pv)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
