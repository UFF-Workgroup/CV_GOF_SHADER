"""Consolida perfil de custo e metricas das execucoes do NeRO em tabela e graficos."""
import csv
import json
import os
from argparse import ArgumentParser

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# (pasta, rotulo, chave em results.json, cor)
METHODS = [("gof", "GOF sem BRDF", "GOF", "#7B6CB5"), ("gof_brdf", "GOF + BRDF", "GOF+BRDF", "#C98A2E"),
           ("gof_brdf_ajustado", "GOF + BRDF ajustado", "GOF+BRDF ajustado", "#A1440C"),
           ("gaussianshader", "GaussianShader", "GaussianShader", "#1F7AA0")]
SCENES = ["bell", "cat", "teapot"]
INK, MUTED, GRID = "#14202E", "#4A5568", "#DDD8CB"


def load(root):
    data = {}
    for s in SCENES:
        for m, _, key, _ in METHODS:
            d = os.path.join(root, s, m)
            perfil = json.load(open(os.path.join(d, "perfil.json")))
            render = json.load(open(os.path.join(d, "render_tempo.json")))
            med = json.load(open(os.path.join(root, s, "metricas", "results.json")))["medias"]
            data[(s, m)] = {**perfil, "render_s": render["tempo_s"], **med[key]}
    return data


def style(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(GRID)
    ax.tick_params(colors=MUTED)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def grouped(ax, data, key, scale, fmt, title, unit):
    w = 0.84 / len(METHODS)
    for j, (m, n, _, c) in enumerate(METHODS):
        xs = [i + (j - (len(METHODS) - 1) / 2) * w for i in range(len(SCENES))]
        vals = [data[(s, m)][key] * scale for s in SCENES]
        bars = ax.bar(xs, vals, w - 0.04, color=c, label=n)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v, fmt.format(v), ha="center", va="bottom", fontsize=7, color=INK)
    ax.set_xticks(range(len(SCENES)), SCENES)
    ax.set_title(f"{title} ({unit})", loc="left", color=INK, fontsize=12)
    style(ax)


def fig_panels(data, keys, out, suptitle, ylim=None):
    fig, axes = plt.subplots(1, len(keys), figsize=(6 * len(keys), 4.6))
    for ax, (key, scale, fmt, title, unit) in zip(axes, keys):
        grouped(ax, data, key, scale, fmt, title, unit)
    for ax in axes:
        ax.set_ylim(top=ax.get_ylim()[1] * 1.08)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, loc="upper right", ncol=len(METHODS))
    fig.suptitle(suptitle, x=0.01, ha="left", color=INK, fontsize=13)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return axes


def main():
    p = ArgumentParser()
    p.add_argument("--root", required=True)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)
    data = load(a.root)

    fig_panels(data, [("PSNR", 1, "{:.2f}", "PSNR", "dB, maior e melhor"), ("SSIM", 1, "{:.3f}", "SSIM", "maior e melhor"),
                      ("LPIPS", 1, "{:.3f}", "LPIPS", "menor e melhor")],
               os.path.join(a.out, "qualidade.png"), "Qualidade nas 32 vistas de teste, por cena")
    axes = fig_panels(data, [("tempo_s", 1 / 3600, "{:.1f}", "Tempo de treino", "horas"), ("energia_wh", 1, "{:.0f}", "Energia do treino", "Wh"),
                             ("vram_max_mib", 1 / 1024, "{:.1f}", "VRAM maxima", "GB")],
                      os.path.join(a.out, "custo.png"), "Custo de 30 mil iteracoes na RTX 3050 (6 GB)")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, s in zip(axes, SCENES):
        pv = json.load(open(os.path.join(a.root, s, "metricas", "per_view.json")))
        g, o = pv["GaussianShader"], pv["GOF+BRDF ajustado"]
        d = sorted(g[k]["PSNR"] - o[k]["PSNR"] for k in g)
        ax.bar(range(len(d)), d, color=["#1F7AA0" if x >= 0 else "#A1440C" for x in d])
        ax.axhline(0, color=MUTED, linewidth=1)
        ax.set_title(f"{s}: GaussianShader menos GOF + BRDF ajustado (dB)", loc="left", color=INK, fontsize=11)
        ax.set_xlabel("vistas em ordem crescente", color=MUTED)
        style(ax)
    fig.suptitle("Diferenca de PSNR por vista (azul: GaussianShader melhor; laranja: GOF + BRDF ajustado melhor)", x=0.01, ha="left", color=INK, fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(a.out, "diferenca_psnr_por_vista.png"), dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)
    for ax, s in zip(axes, SCENES):
        for m, n, _, c in METHODS:
            rows = []
            with open(os.path.join(a.root, s, m, "gpu.csv")) as f:
                for r in csv.reader(f):
                    try:
                        rows.append(float(r[3]))
                    except (ValueError, IndexError):
                        pass
            step = max(1, len(rows) // 600)
            ys = [sum(rows[i:i + step]) / len(rows[i:i + step]) for i in range(0, len(rows), step)]
            ax.plot([i * step * 2 / 3600 for i in range(len(ys))], ys, color=c, linewidth=1.5, label=n)
        ax.set_title(s, loc="left", color=INK, fontsize=12)
        ax.set_xlabel("horas de treino", color=MUTED)
        style(ax)
    axes[0].set_ylabel("potencia da GPU (W)", color=MUTED)
    axes[0].legend(frameon=False)
    fig.suptitle("Potencia da GPU durante o treino", x=0.01, ha="left", color=INK, fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(a.out, "potencia.png"), dpi=150)
    plt.close(fig)

    cols = ["cena", "metodo", "PSNR", "SSIM", "LPIPS", "tempo_h", "energia_wh", "potencia_media_w", "vram_gb", "render_s", "psnr_por_wh"]
    with open(os.path.join(a.out, "resumo.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for s in SCENES:
            for m, n, _, _ in METHODS:
                d = data[(s, m)]
                w.writerow([s, n, round(d["PSNR"], 3), round(d["SSIM"], 4), round(d["LPIPS"], 4), round(d["tempo_s"] / 3600, 2), d["energia_wh"],
                            d["potencia_media_w"], round(d["vram_max_mib"] / 1024, 2), d["render_s"], round(d["PSNR"] / d["energia_wh"], 3)])
    print(open(os.path.join(a.out, "resumo.csv")).read())


if __name__ == "__main__":
    main()
