"""Gera as tabelas LaTeX (tabelas/*.tex) a partir de dados/, para que nenhum numero do texto seja digitado a mao."""
import json
import os
import sys

CENAS = ["bell", "cat", "teapot"]
METODOS = [("gof", "GOF", "GOF"), ("gof_brdf", "GOF+BRDF (config. Truck)", "GOF+BRDF"),
           ("gof_brdf_ajustado", "GOF+BRDF ajustado", "GOF+BRDF ajustado"), ("gaussianshader", "GaussianShader", "GaussianShader")]
GS_PUBLICADO = {"bell": (28.08, 0.920, 0.097), "cat": (31.81, 0.961, 0.056), "teapot": (23.57, 0.899, 0.090)}


def f(x, n):
    return f"{x:.{n}f}".replace(".", "{,}").replace("-", "$-$")


def fs(x, n):
    return ("+" if x >= 0 else "") + f(x, n)


def p(x):
    if x < 1e-3:
        m, e = f"{x:.1e}".split("e")
        return f"${m.replace('.', '{,}')}\\times10^{{{int(e)}}}$"
    return f(x, 3)


def ic(st, n):
    lo, hi = st["ic95_bootstrap"]
    return f"[{fs(lo, n)}; {fs(hi, n)}]"


def tab_nero(res):
    linhas = []
    for s in CENAS:
        med = res[s]["medias"]
        melhor = {m: (max if m != "LPIPS" else min)(med[k][m] for _, _, k in METODOS) for m in ("PSNR", "SSIM", "LPIPS")}
        for i, (_, nome, k) in enumerate(METODOS):
            cel = []
            for m, n in (("PSNR", 2), ("SSIM", 4), ("LPIPS", 4)):
                v = f(med[k][m], n)
                cel.append(f"\\textbf{{{v}}}" if abs(med[k][m] - melhor[m]) < 1e-12 else v)
            linhas.append(f"{s if i == 0 else ''} & {nome} & " + " & ".join(cel) + " \\\\")
        g = GS_PUBLICADO[s]
        linhas.append(f" & \\textit{{GaussianShader, publicado}} & \\textit{{{f(g[0], 2)}}} & \\textit{{{f(g[1], 3)}}} & \\textit{{{f(g[2], 3)}}} \\\\")
        if s != CENAS[-1]:
            linhas.append("\\midrule")
    return "\n".join(linhas) + "\n"


def tab_pares(an):
    contrastes = [("GOF+BRDF vs GOF", "GOF+BRDF $-$ GOF", 1), ("GOF+BRDF ajustado vs GOF+BRDF", "ajustado $-$ GOF+BRDF", 1),
                  ("GaussianShader vs GOF+BRDF ajustado", "ajustado $-$ GaussianShader", -1)]
    linhas = []
    for s in CENAS:
        for i, (chave, rot, sinal) in enumerate(contrastes):
            st = an[s]["pares"][chave]
            cel = []
            for m, n in (("PSNR", 2), ("SSIM", 4), ("LPIPS", 4)):
                x = st[m]
                lo, hi = x["ic95_bootstrap"]
                d, lo, hi = sinal * x["delta_medio"], *sorted((sinal * lo, sinal * hi))
                cel.append(f"{fs(d, n)} [{fs(lo, n)}; {fs(hi, n)}]")
            v = st["PSNR"]["vistas_b_melhor"] if sinal == 1 else st["PSNR"]["n"] - st["PSNR"]["vistas_b_melhor"]
            linhas.append(f"{s if i == 0 else ''} & {rot} & " + " & ".join(cel) + f" & {p(st['PSNR']['wilcoxon_p'])} & {v}/32 \\\\")
        if s != CENAS[-1]:
            linhas.append("\\midrule")
    return "\n".join(linhas) + "\n"


def tab_custo(perfis):
    linhas = []
    for s in CENAS:
        for i, (m, nome, _) in enumerate(METODOS):
            d = perfis[f"{s}/{m}"]
            linhas.append(f"{s if i == 0 else ''} & {nome} & {f(d['tempo_s'] / 3600, 2)} & {f(d['energia_wh'], 0)} & {f(d['potencia_media_w'], 1)} & "
                          f"{f(d['vram_max_mib'] / 1024, 2)} & {f(d['n_gaussianas'] / 1000, 0)} & {f(d['render']['tempo_s'], 0)} \\\\")
        if s != CENAS[-1]:
            linhas.append("\\midrule")
    return "\n".join(linhas) + "\n"


def tab_truck(res, an):
    med = res["medias"] if "medias" in res else res
    b, e = med["GOF-B1(nosso)"], med["GOF+BRDF-E1(nosso)"]
    st = an["pares"]["GOF+BRDF-E1(nosso) vs GOF-B1(nosso)"]
    linhas = [f"GOF & {f(b['PSNR'], 3)} & {f(b['SSIM'], 4)} & {f(b['LPIPS'], 4)} \\\\",
              f"GOF+BRDF & {f(e['PSNR'], 3)} & {f(e['SSIM'], 4)} & {f(e['LPIPS'], 4)} \\\\", "\\midrule",
              "$\\Delta$ médio & " + " & ".join(fs(st[m]["delta_medio"], n) for m, n in (("PSNR", 3), ("SSIM", 4), ("LPIPS", 4))) + " \\\\",
              "IC 95\\% (\\emph{bootstrap}) & " + " & ".join(ic(st[m], n) for m, n in (("PSNR", 2), ("SSIM", 4), ("LPIPS", 4))) + " \\\\",
              "$p$ (Wilcoxon) & " + " & ".join(p(st[m]["wilcoxon_p"]) for m in ("PSNR", "SSIM", "LPIPS")) + " \\\\",
              "$d_z$ (Cohen) & " + " & ".join(fs(st[m]["cohen_dz"], 2) for m in ("PSNR", "SSIM", "LPIPS")) + " \\\\",
              "vistas em que GOF+BRDF é melhor & " + " & ".join(
                  f"{st[m]['vistas_b_melhor']}/32" for m in ("PSNR", "SSIM", "LPIPS")) + " \\\\"]
    return "\n".join(linhas) + "\n"


def tab_truck_externos(res, an):
    med = res["medias"] if "medias" in res else res
    nomes = [("GOF-B1(nosso)", "GOF (este trabalho)"), ("GOF+BRDF-E1(nosso)", "GOF+BRDF (este trabalho)"),
             ("GaussianShader", "GaussianShader$^\\dagger$"), ("GOF(outro-pc)", "GOF, SH grau 3 + aparência desacoplada$^\\dagger$")]
    return "\n".join(f"{n} & {f(med[k]['PSNR'], 3)} & {f(med[k]['SSIM'], 4)} & {f(med[k]['LPIPS'], 4)} \\\\" for k, n in nomes) + "\n"


def tab_material(mat):
    linhas = []
    for s in CENAS:
        for i, (m, nome) in enumerate([("gof_brdf", "config. Truck"), ("gof_brdf_ajustado", "ajustado")]):
            d = mat[f"{s}/{m}"]
            linhas.append(f"{s if i == 0 else ''} & {nome} & {f(d['tom_mediano_pond'], 2)} & {f(100 * d['frac_tom_maior_0_1_pond'], 0)}\\% & "
                          f"{f(d['rugosidade_mediana_pond'], 2)} [{f(d['rugosidade_q25_pond'], 2)}; {f(d['rugosidade_q75_pond'], 2)}] \\\\")
        if s != CENAS[-1]:
            linhas.append("\\midrule")
    return "\n".join(linhas) + "\n"


def tab_pares_completa(an):
    linhas = []
    for s in CENAS:
        for i, (chave, st) in enumerate(an[s]["pares"].items()):
            rot = chave.replace(" vs ", " $-$ ")
            cel = [f"{fs(st[m]['delta_medio'], n)} {ic(st[m], n)}" for m, n in (("PSNR", 2), ("SSIM", 4), ("LPIPS", 4))]
            ps = p(st["PSNR"]["wilcoxon_p"])
            linhas.append(f"{s if i == 0 else ''} & {rot} & " + " & ".join(cel) + f" & {ps} \\\\")
        if s != CENAS[-1]:
            linhas.append("\\midrule")
    return "\n".join(linhas) + "\n"


def main(v0):
    dn = os.path.join(v0, "dados", "nero")
    res = {s: json.load(open(os.path.join(dn, f"{s}_results.json"))) for s in CENAS}
    an = {s: json.load(open(os.path.join(dn, f"{s}_analysis.json"))) for s in CENAS}
    perfis = json.load(open(os.path.join(dn, "perfis.json")))
    mat = json.load(open(os.path.join(dn, "material.json")))
    rt = json.load(open(os.path.join(v0, "dados", "truck", "results.json")))
    at = json.load(open(os.path.join(v0, "dados", "truck", "analysis.json")))
    os.makedirs(os.path.join(v0, "tabelas"), exist_ok=True)
    for nome, conteudo in (("nero_metricas", tab_nero(res)), ("nero_pares", tab_pares(an)), ("custo", tab_custo(perfis)),
                           ("truck", tab_truck(rt, at)), ("truck_externos", tab_truck_externos(rt, at)), ("material", tab_material(mat)),
                           ("nero_pares_completa", tab_pares_completa(an))):
        open(os.path.join(v0, "tabelas", nome + ".tex"), "w").write("% Gerado por scripts/gerar_tabelas.py; nao editar a mao.\n" + conteudo)
        print("tabela", nome)


if __name__ == "__main__":
    main(sys.argv[1])
