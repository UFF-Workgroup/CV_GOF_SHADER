"""Gera REFERENCIAS.md: catalogo das referencias verificadas, onde cada uma e citada e o que sustenta no texto."""
import json
import re
import sys

# O que cada referencia sustenta no manuscrito (para o revisor conferir contra a fonte).
USO = {
    "kerbl2023gaussian": "3DGS: Gaussianas anisotrópicas rasterizadas em tempo real; cor em SH; perda L1 + D-SSIM (λ=0,2).",
    "yu2024gof": "GOF: interseção raio–Gaussiana, normal pelo plano de interseção, conjunto de nível e Marching Tetrahedra, sem TSDF/Poisson.",
    "jiang2024gaussianshader": "Função de sombreamento (difuso, tom especular, rugosidade, residual em SH de 3ª ordem); normal pelo eixo mais curto; sem Fresnel na Eq. 3; L_sparse sobre a opacidade (Eq. 8); Tabela 6: +0,19 dB médio em Tanks and Temples e +0,04 dB no Truck; Tabela 5: valores de bell, cat e teapot.",
    "verbin2022refnerf": "Reparametrização pela direção de reflexão (Eq. 4); atenuação de bandas por rugosidade (IDE); SH baixo borra reflexo nítido.",
    "mezieres2022hvl": "Evidência de que reflexos nítidos exigem muitas bandas SH (os autores relatam precisar de 15 bandas para o reflexo especular forte do chão da cena Sponza).",
    "mildenhall2020nerf": "NeRF como representação implícita de referência.",
    "liu2023nero": "Objetos refletores violam a consistência multivista; split-sum + IDE; origem do conjunto Glossy Synthetic.",
    "knapitsch2017tanks": "Origem da cena Truck (Tanks and Temples).",
    "huang2024twodgs": "2DGS: discos 2D, interseção raio–disco, perdas de distorção e consistência de normal.",
    "guedon2024sugar": "SuGaR: alinhamento das Gaussianas à superfície e extração por Poisson.",
    "yu2024mipsplatting": "Filtro 3D usado pela implementação do GOF.",
    "liang2024gsir": "GS-IR: normal por regularização da derivada da profundidade; oclusão por baking.",
    "shi2025gir": "GIR: normal pelo menor autovetor; iluminação indireta por voxels.",
    "wu2025deferredgs": "Um único SH mistura textura e iluminação; sombreamento diferido.",
    "ye2025geosplatting": "Renderização inversa guiada pela geometria.",
    "yao2025refgaussian": "Ref-Gaussian: renderização diferida com split-sum por pixel, inter-reflexão baseada em Gaussianas, Gaussianas 2D.",
    "gu2025irgs": "IRGS: equação de renderização completa com traçado de raios em Gaussianas 2D.",
    "gu2026irgsext": "IRGS++: extensão do IRGS (terceira onda: luz indireta explícita).",
    "lai2025glossygs": "Renderização inversa específica para objetos brilhantes.",
    "liang2025gusir": "Sombreamento unificado direto/diferido.",
    "zhou2025rtrgs": "Modelo híbrido direto + diferido para reflexos.",
    "sun2025svgir": "Material variando dentro de cada Gaussiana.",
    "chen2024gigs": "Iluminação global com sombreamento diferido.",
    "bao2025survey": "Levantamento geral sobre 3DGS.",
    "moenne2024gaussianrt": "Traçado de raios de partículas Gaussianas.",
    "guedon2025milo": "Extração diferenciável de malha durante o treino.",
    "lyu2024dgsr": "Acoplamento de SDF às Gaussianas.",
    "jiang2025geometryfield": "Splatting de campos de geometria com surfels.",
    "jiang2025radiosity": "Transporte de luz por radiosidade com surfels.",
    "li2026pdrgs": "Colapso da geometria para dentro em cenas especulares.",
    "han2026radiometric": "Consistência radiométrica para luz indireta.",
    "xu2026gspi": "Geração de materiais PBR por difusão.",
    "yang2025gogs": "Objetos brilhantes com surfels 2D em dois estágios.",
    "noras2025gains": "Renderização inversa com poucas vistas.",
    "zhang2024radegs": "Rasterização de profundidade e normal de Gaussianas gerais.",
    "zhu2025gsror2": "Geometria guiada por SDF para objetos refletores.",
    "zhu2025dsdf": "SDF discretizado dentro das Gaussianas.",
    "zhu2026ptirgs": "Renderização inversa por traçado de caminhos em campos Gaussianos.",
    "munkberg2022nvdiffrec": "Formulação diferenciável do split-sum.",
    "schlick1994fresnel": "Aproximação de Fresnel (Eq. 3 do manuscrito).",
    "walter2007ggx": "Lóbulo GGX usado no split-sum (que nosso filtro caixa não reproduz).",
    "wang2004ssim": "Métrica SSIM.",
    "zhang2018lpips": "Métrica LPIPS.",
    "wilcoxon1945": "Teste de postos sinalizados.",
    "efron1979bootstrap": "Bootstrap.",
    "strubell2019energy": "Importância de reportar o custo energético do treino.",
    "schonberger2016colmap": "Poses da cena Truck estimadas por structure-from-motion (COLMAP).",
    "karis2013ue4": "Aproximação split-sum da pré-filtragem da iluminação.",
}
# Divergencias entre o levantamento do Consensus e os metadados oficiais.
NOTA = {
    "jiang2024gaussianshader": "Levantamento cita só o arXiv e omite o último autor (Yuexin Ma); versão publicada no CVPR 2024. PDF local: docs/artigos/2311.17977v1.pdf.",
    "verbin2022refnerf": "PDF local: docs/artigos/2112.03907v1.pdf.",
    "mezieres2022hvl": "PDF local: docs/artigos/2201.01487v1.pdf; o arXiv indica revisão na Computer Graphics Forum.",
    "bao2025survey": "Levantamento dá 2024; publicado em 2025, v. 35, n. 7.",
    "wu2025deferredgs": "Levantamento dá 2024; publicado em 2025, v. 47, n. 8.",
    "lai2025glossygs": "Levantamento dá 2024; publicado em 2025, v. 31, n. 10.",
    "liang2025gusir": "Levantamento dá 2024; publicado em 2025, v. 47, n. 10.",
    "shi2025gir": "Levantamento dá DOI do arXiv; versão TPAMI 2025 (acesso antecipado, sem volume) encontrada por busca no Crossref.",
    "moenne2024gaussianrt": "Levantamento dá o DOI do arXiv para a versão do TOG; DOI correto 10.1145/3687934.",
    "zhang2024radegs": "Levantamento dá 2024 (ano do arXiv); Crossref: TOG v. 45, n. 2, 2026.",
    "zhu2025gsror2": "Levantamento dá 2024; Crossref: TOG v. 45, n. 1, 2025.",
    "jiang2025geometryfield": "Levantamento dá 2024 (arXiv); publicado no CVPR 2025.",
    "gu2025irgs": "Levantamento cita só o arXiv; versão CVPR 2025 encontrada por busca no Crossref.",
    "yao2025refgaussian": "Sem DOI; o arXiv registra aceite no ICLR 2025.",
    "han2026radiometric": "Sem DOI; o arXiv registra ICLR 2026 (oral).",
    "zhu2026ptirgs": "Levantamento sem identificador nem ano; localizado no arXiv (2606.09606), autores conferem.",
    "chen2024gigs": "Só arXiv; comentário 'camera-ready' sem nome do evento.",
    "knapitsch2017tanks": "O Crossref guarda o título sem o subtítulo; DOI, autores, volume e páginas conferem.",
    "walter2007ggx": "DOI fora do Crossref; conferido na biblioteca digital da Eurographics.",
    "karis2013ue4": "Notas de curso sem DOI; conferido na página do curso.",
}


def secoes_por_chave(tex):
    out, secao = {}, "Resumo"
    for linha in tex.splitlines():
        m = re.match(r"\\(?:sub)?section\*?\{(.+?)\}", linha)
        if m:
            secao = m.group(1)
        for grupo in re.findall(r"\\cite\{([^}]+)\}", linha):
            for k in grupo.split(","):
                out.setdefault(k.strip(), [])
                if secao not in out[k.strip()]:
                    out[k.strip()].append(secao)
    return out


def main(verif, resumos, tex, saida):
    v = json.load(open(verif))
    r = json.load(open(resumos))
    usos = secoes_por_chave(open(tex).read())
    n_cit = sum(1 for x in v if x["key"] in usos)
    linhas = [
        "# Catálogo de referências do manuscrito (v0)",
        "",
        "Gerado por `scripts/gerar_catalogo.py`; não editar à mão.",
        "",
        "## Como as referências foram verificadas",
        "- **Origem.** `levantamento` = listada no levantamento do Consensus (`docs/manuscrito/referencias/Este_artigo_apresenta_uma_nova_abordagem_unificada.pdf`); "
        "`projeto` = artigo já usado nos documentos do projeto; `novo` = acrescentada ao escrever o manuscrito.",
        "- **Metadados.** Cada DOI foi resolvido no Crossref (`api.crossref.org/works/<doi>`) e cada identificador arXiv na API do arXiv; "
        "quando havia só o título, buscamos no Crossref e aceitamos o resultado apenas com similaridade de título ≥ 0,9. "
        "Os dados brutos estão em `referencias/verificacao.json`.",
        "- **Conteúdo.** O resumo oficial de cada trabalho foi baixado (arXiv ou OpenAlex, em `referencias/resumos.json`) e comparado com o que o texto afirma; "
        "a coluna *Uso no texto* resume essa afirmação para conferência. Para GaussianShader, Ref-NeRF e HVL, o texto completo foi lido (PDFs em `docs/artigos/`).",
        "- **BibTeX.** `referencias.bib` é gerado de `verificacao.json` por `scripts/gerar_bib.py`, sem transcrição manual.",
        "",
        f"**Resumo:** {len(v)} referências catalogadas; {sum(x['status'] == 'confirmada' and not x.get('manual') for x in v)} confirmadas automaticamente, "
        f"as demais conferidas à mão (ver notas); {n_cit} citadas no texto.",
        "",
        "## Pontos de atenção para a revisão",
        "- O levantamento do Consensus é gerado por IA: errou anos (versão arXiv no lugar da publicada), DOIs (arXiv no lugar do editor) e omitiu autores. "
        "Nenhum metadado do levantamento foi usado sem conferência.",
        "- O levantamento conclui que a novidade defensável é a integração GOF + sombreamento, não a parametrização de material; o manuscrito declara as contribuições nesses termos.",
        "- Referências de 2026 sem DOI (arXiv) ainda podem mudar de versão; conferir antes da submissão.",
        "",
        "## Catálogo",
        "",
        "| # | Chave | Referência | Identificador | Status | Origem | Citada em | Uso no texto | Notas |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for i, x in enumerate(v, 1):
        c, a, m = x.get("crossref"), x.get("arxiv"), x.get("manual")
        src = c or m or {}
        autores = src.get("autores") or (a or {}).get("autores") or []
        aut = autores[0] + (" et al." if len(autores) > 1 else "") if autores else "?"
        titulo = src.get("titulo") or (a or {}).get("titulo")
        if x["key"] == "knapitsch2017tanks":
            titulo = "Tanks and Temples: Benchmarking Large-Scale Scene Reconstruction"
        ano = src.get("ano") or ((a or {}).get("publicado") or "")[:4]
        veic = src.get("veiculo") or "arXiv"
        ids = []
        if src.get("doi"):
            ids.append(f"[doi:{src['doi']}](https://doi.org/{src['doi']})")
        if a:
            ids.append(f"[arXiv:{a['arxiv']}](https://arxiv.org/abs/{a['arxiv']})")
        if m and m.get("url_evidencia") and not src.get("doi"):
            ids.append(f"[página]({m['url_evidencia']})")
        status = "conferida à mão" if (x.get("manual") or x["status"] != "confirmada") else "confirmada"
        cit = "; ".join(usos.get(x["key"], [])) or "não citada"
        linhas.append(f"| {i} | `{x['key']}` | {aut} ({ano}). *{titulo}*. {veic}. | {' '.join(ids)} | {status} | {x['fonte']} | {cit} | "
                      f"{USO.get(x['key'], '')} | {NOTA.get(x['key'], '')} |")
    linhas += ["", "## Resumos oficiais (para conferência)", ""]
    for x in v:
        res = (r.get(x["key"]) or {}).get("resumo")
        linhas.append(f"- **`{x['key']}`** ({(r.get(x['key']) or {}).get('origem') or 'sem resumo disponível'}): "
                      + (res[:600] + ("…" if len(res) > 600 else "") if res else "resumo não disponível nas APIs; trabalho clássico, conferido pelos metadados."))
    open(saida, "w").write("\n".join(linhas) + "\n")


if __name__ == "__main__":
    main(*sys.argv[1:5])
