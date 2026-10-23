"""Gera referencias.bib a partir de verificacao.json, sem transcricao manual de metadados."""
import json
import re
import sys

# Nome do evento quando o Crossref so traz o nome da serie ou um titulo longo demais.
VEICULO = {
    "mildenhall2020nerf": "European Conference on Computer Vision (ECCV)",
    "huang2024twodgs": "ACM SIGGRAPH 2024 Conference Papers",
    "zhou2025rtrgs": "Proceedings of the 33rd ACM International Conference on Multimedia",
}
TITULO = {"knapitsch2017tanks": "Tanks and Temples: Benchmarking Large-Scale Scene Reconstruction"}
NOTA_ARXIV = {"yao2025refgaussian": "International Conference on Learning Representations (ICLR) 2025",
              "han2026radiometric": "International Conference on Learning Representations (ICLR) 2026"}
ACENTOS = {"é": "{\\'e}", "è": "{\\`e}", "ö": '{\\"o}', "ü": '{\\"u}', "ç": "{\\c{c}}", "ã": "{\\~a}", "á": "{\\'a}", "í": "{\\'i}", "ó": "{\\'o}", "ñ": "{\\~n}"}


def tex(s):
    s = s.replace("&", "\\&").replace("%", "\\%").replace("_", "\\_")
    for k, v in ACENTOS.items():
        s = s.replace(k, v)
    return s


def autor(nome):
    partes = nome.replace(".", ". ").split()
    if len(partes) == 1:
        return tex(partes[0])
    return f"{tex(partes[-1])}, {tex(' '.join(partes[:-1]))}"


def titulo(t):
    # Chaves extras preservam siglas e maiusculas no estilo do .bst.
    return "{" + tex(re.sub(r"\s+", " ", t).strip()) + "}"


def paginas(p):
    return p.replace("-", "--") if p else None


def entrada(r):
    k = r["key"]
    c, a, m = r.get("crossref"), r.get("arxiv"), r.get("manual")
    src = c or m
    campos = {}
    if src:
        campos["author"] = " and ".join(autor(x) for x in src["autores"])
        campos["title"] = titulo(TITULO.get(k, src["titulo"]))
        campos["year"] = str(src["ano"])
        if src.get("doi"):
            campos["doi"] = src["doi"]
        tipo = src.get("tipo")
        if tipo == "journal-article":
            kind = "article"
            campos["journal"] = tex(src["veiculo"])
            for f, g in (("volume", "volume"), ("numero", "number")):
                if src.get(f):
                    campos[g] = str(src[f])
            if src.get("paginas"):
                campos["pages"] = paginas(src["paginas"])
        elif tipo == "course-notes":
            kind = "misc"
            campos["howpublished"] = tex(src["veiculo"])
            campos["url"] = m["url_evidencia"]
        else:
            kind = "inproceedings"
            campos["booktitle"] = tex(VEICULO.get(k, src["veiculo"]))
            if src.get("paginas"):
                campos["pages"] = paginas(src["paginas"])
        if a and kind != "misc":
            campos["note"] = f"arXiv:{a['arxiv']}"
    else:
        kind = "article"
        campos["author"] = " and ".join(autor(x) for x in a["autores"])
        campos["title"] = titulo(a["titulo"])
        campos["year"] = a["publicado"][:4]
        campos["journal"] = f"arXiv preprint arXiv:{a['arxiv']}"
        if k in NOTA_ARXIV:
            campos["note"] = NOTA_ARXIV[k]
    corpo = ",\n".join(f"  {f} = {{{v}}}" for f, v in campos.items())
    return f"@{kind}{{{k},\n{corpo}\n}}\n"


def main(entrada_json, saida):
    v = json.load(open(entrada_json))
    with open(saida, "w") as f:
        f.write("% Gerado por scripts/gerar_bib.py a partir de referencias/verificacao.json; nao editar a mao.\n\n")
        for r in v:
            f.write(entrada(r) + "\n")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
