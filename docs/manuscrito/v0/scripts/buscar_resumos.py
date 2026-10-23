"""Baixa o resumo oficial de cada referencia verificada (arXiv ou OpenAlex) para conferir o que o texto afirma sobre ela."""
import json
import os
import sys
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET

# Contato opcional para o pool "polite" do OpenAlex, lido do ambiente para nao ficar no repositorio.
UA = {"User-Agent": "manuscrito-gof-brdf/0.1" + (f" (mailto:{os.environ['MAILTO']})" if os.environ.get("MAILTO") else "")}
ATOM = "{http://www.w3.org/2005/Atom}"


def get(url):
    with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=40) as r:
        return r.read()


def arxiv_resumo(aid):
    e = ET.fromstring(get("https://export.arxiv.org/api/query?id_list=" + aid)).find(ATOM + "entry")
    return " ".join(e.find(ATOM + "summary").text.split())


def openalex_resumo(doi):
    w = json.loads(get("https://api.openalex.org/works/doi:" + urllib.parse.quote(doi)))
    inv = w.get("abstract_inverted_index")
    if not inv:
        return None
    pos = sorted((p, t) for t, ps in inv.items() for p in ps)
    return " ".join(t for _, t in pos)


def main(entrada, saida):
    out = {}
    for r in json.load(open(entrada)):
        k, resumo, origem = r["key"], None, None
        try:
            if r.get("arxiv"):
                resumo, origem = arxiv_resumo(r["arxiv"]["arxiv"]), "arXiv"
            elif r.get("crossref") and r["crossref"].get("doi"):
                resumo, origem = openalex_resumo(r["crossref"]["doi"]), "OpenAlex"
        except Exception as e:  # noqa: BLE001
            origem = f"erro: {e}"
        out[k] = {"origem": origem, "resumo": resumo}
        print(f"{k:24s} {origem} {len(resumo or '')}", flush=True)
        time.sleep(1)
    json.dump(out, open(saida, "w"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
