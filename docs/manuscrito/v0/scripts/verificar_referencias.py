"""Confere cada referencia candidata no Crossref (DOI ou busca por titulo) e no arXiv e grava os metadados oficiais."""
import difflib
import json
import re
import sys
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET

UA = {"User-Agent": "manuscrito-gof-brdf/0.1 (verificacao de referencias)"}
ATOM = "{http://www.w3.org/2005/Atom}"
ARXIV = "{http://arxiv.org/schemas/atom}"


def get(url):
    for tentativa in range(3):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=40) as r:
                return r.read()
        except Exception as e:  # noqa: BLE001
            erro = e
            time.sleep(3 * (tentativa + 1))
    raise erro


def norm(t):
    return re.sub(r"[^a-z0-9]+", " ", (t or "").lower()).strip()


def sim(a, b):
    return round(difflib.SequenceMatcher(None, norm(a), norm(b)).ratio(), 3)


def crossref_msg(m):
    return {
        "doi": m.get("DOI"),
        "titulo": (m.get("title") or [""])[0],
        "autores": [f"{a.get('given', '')} {a.get('family', '')}".strip() or a.get("name", "") for a in m.get("author", [])],
        "veiculo": (m.get("container-title") or [""])[0],
        "volume": m.get("volume"),
        "numero": m.get("issue"),
        "paginas": m.get("page") or m.get("article-number"),
        "ano": (m.get("issued", {}).get("date-parts") or [[None]])[0][0],
        "tipo": m.get("type"),
        "editora": m.get("publisher"),
    }


def crossref_doi(doi):
    return crossref_msg(json.loads(get("https://api.crossref.org/works/" + urllib.parse.quote(doi)))["message"])


def crossref_busca(titulo):
    q = urllib.parse.urlencode({"query.bibliographic": titulo, "rows": 5})
    itens = json.loads(get("https://api.crossref.org/works?" + q))["message"]["items"]
    melhores = sorted((crossref_msg(m) for m in itens), key=lambda x: -sim(x["titulo"], titulo))
    return melhores[0] if melhores and sim(melhores[0]["titulo"], titulo) > 0.9 else None


def arxiv(aid):
    root = ET.fromstring(get("https://export.arxiv.org/api/query?id_list=" + aid))
    e = root.find(ATOM + "entry")
    if e is None or e.find(ATOM + "title") is None:
        return None
    jr = e.find(ARXIV + "journal_ref")
    doi = e.find(ARXIV + "doi")
    return {
        "arxiv": aid,
        "titulo": " ".join(e.find(ATOM + "title").text.split()),
        "autores": [a.find(ATOM + "name").text for a in e.findall(ATOM + "author")],
        "publicado": e.find(ATOM + "published").text[:10],
        "journal_ref": jr.text if jr is not None else None,
        "doi": doi.text if doi is not None else None,
    }


def main(entrada, saida):
    cand = json.load(open(entrada))
    out = []
    for c in cand:
        r = {"key": c["key"], "fonte": c["fonte"], "titulo_esperado": c["titulo"], "crossref": None, "arxiv": None, "erros": []}
        try:
            if c.get("doi"):
                r["crossref"] = crossref_doi(c["doi"])
            elif c.get("query"):
                r["crossref"] = crossref_busca(c["query"])
        except Exception as e:  # noqa: BLE001
            r["erros"].append(f"crossref: {e}")
        try:
            if c.get("arxiv"):
                r["arxiv"] = arxiv(c["arxiv"])
        except Exception as e:  # noqa: BLE001
            r["erros"].append(f"arxiv: {e}")
        if c.get("manual"):
            try:
                pagina = get(c["url_evidencia"]).decode("utf-8", "ignore")
                r["manual"] = {**c["manual"], "url_evidencia": c["url_evidencia"], "titulo_na_pagina": norm(c["titulo"]) in norm(pagina)}
            except Exception as e:  # noqa: BLE001
                r["erros"].append(f"evidencia: {e}")
        sims = [sim(x["titulo"], c["titulo"]) for x in (r["crossref"], r["arxiv"]) if x]
        if r.get("manual", {}).get("titulo_na_pagina"):
            sims.append(1.0)
        r["similaridade_titulo"] = max(sims) if sims else 0.0
        r["status"] = "confirmada" if r["similaridade_titulo"] >= 0.9 else ("divergente" if sims else "nao_encontrada")
        out.append(r)
        print(f"{r['status']:15s} {r['similaridade_titulo']:.2f} {c['key']}", flush=True)
        time.sleep(1)
    json.dump(out, open(saida, "w"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
