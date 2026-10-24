#!/usr/bin/env bash
# Regera dados, figuras, tabelas e o PDF do manuscrito; o LaTeX roda num conteiner TeX Live (sem instalar nada no sistema).
#   ./compilar.sh            # so o PDF, com as figuras e tabelas que ja estao em figuras/ e tabelas/
#   ./compilar.sh tudo       # dados + figuras + tabelas + PDF (precisa das saidas dos treinos em output/)
# Para refazer uma figura so, rode o script dela em scripts/figuras/ (ver FIGURAS.md) e depois ./compilar.sh.
set -euo pipefail
V0="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$V0/../../.." && pwd)"
PY="${PY_GOF:-python}"
source "$V0/scripts/tex.sh"
if [ "${1:-}" = tudo ]; then
  "$PY" "$V0/scripts/extrair_dados.py" "$REPO" "$V0"
  PY_GOF="$PY" "$V0/scripts/figuras/todas.sh"
  python3 "$V0/scripts/gerar_tabelas.py" "$V0"
fi
tex pdflatex -interaction=nonstopmode -halt-on-error gof_brdf_preprint.tex >/dev/null
tex bibtex gof_brdf_preprint >/dev/null
tex pdflatex -interaction=nonstopmode -halt-on-error gof_brdf_preprint.tex >/dev/null
tex pdflatex -interaction=nonstopmode -halt-on-error gof_brdf_preprint.tex >/dev/null
grep -E "Warning.*(undefined|Citation)" "$V0/gof_brdf_preprint.log" || echo "sem referencias indefinidas"
echo "PDF: $V0/gof_brdf_preprint.pdf"
