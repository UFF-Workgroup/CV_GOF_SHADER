#!/usr/bin/env bash
# Regera dados, figuras, tabelas e o PDF do manuscrito; o LaTeX roda num conteiner TeX Live (sem instalar nada no sistema).
#   ./compilar.sh            # so o PDF
#   ./compilar.sh tudo       # dados + figuras + tabelas + PDF (precisa das saidas dos treinos em output/)
set -euo pipefail
V0="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$V0/../../.." && pwd)"
PY="${PY_GOF:-python}"
IMG=texlive-gofbrdf:v0
docker image inspect "$IMG" >/dev/null 2>&1 || docker build -t "$IMG" "$V0/build"
if [ "${1:-}" = tudo ]; then
  "$PY" "$V0/scripts/extrair_dados.py" "$REPO" "$V0"
  "$PY" "$V0/scripts/gerar_figuras.py" "$REPO" "$V0"
  python3 "$V0/scripts/gerar_tabelas.py" "$V0"
fi
tex() { docker run --rm -u "$(id -u):$(id -g)" -v "$V0":/work -w /work "$IMG" "$@"; }
tex sh -c "cd figuras && pdflatex -interaction=nonstopmode -halt-on-error pipeline.tex >/dev/null && rm -f pipeline.aux pipeline.log"
tex pdflatex -interaction=nonstopmode -halt-on-error gof_brdf_preprint.tex >/dev/null
tex bibtex gof_brdf_preprint >/dev/null
tex pdflatex -interaction=nonstopmode -halt-on-error gof_brdf_preprint.tex >/dev/null
tex pdflatex -interaction=nonstopmode -halt-on-error gof_brdf_preprint.tex >/dev/null
grep -E "Warning.*(undefined|Citation)" "$V0/gof_brdf_preprint.log" || echo "sem referencias indefinidas"
echo "PDF: $V0/gof_brdf_preprint.pdf"
