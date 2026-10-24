#!/usr/bin/env bash
# Fig. 1 (fig:pipeline): compila o diagrama TikZ figuras/pipeline.tex em figuras/pipeline.pdf, no conteiner TeX Live.
# Para ajustar, edite figuras/pipeline.tex (cores no topo, textos em cada \node) e rode este script.
set -euo pipefail
V0="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "$V0/scripts/tex.sh"
# Data fixa, para o PDF sair identico quando o .tex nao muda.
TEX_ENV="-e SOURCE_DATE_EPOCH=0 -e FORCE_SOURCE_DATE=1" tex sh -c "cd figuras && pdflatex -interaction=nonstopmode -halt-on-error pipeline.tex >/dev/null && rm -f pipeline.aux pipeline.log"
echo "figura $V0/figuras/pipeline.pdf"
