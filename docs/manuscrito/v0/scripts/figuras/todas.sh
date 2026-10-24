#!/usr/bin/env bash
# Regera todas as figuras do artigo; PY_GOF aponta para um python com matplotlib (e torch, para envmaps.py).
#   ./scripts/figuras/todas.sh             # todas, inclusive as que leem renders e checkpoints em output/
#   ./scripts/figuras/todas.sh so-dados    # so as que dependem apenas de dados/, que esta no git
set -euo pipefail
D="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="${PY_GOF:-python}"
SO_DADOS="nero_metricas efeitos_pareados custo_qualidade mecanismo potencia por_vista_ajustado_gs"
COM_OUTPUT="envmaps truck nero_qualitativo"
if [ "${1:-}" = so-dados ]; then LISTA="$SO_DADOS"; else LISTA="$COM_OUTPUT $SO_DADOS"; fi
for f in $LISTA; do "$PY" "$D/$f.py"; done
"$D/pipeline.sh"
