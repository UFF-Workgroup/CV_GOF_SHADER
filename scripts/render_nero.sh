#!/usr/bin/env bash
# Renderiza as vistas de teste das execucoes do NeRO e registra o tempo de cada renderizacao; pula as ja renderizadas.
#   [GS_DIR=<clone> PY_GS=<python>] METODOS="gof gof_brdf_ajustado gaussianshader" ./scripts/render_nero.sh bell cat teapot
set -uo pipefail
export LC_ALL=C
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
METODOS="${METODOS:-gof_brdf gaussianshader}"
ITER="${ITER:-30000}"
SCENES=("$@"); [ ${#SCENES[@]} -eq 0 ] && SCENES=(bell cat teapot)
PY_GOF="${PY_GOF:-python}"
if [[ " $METODOS " == *" gaussianshader "* ]]; then
  : "${GS_DIR:?defina GS_DIR com o clone do GaussianShader}" "${PY_GS:?defina PY_GS com o python do ambiente do GaussianShader}"
fi
OUT="${OUT:-$REPO/output/nero}"
CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-11.8}"
export CUDA_HOME PATH="$CUDA_HOME/bin:$PATH"
for s in "${SCENES[@]}"; do
  for m in $METODOS; do
    D="$OUT/$s/$m"
    [ -f "$D/render_tempo.json" ] && grep -q '"retorno": 0' "$D/render_tempo.json" && { echo "[render] pula $s/$m"; continue; }
    t0=$(date +%s.%N)
    if [ "$m" = gaussianshader ]; then
      (cd "$GS_DIR" && "$PY_GS" render.py -m "$D" --skip_train --iteration "$ITER" --brdf_dim 0 --sh_degree -1 --brdf_mode envmap --brdf_env 512) > "$D/render.log" 2>&1
    else
      (cd "$REPO" && "$PY_GOF" render.py -m "$D" --skip_train --iteration "$ITER") > "$D/render.log" 2>&1
    fi
    rc=$?
    printf '{"retorno": %d, "tempo_s": %.1f}\n' $rc "$(echo "$(date +%s.%N) - $t0" | bc)" > "$D/render_tempo.json"
    echo "[render] $s/$m rc=$rc"
  done
done
