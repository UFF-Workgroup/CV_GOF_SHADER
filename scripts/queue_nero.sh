#!/usr/bin/env bash
# Fila serial de treinos no NeRO GlossySynthetic com perfil de tempo/energia/VRAM; pula execucoes ja concluidas.
#   NERO_DATA=<pasta> [GS_DIR=<clone> PY_GS=<python>] METODOS="gof gof_brdf_ajustado gaussianshader" ./scripts/queue_nero.sh bell cat teapot
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ITERS="${ITERS:-30000}"
METODOS="${METODOS:-gof_brdf gaussianshader}"
TAG="${TAG:-}"
SCENES=("$@"); [ ${#SCENES[@]} -eq 0 ] && SCENES=(bell cat teapot)
DATA="${NERO_DATA:?defina NERO_DATA com a pasta do GlossySynthetic convertido}"
PY_GOF="${PY_GOF:-python}"
if [[ " $METODOS " == *" gaussianshader "* ]]; then
  : "${GS_DIR:?defina GS_DIR com o clone do GaussianShader}" "${PY_GS:?defina PY_GS com o python do ambiente do GaussianShader}"
fi
OUT="${OUT:-$REPO/output/nero}"
CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-11.8}"
export CUDA_HOME PATH="$CUDA_HOME/bin:$PATH"
CK=$((ITERS/3))
GOF_COMUM=(--eval -w --sh_degree 0 --iterations "$ITERS" --test_iterations "$ITERS" --save_iterations "$ITERS"
           --checkpoint_iterations $CK $((2*CK)) "$ITERS")
# Ajuste para objetos especulares: envmap na resolucao do GaussianShader, sem L_sparse e normal regularizada desde o warm-up.
AJUSTE=(--envmap_res 512 --lambda_specular_sparse 0 --depth_normal_from_iter 3000)

for s in "${SCENES[@]}"; do
  for m in $METODOS; do
    D="$OUT/$s/$m$TAG"
    if [ -f "$D/perfil.json" ] && grep -q '"retorno": 0' "$D/perfil.json"; then echo "[queue] pula $s/$m$TAG (concluido)"; continue; fi
    echo "[queue] $(date -Iseconds) inicio $s/$m$TAG"
    case "$m" in
      gof)               args=(--python "$PY_GOF" --cwd "$REPO" --out "$D" -- -s "$DATA/${s}_gof" "${GOF_COMUM[@]}") ;;
      gof_brdf)          args=(--python "$PY_GOF" --cwd "$REPO" --out "$D" -- -s "$DATA/${s}_gof" "${GOF_COMUM[@]}" --brdf --light_frame world) ;;
      gof_brdf_ajustado) args=(--python "$PY_GOF" --cwd "$REPO" --out "$D" -- -s "$DATA/${s}_gof" "${GOF_COMUM[@]}" --brdf --light_frame world "${AJUSTE[@]}") ;;
      gaussianshader)    args=(--python "$PY_GS" --cwd "$GS_DIR" --out "$D" -- -s "$DATA/${s}_blender" --eval -w --brdf_dim 0 --sh_degree -1
                               --lambda_predicted_normal 2e-1 --brdf_env 512 --iterations "$ITERS" --test_iterations "$ITERS" --save_iterations "$ITERS") ;;
      *) echo "[queue] metodo desconhecido: $m" >&2; continue ;;
    esac
    "$PY_GOF" "$REPO/scripts/run_benchmark.py" "${args[@]}"
    echo "[queue] $(date -Iseconds) fim $s/$m$TAG rc=$?"
  done
done
