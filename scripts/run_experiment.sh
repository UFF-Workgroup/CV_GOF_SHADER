#!/usr/bin/env bash
# Roda um experimento carimbando rastreabilidade no diretorio de saida.
#
# Registro manual e registro que nao acontece. Este script garante que todo numero
# citado no artigo possa ser rastreado ate um commit, uma linha de comando e um ambiente.
#
#   ./scripts/run_experiment.sh <RUN_ID> <args do train.py...>
#
# Exemplo:
#   ./scripts/run_experiment.sh EXP-20260811-01-b1 \
#       -s ~/Documentos/Truck -r 2 --sh_degree 0 --eval
#
# A saida vai para output/<RUN_ID>/ e os artefatos de rastreabilidade para
# docs/experiments/<RUN_ID>/ (que tem excecao no .gitignore, de proposito).
set -euo pipefail

if [ $# -lt 2 ]; then
    echo "uso: $0 <RUN_ID> <args do train.py...>" >&2
    exit 1
fi

RUN_ID="$1"; shift
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="$REPO/output/$RUN_ID"
META="$REPO/docs/experiments/$RUN_ID"
PY="${PYTHON:-python}"

mkdir -p "$OUT" "$META"

# --checkpoint_iterations default default (nao apenas lembrete no doc): o incidente
# EXP-20260811-01-b1 perdeu 7000 iteracoes de treino numa queda de energia porque o
# flag nao foi passado -- nao havia estado do otimizador para retomar. Passar o flag
# manualmente e conveniencia que se esquece; injetar aqui e estrutural, no mesmo
# espirito da verificacao de ambiente abaixo. Sem efeito em runs curtos (smoke tests
# com --iterations < 10000): os valores injetados simplesmente nunca sao atingidos.
ARGS=("$@")
HAS_CKPT=0
for a in "${ARGS[@]}"; do
    if [ "$a" = "--checkpoint_iterations" ]; then HAS_CKPT=1; break; fi
done
if [ "$HAS_CKPT" -eq 0 ]; then
    echo "[run_experiment] --checkpoint_iterations nao informado; aplicando default" >&2
    echo "                 10000 20000 30000 (ver EXP-20260811-01-b1 em 05_EXPERIMENTOS.md)." >&2
    ARGS+=(--checkpoint_iterations 10000 20000 30000)
fi

# --- Verificacao de ambiente (D-1): as extensoes sao instalacoes editaveis. Se
# --- apontarem para outra arvore, o Python importa codigo antigo SEM AVISO.
"$PY" - <<'EOF'
import torch, simple_knn, diff_gaussian_rasterization as d, tetranerf
from diff_gaussian_rasterization import _C
for m in (simple_knn, d, tetranerf):
    assert "CV_GOF_SHADER" in m.__file__, f"extensao da arvore errada: {m.__file__}"
n = _C.rasterize_gaussians.__doc__.count("arg")
assert n == 22, f"rasterizador com {n} args, esperado 22 (== GOF upstream)"
print("[run_experiment] ambiente OK")
EOF

if ! git -C "$REPO" diff --quiet HEAD 2>/dev/null; then
    echo "[run_experiment] AVISO: working tree suja. O commit registrado nao descreve" >&2
    echo "                 exatamente o codigo executado. Commite antes de um run oficial." >&2
    git -C "$REPO" diff HEAD > "$META/uncommitted.patch"
fi

git -C "$REPO" rev-parse HEAD > "$META/commit.txt"
git -C "$REPO" rev-parse --abbrev-ref HEAD >> "$META/commit.txt"
printf '%q ' "$PY" -u train.py -m "$OUT" "${ARGS[@]}" > "$META/command.txt"

{
    echo "run_id:   $RUN_ID"
    echo "data:     $(date -Iseconds)"
    echo "host:     $(hostname)"
    "$PY" -c "import torch; print('torch:   ', torch.__version__, '| cuda', torch.version.cuda)"
    "$PY" -c "import torch; print('gpu:     ', torch.cuda.get_device_name(0))"
    nvidia-smi --query-gpu=driver_version,memory.total --format=csv,noheader | sed 's/^/driver:   /'
} > "$META/env.txt"

echo "[run_experiment] $RUN_ID -> $OUT"
cd "$REPO"
# -u (unbuffered): sem isso o stdout do Python so e descarregado no fim, e uma queda de
# energia no meio do treino leva junto TODO o log -- foi exatamente o que aconteceu na
# primeira tentativa do B1 (EXP-20260811-01-b1), que perdeu 7000 iteracoes de registro.
"$PY" -u train.py -m "$OUT" "${ARGS[@]}" 2>&1 | tee "$META/train.log"

# cfg_args e a config efetiva; guarda-la ao lado das metricas evita ter de reconstruir
# depois quais flags estavam ligados.
cp -f "$OUT/cfg_args" "$META/cfg_args" 2>/dev/null || true

echo "[run_experiment] concluido. Registre em docs/05_EXPERIMENTOS.md com RUN_ID=$RUN_ID"
