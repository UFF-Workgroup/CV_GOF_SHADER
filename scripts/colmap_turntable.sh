#!/usr/bin/env bash
# COLMAP para a cena de testemunho em roletes -- irmao do convert.py, com os parametros
# de extracao e casamento ajustados a esta captura.
#
#   ./scripts/colmap_turntable.sh ~/Documentos/rocha_fs16_16cm
#
# POR QUE NAO O convert.py DIRETO. A captura tem duas escalas de movimento muito
# diferentes, e isso se ve no banco de matches do primeiro run com os defaults:
#
#   pares dentro de uma faixa      : mediana de 2400-3400 inliers  (camera deslizando)
#   pares entre faixas adjacentes  : mediana de     24-60 inliers  (rolete girando ~45 graus)
#
# Dentro de uma faixa a camera translada ao longo do eixo do testemunho e a sobreposicao e
# enorme. Entre faixas o rolete gira a amostra, e sobra uma tira estreita de superficie
# comum, vista em angulo rasante nas duas fotos. Com os defaults, esses elos fracos nao
# sustentam o mapper: ele registrou 54 das 72 fotos e abriu um segundo modelo com o resto
# -- ou seja, faltariam ~90 graus da circunferencia na reconstrucao.
#
# Tres ajustes, cada um atacando uma causa distinta dessa fraqueza:
#
#   1. max_image_size 6400 (default 3200) e max_num_features 32768 (default 8192).
#      As fotos tem 9504x6336; a 3200 px o COLMAP descarta justamente o detalhe fino da
#      tira de sobreposicao, que e estreita. Mais features onde ha textura e o lever de
#      maior retorno.
#   2. guided_matching: usa a geometria epipolar estimada para recuperar casamentos que o
#      teste de razao descartaria. Ajuda exatamente onde ha poucos e bons.
#   3. abs_pose_min_num_inliers 15 (default 30): com 24-60 inliers medidos, o default
#      rejeita registros que sao legitimos. Baixar o limiar so e defensavel porque a
#      geometria e verificada depois em inspect_poses.py.
#
# NAO usar --SiftExtraction.estimate_affine_shape aqui. Ele seria o ajuste teoricamente
# certo para a tira em angulo rasante, mas forca o caminho de CPU, e o COLMAP abre uma
# thread por nucleo: 48 threads x uma imagem de 6400 px cada estouraram os 47 GB de RAM
# (SIGKILL). Se os elos entre faixas ainda faltarem depois destes ajustes, ele volta --
# com --SiftExtraction.num_threads 8.
set -euo pipefail

SRC="${1:?uso: $0 <caminho da cena>}"
CAMERA="${CAMERA:-OPENCV}"

if [ -e "$SRC/distorted" ] || [ -e "$SRC/sparse" ]; then
    echo "[colmap_turntable] $SRC ja tem reconstrucao; remova distorted/ sparse/ images/ stereo/ antes" >&2
    exit 1
fi

mkdir -p "$SRC/distorted/sparse"

echo "[colmap_turntable] extracao de features"
colmap feature_extractor \
    --database_path "$SRC/distorted/database.db" \
    --image_path "$SRC/input" \
    --ImageReader.single_camera 1 \
    --ImageReader.camera_model "$CAMERA" \
    --SiftExtraction.max_image_size 6400 \
    --SiftExtraction.max_num_features 32768

echo "[colmap_turntable] casamento exaustivo"
colmap exhaustive_matcher \
    --database_path "$SRC/distorted/database.db" \
    --SiftMatching.guided_matching 1 \
    --SiftMatching.max_num_matches 32768

echo "[colmap_turntable] mapper"
colmap mapper \
    --database_path "$SRC/distorted/database.db" \
    --image_path "$SRC/input" \
    --output_path "$SRC/distorted/sparse" \
    --Mapper.abs_pose_min_num_inliers 15 \
    --Mapper.ba_global_function_tolerance=0.000001

echo "[colmap_turntable] remocao de distorcao"
colmap image_undistorter \
    --image_path "$SRC/input" \
    --input_path "$SRC/distorted/sparse/0" \
    --output_path "$SRC" \
    --output_type COLMAP

# Mesmo remanejamento que o convert.py faz: o 3DGS espera sparse/0/.
python - "$SRC" <<'EOF'
import os, shutil, sys
src = sys.argv[1]
os.makedirs(os.path.join(src, "sparse", "0"), exist_ok=True)
for f in os.listdir(os.path.join(src, "sparse")):
    if f != "0":
        shutil.move(os.path.join(src, "sparse", f), os.path.join(src, "sparse", "0", f))
EOF

n_modelos=$(ls "$SRC/distorted/sparse" | wc -l)
n_imagens=$(ls "$SRC/images" | wc -l)
n_entrada=$(ls "$SRC/input" | wc -l)
echo "[colmap_turntable] modelos: $n_modelos | registradas: $n_imagens de $n_entrada"
if [ "$n_modelos" -gt 1 ]; then
    echo "[colmap_turntable] AVISO: reconstrucao fragmentada -- sparse/0 nao tem a cena toda" >&2
fi
echo "[colmap_turntable] verifique a geometria: python scripts/inspect_poses.py -s $SRC"
