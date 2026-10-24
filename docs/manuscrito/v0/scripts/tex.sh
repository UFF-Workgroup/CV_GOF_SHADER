# TeX Live em conteiner Docker, compartilhado por compilar.sh e scripts/figuras/pipeline.sh; exige V0 definido.
IMG=texlive-gofbrdf:v0
docker image inspect "$IMG" >/dev/null 2>&1 || docker build -t "$IMG" "$V0/build"
tex() { docker run --rm -u "$(id -u):$(id -g)" ${TEX_ENV:-} -v "$V0":/work -w /work "$IMG" "$@"; }
