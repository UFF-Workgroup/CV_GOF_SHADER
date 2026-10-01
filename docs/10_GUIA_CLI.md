# Guia de uso dos scripts CLI

Referência rápida dos parâmetros de linha de comando dos scripts principais, com
exemplos usando o experimento `EXP-20260901-02-e1` (Truck, BRDF world-frame, 30k
iterações) como base. Ambiente: `conda activate gof`.

Convenção importante: `render.py`, `extract_mesh.py` e `metrics.py` leem
`cfg_args` de dentro de `<model_path>/` (salvo pelo `train.py` no início do
treino) e reaplicam automaticamente os parâmetros de `ModelParams` usados
naquele treino -- `--brdf`, `--light_frame`, `-r`, `--sh_degree` etc. Não é
preciso (e não deve) repeti-los na hora de renderizar/extrair/avaliar; só
passe os parâmetros específicos do próprio script.

---

## `train.py` -- treino

```bash
python -u train.py -m <model_path> -s <source_path> -r 2 --sh_degree 0 --eval \
  --brdf --light_frame world \
  --checkpoint_iterations 10000 20000 30000
```

| Parâmetro | Default | O que faz |
|---|---|---|
| `-m/--model_path` | -- | pasta de saída do experimento |
| `-s/--source_path` | -- | pasta do dataset (COLMAP) |
| `-r/--resolution` | -1 | downscale das imagens (2 = metade da resolução) |
| `--sh_degree` | 3 | grau das spherical harmonics (0 desliga, usado com `--brdf`) |
| `--eval` | off | separa vistas em treino/teste (necessário p/ métricas) |
| `--brdf` | off | liga o sombreamento BRDF (GaussianShader) |
| `--light_frame` | `view` | `view` (mesa giratória) ou `world` (câmera orbita) |
| `--light_repr` | `envmap` | `envmap` (pré-filtrado) ou `sh` (baixa frequência) |
| `--no_fresnel` | off | ablação: desliga Fresnel de Schlick |
| `--test_iterations` | `[7000, 30000]` | iterações em que roda avaliação de teste |
| `--checkpoint_iterations` | `[]` | iterações em que salva `chkpntN.pth` (retomável) |
| `--start_checkpoint` | `None` | caminho de um `chkpntN.pth` para retomar treino |

---

## `render.py` -- renderiza imagens 2D (RGB, com o shading BRDF treinado)

Gera PNGs foto-realistas por vista -- é aqui que aparece cor e especularidade,
mas em 2D (uma imagem por câmera), não um objeto 3D navegável.

```bash
# todas as vistas de teste (default sem --views = todas do split)
python render.py -m output/EXP-20260901-02-e1 --iteration 30000 --skip_train

# só vistas específicas, por indice ou por nome de imagem
python render.py -m output/EXP-20260901-02-e1 --iteration 30000 --skip_train \
  --views "3,7,21"
python render.py -m output/EXP-20260901-02-e1 --iteration 30000 --skip_train \
  --views "00027,00019"
```

| Parâmetro | Default | O que faz |
|---|---|---|
| `-m/--model_path` | -- | experimento a renderizar |
| `--iteration` | -1 (última salva) | checkpoint a carregar |
| `--skip_train` / `--skip_test` | off | pula o split de treino/teste |
| `--views` | `None` (todas) | subconjunto: índices e/ou `image_name`, separados por vírgula |

Saída: `<model_path>/{train,test}/ours_<iter>/test_preds_<res>/NNNNN_<image_name>.png`
(e `gt_<res>/` com o ground-truth correspondente, quando existir).

---

## `extract_mesh.py` -- extrai a malha 3D (marching tetrahedra + busca binária)

Isto é o caminho para ver a reconstrução geométrica de verdade -- superfície
triangulada, navegável em 3D (não nuvem de pontos). Sem `--texture_mesh` a
malha sai só com geometria; com a flag, cada vértice recebe a cor final
(difuso + especular) avaliada pelo BRDF treinado.

```bash
# geometria pura (o que já rodamos para validar o fix de VRAM)
python -u extract_mesh.py -m output/EXP-20260901-02-e1 --iteration 30000

# malha colorida + filtrada (recomendado para inspeção visual)
python -u extract_mesh.py -m output/EXP-20260901-02-e1 --iteration 30000 \
  --texture_mesh --filter_mesh
```

| Parâmetro | Default | O que faz |
|---|---|---|
| `-m/--model_path` | -- | experimento a extrair |
| `--iteration` | 30000 | checkpoint a carregar |
| `--filter_mesh` | off | remove triângulos/vértices com `distance > scale` (artefatos soltos) |
| `--texture_mesh` | off | pinta vértices com a cor BRDF avaliada (cria o `.ply` colorido) |
| `--near` / `--far` | 0.02 / 1e6 | limites de profundidade para gerar os pontos tetra |

Saída: `<model_path>/test/ours_<iter>/mesh_binary_search_7.ply` (nome fixo,
sempre sobrescreve; renomeie antes de rodar de novo se quiser manter os dois).
`cells.pt` fica em cache na mesma pasta -- apagar manualmente se mudar `--near`/
`--far` ou o checkpoint.

---

## `mesh_viewer.py` -- visualizador interativo (Open3D)

```bash
python mesh_viewer.py output/EXP-20260901-02-e1/test/ours_30000/mesh_binary_search_7.ply
```

Abre uma janela com rotação/zoom livres. Detecta sozinho se o `.ply` é malha
(tem `element face`) ou nuvem de pontos. Requer um display X ativo (não
funciona puramente via SSH sem `-X`/`-Y` ou um servidor X remoto).

---

## `metrics.py` -- métricas NVS (PSNR/SSIM/LPIPS) por vista

```bash
python metrics.py -m output/EXP-20260901-02-e1
```

| Parâmetro | Default | O que faz |
|---|---|---|
| `-m/--model_paths` | obrigatório | um ou mais `model_path` (aceita múltiplos, `nargs="+"`) |
| `-r/--resolution` | -1 | resolução usada para casar render x GT |

Saída: `<model_path>/results.json` (médias) e `per_view.json` (PSNR/SSIM/LPIPS
por `image_name`) -- é daí que vem o número ao lado do id da foto nas
comparações (ex.: `00027  21.55` = PSNR em dB daquela vista; quanto maior,
melhor).

---

## `scripts/pack_results.py` e `scripts/compare_preds.py` -- comparar com outros metodos

```bash
python scripts/pack_results.py -m output/EXP-20260901-02-e1        # gera release/<RUN_ID>/
python scripts/compare_preds.py --gt release/EXP-20260901-02-e1/gt \
    -m GOF=<pasta_preds_gof> -m GaussianShader=<pasta_preds_gs> -m GOF+BRDF=release/EXP-20260901-02-e1/preds \
    --out comparison
```

`pack_results.py` copia predicoes (`preds/`), ground truth (`gt/`), metricas, `cfg_args`,
comando, commit e `test_views.csv` (indice -> `image_name`; indice k = k-esima vista de teste,
isto e, a imagem `8k` na ordenacao por nome, `llffhold=8`).

`compare_preds.py` casa as vistas pelo indice no inicio do nome (`00027.png` ou
`00027_000217.png`), exige imagens do mesmo tamanho do GT e usa a mesma PSNR/SSIM/LPIPS-VGG
do `metrics.py`. Saidas em `--out`: `results.json` (medias e diferencas par a par),
`per_view.json` e `per_view.csv`.

---

## Fluxo típico ponta-a-ponta

```bash
conda activate gof
python -u train.py -m output/EXP -s <dataset> -r 2 --sh_degree 0 --eval --brdf --light_frame world --checkpoint_iterations 30000
python render.py -m output/EXP --iteration 30000
python metrics.py -m output/EXP
python -u extract_mesh.py -m output/EXP --iteration 30000 --texture_mesh --filter_mesh
python mesh_viewer.py output/EXP/test/ours_30000/mesh_binary_search_7.ply
```
