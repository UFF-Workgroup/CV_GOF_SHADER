# CV_GOF_SHADER

Sombreamento especular do **GaussianShader** dentro do **Gaussian Opacity Fields (GOF)**, sem
alterar o rasterizador CUDA. O objetivo é reconstruir superfícies brilhantes sem perder a
extração de malha do GOF.

Fork de [`autonomousvision/gaussian-opacity-fields`](https://github.com/autonomousvision/gaussian-opacity-fields)
(SIGGRAPH Asia 2024), com a formulação de sombreamento de
[`asparagus15/GaussianShader`](https://github.com/asparagus15/GaussianShader) (CVPR 2024).

A motivação original eram gêmeos digitais de **testemunhos de rocha**. Como o uso desses dados
não foi autorizado, a avaliação é feita em cenas públicas: Truck (Tanks and Temples) e NeRO
GlossySynthetic (ver ADR-011 em [`docs/02_DECISOES.md`](docs/02_DECISOES.md)).

---

## Sumário

1. [O projeto em uma página](#o-projeto-em-uma-página)
2. [Getting started](#getting-started)
3. [Flags do BRDF](#flags-do-brdf)
4. [Objetos brilhantes (NeRO GlossySynthetic)](#objetos-brilhantes-nero-glossysynthetic)
5. [Testes](#testes)
6. [Mapa do repositório](#mapa-do-repositório)
7. [Guia da documentação](#guia-da-documentação)
8. [Branches e remotos](#branches-e-remotos)
9. [Duas armadilhas do ambiente](#duas-armadilhas-do-ambiente)

---

## O projeto em uma página

**Equação de cor por Gaussiana**, com o tom especular $\mathbf{s}$ e a rugosidade $\rho$
aprendidos por Gaussiana e um mapa de ambiente $L_s$ aprendido por cena:

$$\mathbf{c} = \max\big(c_d + c_r(\omega_o) + F(\mathbf{n}\cdot\omega_o, \mathbf{s}) \odot L_s(\mathbf{r}, \rho),\ 0\big)$$

- $c_d + c_r$ são as SH que o GOF já tinha. Grau 0 é o difuso; os graus maiores são o residual.
  Todos os experimentos usam `--sh_degree 0`, então ali $c_r = 0$.
- $F$ é o Fresnel–Schlick e $\mathbf{r}$ é a reflexão de $\omega_o$ na normal. A normal é o
  eixo de menor escala da Gaussiana, orientado para a câmera.
- $L_s$ é um mapa equirretangular com uma pirâmide de pré-filtragem; $\rho$ escolhe o nível.

**O que o repositório entrega:**

1. **Rasterizador inalterado.** O BRDF fica em PyTorch ([`scene/brdf.py`](scene/brdf.py),
   [`scene/lighting.py`](scene/lighting.py)) e entra no GOF pelo caminho `colors_precomp`, que
   já era diferenciável. Nenhuma derivada foi escrita à mão.
2. **Equivalência verificada.** Com especular zero e Fresnel desligado, o modelo reproduz o GOF
   (diferença máxima < 1e-5, teste T1).
3. **Avaliação controlada.** A mesma máquina e o mesmo protocolo para os dois métodos; IC por
   bootstrap pareado entre vistas, teste de Wilcoxon, e tempo, energia e VRAM medidos.

**Resultados, com uma semente por configuração:**

| Cena | Comparação | ΔPSNR (dB) |
|---|---|---|
| Truck | GOF+BRDF − GOF | +0,41 [0,29; 0,53] |
| NeRO cat / teapot | GOF+BRDF ajustado − GaussianShader | +1,37 / +0,58 |
| NeRO bell | GOF+BRDF ajustado − GaussianShader | −0,01 (empate) |

No NeRO, o GOF+BRDF ajustado gasta de 2,7 a 4,0× menos energia que o GaussianShader e cerca de
metade da VRAM. Detalhes em [`docs/11_RESULTADOS_COMPARATIVOS.md`](docs/11_RESULTADOS_COMPARATIVOS.md),
[`docs/12_RESULTADOS_NERO.md`](docs/12_RESULTADOS_NERO.md) e no preprint
[`docs/manuscrito/v0/`](docs/manuscrito/v0/).

`--light_frame view` (mapa de ambiente no espaço de vista, para captura em mesa giratória) está
implementado e derivado em [`docs/03_FORMULACAO.md`](docs/03_FORMULACAO.md) §6, mas **não foi
validado**: todos os experimentos usam `--light_frame world`.

---

## Getting started

### 1. Pré-requisitos

| Item | Versão usada | Para quê |
|---|---|---|
| Linux + GPU NVIDIA | RTX 3050 6 GB | treino e testes |
| CUDA Toolkit (`nvcc`) | 11.8 | compilar as extensões |
| Conda (Miniconda) | qualquer | ambiente Python |
| CGAL, GMP, CMake | 6.0.1, 6.3.0, 3.25 | só para extrair malha |
| NVIDIA OptiX SDK | 7.6.0 | só para extrair malha (exigido pelo `tetra-nerf-master`) |
| Docker | qualquer | só para compilar o manuscrito |

### 2. Clonar

```bash
# DEV: código, docs/ e manuscrito; o trabalho atual está em feat/brdf-especular
git clone -b feat/brdf-especular git@github.com:UFF-Workgroup/CV_GOF_SHADER.git
# ou, sem chave SSH:
git clone -b feat/brdf-especular https://github.com/UFF-Workgroup/CV_GOF_SHADER.git

cd CV_GOF_SHADER
```

Sem `-b`, o clone abre na `main`, que é a entrega anterior ("Entrega final CV"). Não precisa de
`--recursive`: as dependências em `submodules/` são pastas comuns. A versão de publicação
(PRD, sem `docs/`) fica em `medialab-uff/CV_GOF_SHADER`, de acesso restrito; ver
[Branches e remotos](#branches-e-remotos).

### 3. Ambiente Python

Os comandos abaixo reproduzem as versões do ambiente `gof` usado nos experimentos.
`requirements.txt` e `environment.yml` foram herdados do GOF e do 3DGS e **não** correspondem a
esse ambiente.

```bash
conda create -n gof python=3.9 -y
conda activate gof

pip install torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cu118
pip install "numpy<2" plyfile tqdm opencv-python lpips trimesh einops matplotlib \
            open3d scipy scikit-image GPUtil ninja pytest

# Extensões CUDA, em modo editável, compiladas a partir DESTA árvore
export CUDA_HOME=/usr/local/cuda-11.8
pip install -e submodules/diff-gaussian-rasterization
pip install -e submodules/simple-knn
```

Para extrair malha (`extract_mesh.py`), compile também a triangulação tetraédrica:

```bash
conda install -c conda-forge cgal gmp -y
pip install cmake==3.25.0
export OPTIX_PATH=<pasta do NVIDIA-OptiX-SDK-7.6.0-linux64-x86_64>
cd submodules/tetra-nerf-master
cmake . && make && pip install -e .
cd ../..
```

### 4. Verificar o ambiente

Antes de confiar em qualquer número, confira que as extensões vêm desta árvore e rode os testes:

```bash
python -c "
import torch, simple_knn, diff_gaussian_rasterization as d
for m in (simple_knn, d): assert 'CV_GOF_SHADER' in m.__file__, m.__file__
print('ambiente OK')"

pytest tests/ -q     # 36 testes; precisa de GPU
```

Importe `torch` antes das extensões; sem isso aparece `ImportError: libc10.so`. Se instalou a
triangulação, acrescente `tetranerf` à verificação.

### 5. Dados

- **Truck:** formato COLMAP, com `images/` e `sparse/0/`. Pode ser o Truck do `tandt_db.zip`
  distribuído pelo 3DGS ou uma reconstrução própria feita com `convert.py`.
- **NeRO GlossySynthetic:** precisa de conversão; ver
  [Objetos brilhantes](#objetos-brilhantes-nero-glossysynthetic).

### 6. Primeiro treino

Um teste curto, que já liga o ramo especular (cerca de 15 min numa RTX 3050):

```bash
python train.py -s <Truck> -m output/smoke -r 2 --eval --sh_degree 0 \
    --brdf --light_frame world --iterations 2000 --brdf_from_iter 500
python render.py  -m output/smoke --skip_train
python metrics.py -m output/smoke
```

Os treinos oficiais do Truck usam `scripts/run_experiment.sh`. Ele verifica o ambiente, força
checkpoints e carimba commit, comando e hardware em `docs/experiments/<RUN_ID>/`. A verificação
exige o `tetranerf`, então instale também a triangulação do passo 3:

```bash
# B1: GOF puro (6h51 numa RTX 3050)
./scripts/run_experiment.sh EXP-AAAAMMDD-01-b1 -s <Truck> -r 2 --sh_degree 0 --eval
# E1: GOF+BRDF (8h22)
./scripts/run_experiment.sh EXP-AAAAMMDD-01-e1 -s <Truck> -r 2 --sh_degree 0 --eval \
    --brdf --light_frame world

# Avaliação: NÃO repetir os flags de BRDF, eles vêm do cfg_args
python render.py  -m output/<RUN_ID>
python metrics.py -m output/<RUN_ID>
python extract_mesh.py -m output/<RUN_ID> --iteration 30000
```

A máquina de referência não tem no-break. Se o treino cair, retome do último checkpoint com
`--start_checkpoint output/<RUN_ID>/chkpnt<iter>.pth`.

---

## Flags do BRDF

| Flag | Default | O que faz |
|---|---|---|
| `--brdf` | off | liga a equação de sombreamento |
| `--light_frame` | `view` | `world` = câmera orbita (usado nos experimentos) · `view` = mesa giratória (não validado) |
| `--light_repr` | `envmap` | `envmap` pré-filtrado · `sh` de baixa frequência |
| `--envmap_res` | 128 | altura do mapa equirretangular (largura = 2×) |
| `--lambda_specular_sparse` | 0.001 | peso da esparsidade L1 do tom especular |
| `--depth_normal_from_iter` | 15000 | início da perda de consistência profundidade–normal |
| `--brdf_from_iter` | 3000 | aquecimento antes de ligar o especular |
| `--no_fresnel` | off | ablação: desliga o Fresnel–Schlick |
| `--use_normal_residual` | off | ablação: liga o resíduo de normal $\Delta n$ |
| `--use_tonemap` | off | linear→sRGB; desligado preserva a equivalência com o GOF |

Referência completa de todos os scripts em [`docs/10_GUIA_CLI.md`](docs/10_GUIA_CLI.md).

---

## Objetos brilhantes (NeRO GlossySynthetic)

Benchmark contra o GaussianShader em cenas sintéticas espelhadas, com tempo, energia e VRAM
medidos por treino (`nvidia-smi` a cada 2 s). Requer um clone do
[GaussianShader](https://github.com/Asparagus15/GaussianShader) com seu próprio ambiente.

```bash
# 1. Dados: GlossySynthetic do NeRO convertido pelo nero2blender.py do GaussianShader
#    (precisa de configs/synthetic_split_128.pkl do NeRO); depois a versão lida pelo GOF
python scripts/nero_to_gof.py --src <GlossySynthetic>/bell_blender --dst <GlossySynthetic>/bell_gof

# 2. Treinos em fila serial (pula os já concluídos); GS_DIR e PY_GS só para o GaussianShader
export NERO_DATA=<GlossySynthetic> GS_DIR=<clone do GaussianShader> PY_GS=<python do ambiente gs>
METODOS="gof gof_brdf gof_brdf_ajustado gaussianshader" ./scripts/queue_nero.sh bell cat teapot

# 3. Render das vistas de teste, métricas e gráficos
METODOS="gof gof_brdf gof_brdf_ajustado gaussianshader" ./scripts/render_nero.sh bell cat teapot
python scripts/compare_preds.py --gt output/nero/bell/gof/test/ours_30000/gt_-1 \
    -m GOF=output/nero/bell/gof/test/ours_30000/test_preds_-1 \
    -m GOF+BRDF=output/nero/bell/gof_brdf/test/ours_30000/test_preds_-1 \
    -m "GOF+BRDF ajustado=output/nero/bell/gof_brdf_ajustado/test/ours_30000/test_preds_-1" \
    -m GaussianShader=output/nero/bell/gaussianshader/test/ours_30000/renders \
    --out output/nero/bell/metricas
python scripts/nero_report.py --root output/nero --out output/nero/relatorio
```

`gof_brdf` usa a configuração do Truck. `gof_brdf_ajustado` acrescenta
`--envmap_res 512 --lambda_specular_sparse 0 --depth_normal_from_iter 3000`, para que o
reflexo de superfícies espelhadas não seja borrado nem zerado pela regularização. Variáveis
opcionais: `PY_GOF` (padrão `python`), `CUDA_HOME` (padrão `/usr/local/cuda-11.8`), `ITERS`
(padrão 30000) e `OUT` (padrão `output/nero`). `nero_report.py` espera as quatro variantes
nas três cenas.

---

## Testes

```bash
pytest tests/ -q     # 36 testes; precisa de GPU
```

Dois são bloqueantes:

- **T1:** o BRDF reduz exatamente ao baseline quando o especular é zero. É o que torna a
  ablação interpretável.
- **T3:** o invariante de vínculo com o otimizador, que protege contra a classe de bug descrita
  em [`docs/06_AUDITORIA.md`](docs/06_AUDITORIA.md).

---

## Mapa do repositório

| Caminho | Conteúdo |
|---|---|
| `train.py`, `render.py`, `metrics.py`, `extract_mesh.py` | treino, render das vistas, métricas e extração de malha |
| `arguments/__init__.py` | todos os flags e seus defaults |
| `scene/brdf.py` | Fresnel, reflexão e normal, em PyTorch |
| `scene/lighting.py` | mapa de ambiente pré-filtrado e alternativa em SH |
| `scene/gaussian_model.py` | Gaussianas com tom especular e rugosidade; densificação e poda |
| `gaussian_renderer/__init__.py` | monta a cor por Gaussiana e chama o rasterizador do GOF |
| `scripts/` | experimentos oficiais, fila do NeRO, comparação e figuras |
| `tests/` | testes de sombreamento, ciclo de vida do material e relighting |
| `submodules/` | rasterizador do GOF, `simple-knn` e triangulação tetraédrica (cópias, não submódulos git) |
| `docs/` | documentação do estudo, registro de experimentos e manuscrito (só na DEV) |
| `output/`, `release/` | saídas dos treinos; nunca entram no git |

---

## Guia da documentação

| Se você quer… | Leia |
|---|---|
| Uma visão geral e o estado atual | [`docs/00_VISAO_GERAL.md`](docs/00_VISAO_GERAL.md) |
| Entender tudo do zero, num documento só | [`docs/09_RESUMO_EXECUTIVO.md`](docs/09_RESUMO_EXECUTIVO.md) |
| O método explicado desde o 3DGS | [`docs/01_FUNDAMENTOS.md`](docs/01_FUNDAMENTOS.md) |
| Saber **por que** cada decisão foi tomada | [`docs/02_DECISOES.md`](docs/02_DECISOES.md) (ADRs) |
| Verificar a matemática | [`docs/03_FORMULACAO.md`](docs/03_FORMULACAO.md) |
| Rodar experimentos com rastreabilidade | [`docs/04_PROTOCOLO.md`](docs/04_PROTOCOLO.md) |
| Ver os números medidos | [`docs/05_EXPERIMENTOS.md`](docs/05_EXPERIMENTOS.md), [`11`](docs/11_RESULTADOS_COMPARATIVOS.md), [`12`](docs/12_RESULTADOS_NERO.md) |
| Saber o que estava quebrado no código herdado | [`docs/06_AUDITORIA.md`](docs/06_AUDITORIA.md) |
| Saber onde o método falha | [`docs/07_LIMITACOES.md`](docs/07_LIMITACOES.md) |
| As fórmulas dos artigos-fonte, com equação e página | [`docs/08_ARTIGOS_REFERENCIA.md`](docs/08_ARTIGOS_REFERENCIA.md) |
| Usar os scripts de linha de comando | [`docs/10_GUIA_CLI.md`](docs/10_GUIA_CLI.md) |
| Ler ou recompilar o preprint | [`docs/manuscrito/v0/README.md`](docs/manuscrito/v0/README.md) |

`docs/ARQUIVO_GUIA_DA_APLICACAO_v1.md` é o guia anterior, arquivado.

---

## Branches e remotos

| Remoto | Repositório | Papel |
|---|---|---|
| `origin` (DEV) | `UFF-Workgroup/CV_GOF_SHADER` | estudo e desenvolvimento, com `docs/` |
| `upstream` (PRD) | `medialab-uff/CV_GOF_SHADER` | versão organizada para publicação, sem `docs/` |

- Todo trabalho começa em `feat/brdf-especular`.
- Mudanças vão para o PRD por `git cherry-pick` a partir da branch `prd`, **nunca por merge**,
  para que `docs/` não volte.
- `output/` e `release/` nunca entram no git.

As regras completas estão em [`CLAUDE.md`](CLAUDE.md).

---

## Duas armadilhas do ambiente

**As extensões CUDA são instalações editáveis.** Se apontarem para outra árvore, o Python
importa código antigo **sem nenhum aviso**. Rode a verificação do passo 4 sempre que trocar de
clone ou de ambiente.

**O treino não é bit-reprodutível.** O `atomicAdd` do rasterizador faz a ordem de acumulação
dos gradientes variar. Seeds fixas não bastam: diferenças de PSNR abaixo de ~0,1 dB não são
interpretáveis sem repetição por seed.

---

## Hardware de referência

RTX 3050 **6 GB** · CUDA 11.8 · PyTorch 2.0.1 · Python 3.9. Os 6 GB são a restrição
dominante e moldam o default de `--sh_degree`; ver
[`docs/04_PROTOCOLO.md`](docs/04_PROTOCOLO.md).

## Licença

Herda os termos dos projetos originais: uso não comercial, de pesquisa e avaliação.
Ver [`LICENSE.md`](LICENSE.md).
