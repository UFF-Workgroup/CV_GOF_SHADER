# Protocolo Experimental

---

## Ambiente

| | |
|---|---|
| GPU | NVIDIA RTX 3050, **6 GB** (driver 595.71.05) — restrição dominante |
| CUDA / PyTorch | 11.8 / 2.0.1+cu118 |
| Python | 3.9.25, conda env `gof` (`/home/geomesh/miniconda3/envs/gof`) |
| Repositório | `~/Documentos/CV_GOF_SHADER`, branch `feat/brdf-especular` |

**Verificação obrigatória antes de qualquer run.** As extensões são instalações
*editáveis*; se apontarem para outra árvore, o Python importa código antigo **sem aviso**:

```bash
python -c "
import torch, simple_knn, diff_gaussian_rasterization as d, tetranerf
for m in (simple_knn, d, tetranerf):
    assert 'CV_GOF_SHADER' in m.__file__, m.__file__
from diff_gaussian_rasterization import _C
assert _C.rasterize_gaussians.__doc__.count('arg') == 22   # rasterizador == GOF upstream
print('ambiente OK')"
```

---

## Reprodutibilidade — e seus limites

As seeds são fixadas (`random`, `numpy`, `torch` = 0). **Mesmo assim o treino não é
bit-reprodutível:** o rasterizador acumula gradientes com `atomicAdd`, cuja ordem depende
do escalonamento dos blocos CUDA.

Consequências práticas, que precisam constar do artigo:

- Diferenças de PSNR **abaixo de ~0,1 dB não são interpretáveis** a partir de execução
  única.
- Números de destaque (B0, E2) exigem **≥ 2 seeds**, reportados como média ± desvio.
- Toda run registra `git rev-parse HEAD`. Número sem commit não entra em
  `05_EXPERIMENTOS.md`.

---

## Cenas

**Truck** (Tanks&Temples) — cena de **controle**. Objeto parado, câmera orbitando, sol
fixo: é o regime em que `--light_frame world` é o fisicamente correto. Tem ground-truth de
malha e permite comparação com a literatura.

**Testemunhos de rocha** — cena **alvo**. Mesa giratória, luz fixa na sala:
`--light_frame view`. *Bloqueio atual:* falta saber quantas estações de câmera a captura
usa (ver `03_FORMULACAO.md` §6).

Split: `--eval` (o holdout padrão do 3DGS, cada 8ª imagem para teste).

---

## Matriz de runs

| ID | Configuração | Pergunta que responde |
|---|---|---|
| **B0** | GOF upstream, defaults | referência da literatura |
| **B1** | árvore atual, sem `--brdf` | a reversão CUDA e as correções foram neutras? |
| **E1** | `--brdf --light_frame world`, Truck | o especular ajuda com luz fixa no mundo? |
| **E2** | `--brdf --light_frame view`, rocha | **contribuição principal** |
| **E3** | E2 × `--sh_degree {0,1,2,3}` | quanto de $c_r$ é preciso com especular explícito? |
| **E4** | E2 × `--light_repr {envmap,sh}` | quanta alta frequência a luz exige? |
| **E5** | E2 + `--no_fresnel` | o brilho rasante importa em rocha? |
| **E6** | E2 + `--use_normal_residual` | o resíduo de normal se paga? |
| **E7** | E2 + `--lambda_shading_normal > 0` | a `depth_normal_loss` do GOF já basta? |

**B1 é obrigatório antes de qualquer E.** Se B1 divergir de B0 além do ruído, algo nas
correções mudou o baseline e todo E fica sem referência.

**Marco de decisão.** Se E2 não superar B1 **em NVS e em geometria**, parar e diagnosticar
antes de seguir com as ablações. A hipótese de que o especular ajuda pode não valer para
rocha fosca — e isso, com evidência, também é publicável.

---

## Comandos

```bash
# Treino
python train.py -s <fonte> -m output/<RUN_ID> -r 2 --eval \
    --sh_degree 1 --brdf --light_frame view --brdf_from_iter 3000

# Avaliação — NAO repetir os flags de BRDF: eles vêm do cfg_args (ADR-007)
python render.py -m output/<RUN_ID>
python metrics.py -m output/<RUN_ID>
python extract_mesh.py -m output/<RUN_ID> --iteration 30000
```

`scripts/run_experiment.sh` carimba commit, config e ambiente no diretório de saída
automaticamente. Registro manual é registro que não acontece.

---

## Métricas

**Aparência:** PSNR, SSIM, LPIPS (`metrics.py`).
**Geometria:** Chamfer, F1 (`eval_tnt/`).
**Custo:** nº de Gaussianas, **pico de VRAM** (`torch.cuda.max_memory_allocated()`, já
logado no tensorboard), tempo/iteração, tempo total. Em 6 GB o pico é resultado a
reportar, não detalhe operacional.
**Saúde do material:** médias de `specular_tint` e `roughness` no tensorboard. Curvas
chatas nos valores iniciais (0.05 / 0.70) significam material não aprendendo — o sintoma
do bug A-1.
**Relighting:** renders qualitativos com o envmap substituído.

---

## Orçamento de VRAM

Medido para Truck `-r 2` (~2,0 M Gaussianas):

| Configuração | Params/Gauss. | VRAM (param+grad+2 Adam) |
|---|---|---|
| baseline `sh_degree=0` | 14 | ~448 MB |
| + $s,\rho,\Delta n$ | +7 | ~672 MB |
| + `sh_degree=1` | +9 | ~816 MB |
| + `sh_degree=3` | +45 | **~2,1 GB** — inviável |

**Alavancas, em ordem:**
1. `--sh_degree ≤ 1` (e isso é a hipótese E3, não só engenharia)
2. `--lambda_shading_normal 0` — evita uma segunda rasterização por iteração
3. `torch.utils.checkpoint` no sombreamento (~10 tensores `[N,3]` intermediários)
4. poda mais agressiva / teto de Gaussianas
5. `-r 4` como último recurso

> ~~`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`~~ — **indisponível**: só existe a
> partir do torch 2.1; no 2.0.1 aborta com `Unrecognized CachingAllocator option`.
