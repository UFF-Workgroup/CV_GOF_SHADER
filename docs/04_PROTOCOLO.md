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
- Números de destaque (B0, E1) exigem **≥ 2 seeds**, reportados como média ± desvio.
- Toda run registra `git rev-parse HEAD`. Número sem commit não entra em
  `05_EXPERIMENTOS.md`.

---

## Cenas

**Truck** (Tanks&Temples) — cena **única** deste protocolo (ADR-011). Objeto parado, câmera
orbitando, sol fixo: é o regime em que `--light_frame world` é o fisicamente correto. Tem
ground-truth de malha e permite comparação com a literatura.

> **Testemunho FS16 (`~/Documentos/rocha_fs16_16cm`) está fora do escopo ativo.** Não há
> autorização de uso desse dataset (ADR-011, 2026-09-23) — não é uma limitação técnica, é um
> impedimento de uso dos dados. Toda a preparação já feita para essa cena (verificação de
> captura por correlação de fase, decisão de não mascarar fundo — ADR-010, resolução de
> `images_8`) permanece documentada abaixo **só como referência arquivada**, caso uma
> autorização futura reabra essa linha; nenhuma run ativa depende dela.

<details>
<summary>Preparo arquivado da cena de rocha (fora de escopo — ADR-011)</summary>

`--light_frame view`. A amostra gira sobre **dois roletes** enquanto a câmera fotografa
faixa a faixa, do topo para a base; as luzes ficam paradas na bancada. Cada foto é um
*focus stack* de 15 camadas (1080 fotos brutas → 72 finais, 9504×6336).

A estrutura da captura foi **verificada a partir dos pixels**, não do relato
(`scripts/prepare_turntable.py check` → `preparo/captura.json`): por correlação de fase o
passo entre quadros consecutivos é constante dentro da faixa (+0,437 do quadro, dispersão
0,007) e destoa na troca de faixa. Os **7 passos atípicos caem exatamente nas 7 fronteiras
esperadas, nenhum fora** — confirmando 8 faixas × 9 poses = 72.

As 8 faixas diferem por uma **translação** ao longo do eixo, e a derivação de
`03_FORMULACAO.md` §6 exige apenas que a **rotação** sala→câmera seja constante —
`--num_light_stations` não seria necessário aqui.

*Sem máscaras de fundo* — a premissa de fundo estático foi medida e é falsa nesta captura
(ADR-010).

**Resolução de treino planejada.** `prepare_turntable.py images --downscale 8` gera
`images_8/` (1188×792), comparável ao Truck em `-r 2` (960×540). Treinaria com
`-i images_8 -r 1`.

</details>

---

## Matriz de runs

| ID | Configuração | Pergunta que responde |
|---|---|---|
| **B0** | GOF upstream, defaults | referência da literatura |
| **B1** | árvore atual, sem `--brdf`, Truck | a reversão CUDA e as correções foram neutras? |
| **E1** | `--brdf --light_frame world`, Truck | o especular ajuda com luz fixa no mundo? **É a comparação central do projeto (B1 vs. E1) — ADR-011.** |
| **E3** | E1 × `--sh_degree {0,1,2,3}` | quanto de $c_r$ é preciso com especular explícito? |
| **E4** | E1 × `--light_repr {envmap,sh}` | quanta alta frequência a luz exige? |
| **E5** | E1 + `--no_fresnel` | o brilho rasante importa? |
| **E6** | E1 + `--use_normal_residual` | o resíduo de normal se paga? |
| **E7** | E1 + `--lambda_shading_normal > 0` | a `depth_normal_loss` do GOF já basta? |

~~B2~~ e ~~E2~~ (controle e run principal em rocha) foram removidas da matriz ativa por
ADR-011 — sem autorização de uso do dataset, não há como rodá-las. Os IDs não são
reaproveitados (convenção do log de ADRs); se uma cena substituta ou autorização surgir,
entram como novos IDs.

**B1 é obrigatório antes de qualquer E.** Se B1 divergir de B0 além do ruído, algo nas
correções mudou o baseline e todo E fica sem referência.

**Marco de decisão.** Se E1 não superar B1 **em NVS e em geometria**, parar e diagnosticar
antes de seguir com as ablações (E3–E7). A hipótese de que o especular ajuda pode não valer
para Truck (cena majoritariamente fosca) — e isso, com evidência, também é publicável; ver
`08_ARTIGOS_REFERENCIA.md` §1.7 sobre o ganho esperado ser modesto em cenas reais difusas.

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
**Relighting:** renders qualitativos com o envmap substituído — `scripts/relight.py`.

```bash
python scripts/relight.py -m output/<RUN_ID> --views 0 12 24
```

Quatro baterias sintéticas (procedurais, sem HDR externo), cada uma variando **uma**
dimensão: `estudio_lateral` (fonte única e aguda), `contraluz_dupla` (duas fontes de
temperaturas opostas), `ceu_chao` (baixa frequência pura) e `uniforme` (sem estrutura).
Todas são reescalonadas para a **mesma radiância média do mapa aprendido** — sem isso a
comparação mede exposição, não distribuição de luz. O mapa aprendido entra como controle.

`uniforme` não é decoração: é o **controle negativo**. Se a imagem sob luz uniforme for
indistinguível da renderizada com o mapa aprendido, o ramo especular não está
contribuindo — e nenhuma métrica de NVS revelaria isso sozinha, porque um especular
inerte não piora o PSNR, só deixa de ajudar.

---

## Orçamento de VRAM

> **Meça com `torch.cuda.max_memory_allocated()`, não com `nvidia-smi`.**
> Medido em B1 (Truck `-r 2 --sh_degree 0`, iteração 6777, 1,64 M Gaussianas):
> `nvidia-smi` reportava **5,67 GB** enquanto o pico real de alocação era **2,91 GB**.
> A diferença de ~2× é o pool reservado-mas-livre do alocador do PyTorch mais o contexto
> CUDA — o PyTorch cresce a pool e não a devolve. Ler `nvidia-smi` levaria à conclusão
> falsa de que o treino está a 92 % do limite e prestes a estourar.
>
> O valor de `vram/peak_gb` já vai para o tensorboard a cada iteração; é ele que deve
> entrar na tabela do artigo.

**Curva medida em B1** (pico alocado, GB):

| iteração | 848 | 1695 | 2542 | 3389 | 4236 | 5083 | 5930 | 6777 |
|---|---|---|---|---|---|---|---|---|
| pico (GB) | 1,86 | 2,16 | 2,46 | 2,58 | 2,63 | 2,77 | 2,89 | 2,91 |
| Gaussianas (M) | 0,18 | 0,67 | 1,15 | 1,08 | 1,43 | 1,64 | 1,84 | 1,64 |

Os incrementos caem (0,30 → 0,30 → 0,12 → 0,05 → 0,14 → 0,12 → 0,02) e a contagem de
Gaussianas **oscila** em vez de crescer monotonamente, porque poda e reset de opacidade
contrabalançam a densificação. Com `sh_degree=0` sobra folga confortável nos 6 GB.

**Estimativa analítica** de parâmetros + gradiente + 2 estados do Adam, para
dimensionar configurações ainda não medidas (~2,0 M Gaussianas):

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
