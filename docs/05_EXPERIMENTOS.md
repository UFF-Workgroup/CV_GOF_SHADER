# Registro de Experimentos

Log corrido. **Nenhum número entra aqui sem um `RUN_ID` e o commit correspondente.**

Formato do ID: `EXP-AAAAMMDD-NN-<slug>` · Artefatos em `docs/experiments/<RUN_ID>/`.

> **Aviso de reprodutibilidade.** O rasterizador usa `atomicAdd`, então a ordem de
> acumulação dos gradientes varia entre execuções e o treino **não é bit-reprodutível**
> mesmo com seeds fixas. Diferenças de PSNR abaixo de ~0,1 dB **não são interpretáveis**
> a partir de uma execução única. Números de destaque (B0, E2) exigem ≥ 2 seeds com média
> ± desvio.

---

## Runs

### SMOKE-20260810-01/02 — teste de fumaça, **não é resultado**

| | baseline | BRDF |
|---|---|---|
| RUN_ID | `SMOKE-20260810-01-base` | `SMOKE-20260810-02-brdf` |
| Commit | `35e8843` | `35e8843` |
| Cena | Truck | Truck |
| Config | `-r 4 --sh_degree 0 --eval --iterations 800 --densify_until_iter 700` | idem + `--brdf --brdf_from_iter 300 --light_frame view` |
| **PSNR teste** | **21.939** | **22.416** |
| PSNR treino | 22.269 | 22.921 |
| L1 teste | 0.05103 | 0.04617 |
| Gaussianas | — | 117 892 |
| Tempo | ~2 min | ~2,3 min |

**Material aprendido (BRDF, iteração 800):**

| | média | mín | máx | σ | init |
|---|---|---|---|---|---|
| `specular_tint` | 0.0455 | 0.0070 | 0.1509 | 0.0100 | 0.05 |
| `roughness` | 0.6938 | 0.4275 | 0.8871 | — | 0.70 |
| envmap | 0.2484 | — | — | 0.3262 | 0.5 ± 0.02 |

**Para que serve este par de runs.** Provar que o pipeline roda ponta a ponta e que o
material **de fato aprende num treino real** — a dispersão por-Gaussiana (σ = 0.0100 no
tint, rugosidade espalhando de 0.43 a 0.89) é a evidência de produção de que o bug A-1
está corrigido. Com o bug, todos os valores estariam presos exatamente em 0.05 e 0.70.

**Por que a diferença de +0,48 dB NÃO deve ser citada como resultado.** Quatro razões,
qualquer uma delas suficiente:

1. **Execução única.** Sem repetição por seed, e com o não-determinismo declarado acima.
2. **Referencial de luz errado de propósito.** Truck tem objeto parado e câmera orbitando
   com sol fixo — o caso em que `--light_frame world` é o correto. Este run usou `view`
   apenas para exercitar o caminho novo.
3. **Capacidade a mais.** O modelo com BRDF tem mais parâmetros; parte de qualquer ganho
   vem de capacidade, não de física. Só a ablação controlada separa as duas coisas.
4. **800 iterações**, das quais só 500 com o BRDF ativo, e `-r 4`. Nada aqui se parece com
   o regime de treino real (30k, `-r 2`).

O número correto a citar virá de B0/B1/E1/E2 conforme `04_PROTOCOLO.md`.

---

### REF-fase2 — referência do baseline de 30k (treinado antes das correções)

| | |
|---|---|
| Origem | `~/Documentos/gaussian-opacity-fields/output/fase2_linux_validacao` |
| Config | Truck, `-r 2 --sh_degree 0 --eval`, 30 000 it |
| **PSNR** | **25.236** |
| SSIM | 0.8868 |
| LPIPS | 0.1337 |

Treinado com o código **anterior às correções**, mas é um baseline **válido do GOF puro**:
os três tensores de material existiam e estavam sujeitos ao bug A-1, porém nenhum kernel os
lia — eram peso morto e não influenciaram um único pixel. Serve como referência para B1.

*Ressalva:* `--sh_degree 0` remove toda dependência de vista, o que isola bem o efeito do
especular mas torna este baseline **artificialmente fraco** para comparação com a
literatura. Por isso B0 (GOF upstream com defaults) continua necessário.

### T8 — extração de malha com BRDF · `35e8843` + `96b8918`

Malha extraída do modelo `SMOKE-20260810-02-brdf`: 1 123 849 vértices, 2 251 678 faces,
**watertight**, zero faces degeneradas, 40,8 MB. 1 059 209 pontos tetra, 7 071 262
tetraedros, 8 passos de busca binária.

Rodar de verdade revelou um bug que nenhum teste unitário pegaria: `evaluage_alpha` usava
`brdf_args` sem recebê-lo — extrair malha de modelo com BRDF estava quebrado
(`NameError`). Corrigido em `96b8918`.

### EXP-20260811-01-b1 — **ABORTADO** (queda de energia), não é resultado

| | |
|---|---|
| Commit | `96b8918` |
| Config | Truck, `-r 2 --sh_degree 0 --eval`, 30 000 it |
| Início | 2026-08-11 00:18:22 |
| Última evidência | `point_cloud/iteration_7000/` gravado às 01:27:51 |
| Desfecho | processo morto por queda de energia; **sem métricas** |

O treino passou da iteração 7000 e foi interrompido em algum ponto depois. Não havia
`--checkpoint_iterations`, então **não existe estado do otimizador para retomar** — o PLY
de 7000 guarda os parâmetros, mas não os momentos do Adam nem o estado da densificação.
Retomar dali não seria o mesmo treino. O run foi **relançado do zero** como
`EXP-20260811-02-b1`.

**Duas lições incorporadas ao processo, não só ao relato:**

1. **O log foi perdido junto.** `train.log` tem apenas 6 linhas úteis, terminando em
   "Computing 3D filter", porque o stdout do Python estava bufferizado e nunca foi
   descarregado. `scripts/run_experiment.sh` passou a invocar `python -u`.
2. **Runs longos agora salvam checkpoint.** Em 30k iterações num ambiente sem no-break, a
   probabilidade de perder tudo não é desprezível. Os runs longos passam a usar
   `--checkpoint_iterations`. Isso só é seguro porque a correção A-2 (`capture`/`restore`
   incluindo o material) foi feita na Fase 1 — antes dela, retomar de checkpoint restauraria
   um modelo com material vazio.

### EXP-20260811-02-b1 — **B1 concluído** · `97cf882`

| | |
|---|---|
| Commit | `97cf882` (branch `feat/brdf-especular`) |
| Cena / config | Truck, `-r 2 --sh_degree 0 --eval`, 30 000 it, sem `--brdf` |
| **PSNR teste @30k** | **25.2311** |
| PSNR treino @30k | 26.8324 |
| L1 teste @30k | 0.03323 |
| PSNR teste @7k | 24.0410 |
| Gaussianas | 2 089 655 |
| **Pico de VRAM** | **3.07 GB** (`max_memory_allocated`) |
| Tempo total | 6 h 50 min (1,22 it/s médio) |

**Fechamento por `metrics.py` — comparação pelo caminho idêntico ao da referência**
(`render.py --skip_train` + `metrics.py -r 2`, 32 vistas de teste):

| | REF-fase2 | **B1** | Δ |
|---|---|---|---|
| PSNR | 25.2358 | **25.2200** | −0.016 dB |
| SSIM | 0.88677 | **0.88632** | −0.00045 |
| LPIPS | 0.13371 | **0.13379** | +0.00009 |

**Conclusão: a reversão da plumbagem CUDA e as correções da auditoria foram neutras para o
baseline.** As três diferenças estão muito abaixo do piso de ruído declarado (~0,1 dB em
PSNR), e as duas métricas independentes (SSIM e LPIPS) concordam com a leitura. Essa era a
pergunta que B1 existe para responder, e **B0 deixa de ser pré-requisito para liberar os
runs E** — continua desejável como referência de literatura, não como portão.

> **A ressalva de método se confirmou quantitativamente.** A avaliação interna do `train.py`
> deu 25.2311 e o `metrics.py` deu 25.2200 sobre o mesmo modelo: **0,011 dB de diferença só
> pela quantização em 8 bits do PNG**, dentro da faixa de 0,01–0,05 dB antecipada. Como esse
> deslocamento é da mesma ordem do efeito que se quer medir, **comparações entre runs devem
> usar sempre o mesmo caminho** — de preferência `metrics.py`, que é o que a literatura
> reporta. Números do log de treino servem para acompanhar, não para comparar.

**Notas de leitura da curva de perda.** Dois padrões aparecem no log e nenhum é
divergência: (i) oscilações em 3k, 6k, 9k e 12k são os resets de opacidade
(`iteration % 3000 == 0`, dentro do bloco `iteration < densify_until_iter`); (ii) um degrau
em **15 000** (0,0495 → 0,0821 na média por janela), que é `distortion_from_iter` e
`depth_normal_from_iter` ligando ao mesmo tempo, com `lambda_distortion = 100`. **A função
de perda muda de definição em 15k**, então valores antes e depois não são comparáveis.

**T9 (regressão de recursos): passa com folga.** Pico de 3,07 GB contra o teto de 5,5 GB.
Sobram ~2,4 GB para o ramo BRDF nos runs E — margem confortável para os +7 parâmetros por
Gaussiana previstos no orçamento.

**Incidente de infraestrutura durante o run.** O driver NVIDIA foi atualizado no disco com
o treino em andamento (NVML 595.84 vs. módulo 595.71.05 carregado), quebrando o
`nvidia-smi`. O treino não foi afetado e processos CUDA novos continuaram subindo — só a
interface de gerência caiu. A medição de VRAM do protocolo é imune por já usar
`torch.cuda.max_memory_allocated()`, interno ao processo. **Um reboot pendente derruba
qualquer treino em curso**; os checkpoints de 10k/20k/30k cobrem esse risco.

---

Os artefatos ficam arquivados em `docs/experiments/EXP-20260811-01-b1/` como registro do
que foi executado, não como resultado.

---

### EXP-20260901-01-e1 (tentativa 1) — **ABORTADO** (loss NaN), não é resultado

| | |
|---|---|
| Commit | `0a4567d` |
| Config | Truck, `-r 2 --sh_degree 0 --eval --brdf --light_frame world`, 30 000 it |
| Início | 2026-09-01 14:52 |
| Loss finito até | iteração 3010 (0,0632) |
| Loss NaN a partir de | iteração 3020 — dez iterações depois de `brdf_from_iter=3000` |
| Desfecho | processo interrompido manualmente ~4200 it (loss NaN havia mais de 1000 it) |

Achado durante o monitoramento, não um crash silencioso: o processo continuou vivo (sem
OOM, VRAM em 5,1/6,1 GB, dentro do teto), só que produzindo `NaN` desde a primeira leva
de iterações com o ramo especular ligado. Causa raiz identificada e corrigida como
**A-5** (`06_AUDITORIA.md`): singularidade de gradiente nos polos do envmap
equirretangular (`atan2`/`acos`). Nenhum checkpoint chegara a ser salvo (interrupção
antes dos 10k), então não há nada para retomar — relançado do zero, sobre o commit com
a correção, como `EXP-20260901-02-e1` (mesma convenção de `EXP-20260811-02-b1`: RUN_ID
novo em vez de reaproveitar o antigo, para não misturar as linhas do tensorboard de uma
tentativa que rodou até NaN com a retomada limpa).

---

## Runs planejados

Ver a matriz completa em `04_PROTOCOLO.md`.

| ID | Status | Pergunta |
|---|---|---|
| B0 | pendente | GOF upstream — referência da literatura |
| B1 | **concluído e fechado** (`EXP-20260811-02-b1`) — PSNR 25.2200 vs 25.2358 (Δ 0,016 dB), SSIM e LPIPS idem: **reversão neutra** | árvore atual sem BRDF — a reversão CUDA foi neutra? |
| E1 | tentativa 1 abortada por A-5 (NaN); tentativa 2 relançada após o fix | BRDF `light_frame=world` em Truck |
| E2 | pendente | BRDF `light_frame=view` na cena de rocha — **contribuição principal** |
| E3 | pendente | `sh_degree ∈ {0,1,2,3}` — quanto de $c_r$ é preciso? |
| E4 | pendente | `light_repr=sh` vs `envmap` |
| E5 | pendente | sem Fresnel |
| E6 | pendente | sem resíduo de normal |
| E7 | pendente | com `L_shading_normal` |

**Bloqueio conhecido para E2.** As cenas de rocha ainda não estão disponíveis neste
ambiente, e falta uma informação da captura: **quantas estações de câmera** foram usadas.
A derivação da mesa giratória (`03_FORMULACAO.md` §6) supõe uma só; com múltiplos anéis é
preciso uma rotação aprendível por estação.
