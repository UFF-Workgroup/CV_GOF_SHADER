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

## Runs planejados

Ver a matriz completa em `04_PROTOCOLO.md`.

| ID | Status | Pergunta |
|---|---|---|
| B0 | pendente | GOF upstream — referência da literatura |
| B1 | pendente | árvore atual sem BRDF — a reversão CUDA foi neutra? |
| E1 | pendente | BRDF `light_frame=world` em Truck |
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
