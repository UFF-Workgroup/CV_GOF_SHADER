# Registro de Decisões de Arquitetura (ADRs)

Log **append-only**. Cada decisão recebe um número que nunca é reusado. Se uma decisão
for revertida, escreve-se um ADR novo que a supersede — o original permanece, porque o
raciocínio descartado é parte do registro científico.

Formato: Contexto · Opções · Decisão · Consequências.

---

## ADR-001 — Sombreamento BRDF em PyTorch, não em CUDA

**Data:** 2026-08-10 · **Status:** aceito · **Supersede:** Fases 3 e 4 de
`ARQUIVO_GUIA_DA_APLICACAO_v1.md`

**Contexto.** O plano original previa implementar a equação de sombreamento dentro de
`cuda_rasterizer/forward.cu` e derivar analiticamente todos os gradientes à mão em
`backward.cu`. A auditoria do rasterizador mostrou que o GOF já expõe um caminho
`colors_precomp` — uma cor RGB por Gaussiana, calculada fora — e que o backward já
devolve o gradiente correspondente:

- `cuda_rasterizer/backward.cu` → `atomicAdd(&dL_dcolors[global_id*C+ch], ...)`
- `rasterize_points.cu` → devolve `dL_dcolors`
- `diff_gaussian_rasterization/__init__.py` → entrega como `grad_colors_precomp`

**Opções.**
1. BRDF em CUDA, com derivadas manuais (plano original).
2. BRDF em PyTorch, entregue como `colors_precomp`, com autograd.

**Decisão.** Opção 2.

**Consequências.**
- *Positivas.* Zero derivada escrita à mão — elimina a classe de erro mais cara e mais
  difícil de detectar neste tipo de trabalho: um gradiente silenciosamente errado, que
  não gera exceção e só se manifesta como convergência ruim. Ciclo de iteração cai de
  "recompilar CUDA" para "salvar arquivo". O rasterizador volta a ser **idêntico ao GOF
  upstream**, o que é um argumento a favor no artigo. O caminho `integrate` (extração de
  malha) herda o sombreamento sem trabalho extra.
- *Negativa, e declarada.* O sombreamento passa a ser **por-Gaussiana** (uma cor por
  splat por vista), não por-pixel. A nitidez do reflexo fica limitada pela densidade de
  Gaussianas. É o que o GaussianShader oficial faz, e funciona — mas é uma limitação
  real. Registrada em `07_LIMITACOES.md`; a variante por-pixel fica como trabalho futuro.

---

## ADR-002 — Environment map ancorado no espaço de vista (mesa giratória)

**Data:** 2026-08-10 · **Status:** aceito

**Contexto.** A captura dos testemunhos usa **mesa giratória**: o objeto gira, a câmera e
as luzes ficam paradas na sala. No referencial do objeto — que é o que o COLMAP
reconstrói — a iluminação gira junto com a câmera. Um environment map ancorado no mundo,
que é o que o GaussianShader e o resto da literatura assumem, é **fisicamente inválido**
aqui: tentaria explicar como estático um reflexo que se move.

**Opções.**
1. Envmap no mundo (padrão da literatura) — errado para este setup.
2. Estimar explicitamente o eixo da mesa e o ângulo por imagem, e desfazer a rotação.
3. Ancorar o envmap no espaço de vista.

**Decisão.** Opção 3, com a opção 1 disponível via `--light_frame world` como controle.

A derivação (completa em `03_FORMULACAO.md`) mostra que a opção 3 é **exata**, não uma
aproximação: sendo $W_i = C R_i$ a matriz mundo→câmera do COLMAP, com $C$ constante
(câmera fixa na sala) e $R_i$ a rotação da mesa, uma direção $r_O$ do objeto vista da
sala é $r_R = C^{-1} r_{\text{view}}$. Como $C$ é uma rotação global constante, o mapa
aprendido a absorve. **Não é preciso estimar o eixo da mesa nem o ângulo por imagem** —
a opção 2 é trabalho desnecessário.

**Consequências.**
- Uma linha de código (`reflect_dirs @ R_w2v.T`) resolve um erro de modelagem que
  invalidaria os reflexos.
- Ganho de consistência: as normais renderizadas pelo GOF já saem em espaço de vista.
- **Suposição declarada:** uma única estação de câmera. Com $K$ alturas/anéis, $C$ muda
  por estação e é preciso uma rotação aprendível por estação. Mitigação prevista em
  `07_LIMITACOES.md`.
- Para relighting, o mapa vive no referencial da sala; renderizar sob luz "de mundo"
  exige fixar um ângulo de mesa e compor $C$.

---

## ADR-003 — `c_r(ω_o)` é a base SH que já existe, não um tensor novo

**Data:** 2026-08-10 · **Status:** aceito

**Contexto.** A implementação anterior criou `_residual_color`, um RGB fixo por
Gaussiana. O artigo define o residual como $c_r(\omega_o)$ — uma **função da direção de
vista**. Um RGB fixo não é função de $\omega_o$: apenas duplica o termo difuso e agrava
a ambiguidade difuso/especular.

**Decisão.** Remover `_residual_color`. O papel de $c_r(\omega_o)$ passa para os graus
$\ell \geq 1$ das SH que o 3DGS já carrega em `_features_rest`.

| Termo do artigo | Onde vive no código |
|---|---|
| $c_d$ (difuso) | `_features_dc` (grau 0) |
| $c_r(\omega_o)$ (residual) | `_features_rest` (graus ≥ 1), via `eval_sh` |
| $s$ (tom especular) | `_specular_tint` |
| $\rho$ (rugosidade) | `_roughness` |
| $\Delta n$ (resíduo de normal) | `_normal_residual` |

**Consequências.** Corrige a física, elimina um tensor redundante (3 floats + 3 estados
de Adam por Gaussiana — relevante em 6 GB), e responde ao pedido de "usar harmônicos
esféricos" de forma fundamentada em vez de decorativa. O grau das SH vira um eixo de
ablação (E3): *com o especular explícito, de quanta SH ainda se precisa?*

---

## ADR-004 — O modelo com BRDF reduz **exatamente** ao baseline

**Data:** 2026-08-10 · **Status:** aceito

**Contexto.** Para atribuir uma diferença de PSNR ao termo especular, é preciso garantir
que nada mais mudou. Se o caminho BRDF reparametrizasse a cor difusa, a comparação
mediria as duas coisas somadas e a ablação perderia o sentido.

**Decisão.** $c_d$ e $c_r$ **não** são calculados em separado no forward: `eval_sh` sobre
todas as SH já devolve a soma dos dois. A cor final é
`clamp_min(eval_sh(...) + 0.5 + especular, 0)`. Com $s=0$ e sem Fresnel, isso é
literalmente a expressão do caminho `convert_SHs_python` do GOF.

**Consequências.** A propriedade é **testada**, não assumida: `test_t1_*` compara as
imagens renderizadas pelos dois caminhos e exige `max|Δpixel| < 1e-5`. O teste é
bloqueante.

*Ressalva registrada:* com o Fresnel de Schlick ligado a identidade não vale, e
corretamente — $F = F_0 + (1-F_0)(1-\cos)^5$ tende a 1 em ângulo rasante mesmo com
$F_0 = 0$. Há um teste que verifica justamente que o Fresnel quebra a redução, para que
ninguém "conserte" isso por engano.

---

## ADR-005 — Grau de SH é decisão de orçamento de memória, e vira hipótese

**Data:** 2026-08-10 · **Status:** aceito

**Contexto.** RTX 3050 com 6 GB. Truck a `-r 2` chega a ~2,0 M Gaussianas.

| Configuração | Params/Gauss. | VRAM (param+grad+2 Adam) |
|---|---|---|
| baseline `sh_degree=0` | 14 | ~448 MB |
| + $s,\rho,\Delta n$ | +7 | ~672 MB |
| + `sh_degree=3` | +45 | **~2,1 GB** — inviável |
| + `sh_degree=1` | +9 | ~816 MB |

**Decisão.** `sh_degree ≤ 1` como default do modelo com BRDF, e o grau vira eixo de
ablação (E3) em vez de constante.

**Consequências.** A restrição de hardware é convertida em pergunta científica: *com o
especular modelado explicitamente, a necessidade de SH de alta ordem cai?* Se sim, é
resultado de artigo, não concessão. Se não, a limitação fica documentada honestamente.

---

## ADR-006 — Reverter a plumbagem C++/CUDA já escrita

**Data:** 2026-08-10 · **Status:** aceito

**Contexto.** As Fases 1–2 anteriores canalizaram três tensores por Python → PyBind11 →
C++ → assinaturas CUDA. Nenhum kernel os lia. Além de código morto, o wrapper passava
`torch.Tensor([])` (tensor de **CPU**) e `rasterize_points.cu` fazia `.data<float>()`
nele, entregando ponteiro de *host* a kernel CUDA — inofensivo apenas porque nada
desreferenciava.

**Decisão.** Remover tudo. Assinaturas conferidas após a reversão: forward 25 → 22 args,
backward 27 → 24, integrate 26 → 23.

**Consequências.** Elimina risco de ABI, remove a obrigação de recompilar a cada
alteração, e permite afirmar no artigo que **o método não modifica o rasterizador**.
Custo: descarta trabalho já feito — aceito, porque manter código morto que parece
funcional é pior que apagá-lo.

---

## ADR-007 — Flags do BRDF em `ModelParams`, não em `PipelineParams`

**Data:** 2026-08-10 · **Status:** aceito

**Contexto.** `render.py` e `extract_mesh.py` reconstroem a configuração com
`get_combined_args`, que mescla o `cfg_args` gravado no treino com a linha de comando.
`cfg_args` contém apenas `ModelParams`.

**Decisão.** `brdf`, `light_frame`, `light_repr`, `envmap_res`, `use_tonemap`, etc. vão
em `ModelParams`. Só o que é exclusivo de treino (`brdf_from_iter`, `envmap_lr`, os
`lambda_*`) fica em `OptimizationParams`.

**Consequências.** Avaliar um modelo treinado com BRDF não exige repetir os flags à mão.
Se ficassem em `PipelineParams`, esquecer um flag na avaliação produziria números
silenciosamente errados — a mesma classe de falha do bug A-3.

---

## ADR-008 — Rugosidade é não-identificável sob luz uniforme

**Data:** 2026-08-10 · **Status:** aceito (achado, não escolha)

**Contexto.** Descoberto porque o teste de fluxo de gradiente falhou para
`light_repr="sh"`: $\partial L/\partial\rho$ era **exatamente zero**. A investigação
mostrou que não é bug, é identificabilidade — se toda direção tem a mesma radiância,
borrar o lóbulo especular não muda a imagem, então $\rho$ não deixa assinatura e não pode
ser estimada.

O envmap "passava" no mesmo teste apenas por ruído de ponto flutuante (gradiente 2e-05
com variância de saída 3e-17, isto é, numericamente constante) — ou seja, **o teste
estava fraco**, não o código certo.

**Decisão.** Três consequências, todas implementadas:
1. `EnvironmentMap` e `SHLighting` inicializam com ruído pequeno (`init_std=0.02`) para
   não partir de um ponto exatamente degenerado.
2. O teste que exigia gradiente não-nulo sob luz uniforme estava **errado** e foi
   substituído por um que **documenta** a degenerescência.
3. A limitação vai para `07_LIMITACOES.md`.

**Consequências para as cenas de rocha.** Se o testemunho for fotografado em caixa de luz
difusa — situação comum em bancada — a iluminação é quase uniforme, $\rho$ fica mal
condicionado e **o valor final de rugosidade deve ser lido com ceticismo**. Isso precisa
constar do artigo antes de qualquer interpretação petrofísica dos mapas de material.

---

## ADR-009 — Invariante de vínculo com o otimizador como assert de runtime

**Data:** 2026-08-10 · **Status:** aceito

**Contexto.** O bug A-1 (ver `06_AUDITORIA.md`) quebrou a identidade entre o tensor que o
modelo usa e o que o Adam atualiza. Não gerou exceção, não mudou formas: apenas congelou
os parâmetros a partir da iteração 600. É praticamente indetectável numa curva de perda.

**Decisão.** `_assert_optimizer_binding()` roda após **toda** densificação e poda,
verificando `self._X is optimizer.param_groups[k]["params"][0]` para todo parâmetro
por-Gaussiana. Além disso, `MATERIAL_PARAMS` vira fonte única da verdade, da qual
inicialização, otimizador, densificação, poda, checkpoint e PLY são todos derivados.

**Consequências.** Custo desprezível (~9 comparações de identidade por densificação).
A causa raiz — manter listas de parâmetros em sincronia à mão em cinco lugares — deixa
de existir. É defesa **estrutural**, não apenas um teste, porque a falha é silenciosa e
um teste só protege o que alguém lembrou de testar.

---

## ADR-010 — Sem máscaras de fundo na cena de testemunho (premissa medida, não suposta)

**Data:** 2026-08-12 · **Status:** aceito

**Contexto.** A captura do testemunho FS16 põe a amostra sobre dois roletes que a giram
enquanto a câmera fotografa faixa a faixa, do topo para a base. Dentro de uma faixa a
câmera fica **parada**. Essa configuração tem um modo de falha clássico e silencioso no
SfM: um fundo estático visto por uma câmera parada tem **paralaxe zero**, o COLMAP casa
essas features de bom grado, conclui que a câmera não se moveu, e o objeto — a única
coisa que se move — vira outlier. Sai uma reconstrução; só que errada.

Pelo risco, o plano previa mascarar o fundo antes do COLMAP. A máscara seria obtida sem
modelo aprendido: dentro de uma faixa o fundo é constante no tempo e a rocha não, então o
desvio-padrão temporal por pixel separaria os dois.

**O que a medição mostrou.** Duas coisas, ambas contra a hipótese:

1. **Não há fundo estático a mascarar.** A região escura aparece com 33 % do quadro em
   **um** dos 72 quadros (o 001) e cai para 1–3 % nos demais — e essa fração residual é
   sombra da própria rocha. O que parecia fundo é a borda do cilindro girando para fora de
   vista: rastreando a região escura ao longo da faixa 01, o centroide anda junto com a
   textura, em vez de ficar parado como um fundo ficaria.
2. **O discriminador não discrimina.** Numa região de fundo declarado, o desvio-padrão
   temporal deu **28,9**, contra **21,1** na rocha. Otsu, aplicado a uma distribuição
   dominada por rocha, escolhe um limiar *dentro* da rocha: as máscaras resultantes
   cobriam de 33 % a 66 % do quadro e recortavam o próprio testemunho.

**Decisão.** Não mascarar. O COLMAP roda pelo `convert.py` upstream, idêntico ao usado no
Truck — o que preserva a comparabilidade entre as duas cenas e elimina uma etapa não
justificada do protocolo.

**O que sobrevive do trabalho descartado.** A verificação da estrutura da captura, em
`scripts/prepare_turntable.py check`: por correlação de fase, o passo entre quadros
consecutivos é constante dentro da faixa (o giro, +0,437 do quadro) e destoa na troca de
faixa (o deslize da câmera). Os 7 passos atípicos caíram **exatamente** nas 7 fronteiras
esperadas, nenhum fora — confirmando 8 faixas × 9 poses = 72 a partir dos pixels, e não
do relato. É esse teste que sustenta a suposição de orientação de câmera constante da qual
`--light_frame view` depende (ADR-002).

**Consequência de método.** A hipótese do fundo estático era plausível, barata de testar e
falsa. Testá-la custou noventa segundos de estatística; aceitá-la teria custado um COLMAP
sobre máscaras que cortam a amostra — e, pior, uma reconstrução plausível o bastante para
não levantar suspeita. Vale como precedente: **premissa sobre os dados se mede antes de
virar etapa de pipeline.**

---

## ADR-011 — Escopo restrito a Truck/Tanks&Temples; testemunho de rocha fica fora por falta de autorização

**Data:** 2026-09-23 · **Status:** aceito

**Contexto.** Todo o desenho experimental até aqui (`04_PROTOCOLO.md`) girava em torno de
**E2** — `--brdf --light_frame view` na cena `rocha_fs16_16cm` — como a run que responde à
pergunta central do projeto (*o sombreamento especular ajuda a reconstruir testemunhos de
rocha fotografados em mesa giratória?*). B2 (controle na mesma cena), a investigação da
fragmentação do COLMAP e os passos 1–2 de `09_RESUMO_EXECUTIVO.md` §6 dependiam todos dela.

**O que mudou.** Não há autorização de uso do dataset de rocha (`~/Documentos/rocha_fs16_16cm`).
Isso não é uma limitação técnica (como o COLMAP fragmentado) — é um impedimento de uso dos
dados em si, e não se resolve investigando ou reprocessando nada.

**Opções consideradas.**
1. Buscar outra cena pública/autorizada com especularidade forte (ex.: as cenas do próprio
   artigo do GaussianShader — Glossy Synthetic, Shiny Blender) para preservar a pergunta
   "especular ajuda em superfícies reflexivas?" com um substituto da rocha.
2. Pausar a definição de escopo e só marcar a rocha como bloqueada, sem redefinir o que a
   substitui.
3. **Reformular a contribuição como metodológica, restrita a Truck/Tanks&Temples.**

**Decisão.** Opção 3. O projeto passa a comparar **só GOF puro (B1) vs. GOF+BRDF (E1)**,
ambos em Truck — já é, por construção, a comparação "método separado vs. método mesclado"
que a pergunta original buscava (ADR local, não precisa de uma reprodução à parte do
GaussianShader original, que teria um backbone diferente — sem campo de opacidade nem
extração de malha — e não daria uma comparação limpa). A contribuição do projeto deixa de
ser "validar a fusão no caso de uso de gêmeos digitais de rocha" e passa a ser:

1. A fusão em si (GOF + sombreamento BRDF do GaussianShader, sem modificar o rasterizador —
   ADR-001) e sua validação em NVS + geometria, com múltiplas seeds, em Truck.
2. As ablações E3–E7 (grau de SH, envmap vs. SH, Fresnel, resíduo de normal), agora rodadas
   sobre **E1** em vez de E2 — mesma pergunta científica (*por que o especular ajuda, quando
   ajuda*), só que sem depender da rocha.
3. A correção do referencial de luz para mesa giratória (ADR-002, `03_FORMULACAO.md` §6)
   permanece documentada como contribuição teórica, com a ressalva explícita de que **não
   foi validada empiricamente** nesta fase — fica como trabalho futuro, condicionado a
   autorização de uso de uma cena em mesa giratória.

**Consequências.**
- `04_PROTOCOLO.md`: B2 e E2–E7 (rocha) saem da matriz ativa; E3–E7 são redefinidas sobre
  E1/Truck. O "marco de decisão" (parar e diagnosticar se o especular não ajudar) passa a
  valer para E1 vs. B1.
- A investigação da fragmentação do COLMAP da rocha (bloqueio nº1 do plano anterior) **não
  é mais necessária** para o caminho crítico do projeto — fica arquivada, não descartada,
  caso uma autorização futura reabra essa linha.
- O orçamento de tempo do projeto cai (não há mais B2/E2 nem as extrações de malha
  correspondentes) — ver `04_PROTOCOLO.md` para a matriz atualizada.
- `00_VISAO_GERAL.md` e `09_RESUMO_EXECUTIVO.md` precisam de reescrita para não apresentar
  "gêmeos digitais de testemunhos de rocha" como o objetivo corrente do trabalho.
