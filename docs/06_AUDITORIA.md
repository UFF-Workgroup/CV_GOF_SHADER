# Auditoria do código herdado (2026-08-10)

Registro do que estava errado no estado recebido, por que estava errado, e como foi
corrigido. Existe para rastreabilidade: se um número antigo aparecer numa comparação, este
documento diz se ele é confiável.

**Estado auditado:** commit `10704b8` ("Feat | Entrega final CV"), Fases 0–2 do plano
anterior. Três tensores por-Gaussiana criados e canalizados até as assinaturas CUDA;
nenhum kernel os lia.

**Critério de severidade.** O que classifica um bug como crítico aqui não é ele quebrar o
programa — é ele **não** quebrar. Os três críticos abaixo produzem formas corretas,
nenhuma exceção e resultado errado.

---

## A-1 · `prune_points` desconectava o material do otimizador — **crítico, silencioso**

**Local.** `scene/gaussian_model.py:678-680` (numeração original)

```python
self._specular_tint = nn.Parameter(self._specular_tint[valid_points_mask])   # ERRADO
```

**O que acontecia.** `_prune_optimizer(mask)` já itera **todos** os grupos do otimizador e
devolve os tensores podados em `optimizable_tensors`. As três linhas acima descartavam esse
resultado e criavam um `nn.Parameter` **diferente** a partir do tensor antigo. A partir daí:

```
self._specular_tint  ─────────────►  tensor A   (usado no forward, recebe .grad)
optimizer.param_groups[6]["params"][0] ──►  tensor B   (atualizado pelo Adam)
```

O Adam atualizava B; o modelo lia A. As formas continuavam corretas, então **nenhuma
exceção**. Os parâmetros de material simplesmente congelavam.

**Alcance.** `densify_and_prune` → `prune_points` roda a cada `densification_interval=100`
a partir de `densify_from_iter=500`, ou seja **a partir da iteração 600, em todo treino**.

**Comprovação.** Reintroduzindo a linha original e medindo o deslocamento de
`_specular_tint` após um passo do Adam:

| Versão | Deslocamento após 1 passo |
|---|---|
| Código original | `0.000e+00` — congelado |
| Corrigido | `5.000e-03` — igual à learning rate, como esperado |

**Correção.** Usar `optimizable_tensors[...]`, como as seis linhas acima já faziam para os
parâmetros originais. Além disso, `_assert_optimizer_binding()` passou a rodar após toda
densificação e poda (ADR-009).

**Impacto em resultados anteriores.** Nenhum treino anterior é invalidado por A-1, porque
nenhum kernel lia esses tensores — eram peso morto. O baseline
`output/fase2_linux_validacao` continua válido **como baseline do GOF puro**, e é assim
que deve ser citado, nunca como "GOF + material".

---

## A-2 · `capture`/`restore` não incluíam o material — **crítico**

**Local.** `scene/gaussian_model.py:189-221`

Retomar de checkpoint (`--start_checkpoint`) restaurava um modelo com
`_specular_tint = torch.empty(0)` e então chamava `training_setup`, que montava o Adam
sobre parâmetros de tamanho zero.

**Correção.** `capture` inclui um dicionário derivado de `MATERIAL_PARAMS`; `restore` o
reconstrói antes de `training_setup`. Teste T5.

---

## A-3 · `save_ply`/`load_ply` não persistiam o material — **crítico, o mais perigoso**

**Local.** `scene/gaussian_model.py:468-502` e `580-624`

Todo o pipeline a jusante recarrega o modelo por `load_ply`: `render.py`,
`extract_mesh.py`, `metrics.py`, `create_fused_ply.py`, `scene/__init__.py`. Com o BRDF
ativo, as imagens de **avaliação** seriam renderizadas com material default — diferentes
das de treino — e nada no log denunciaria. É o bug com maior potencial de corromper os
números do artigo sem deixar rastro.

**Correção.** Atributos de material entram em `construct_list_of_attributes`, `save_ply` e
`load_ply`, este último **com fallback**: PLYs antigos (sem os atributos) continuam
carregando com os defaults, para não invalidar o checkpoint de 30k já existente.
`save_fused_ply` mantém `include_material=False` de propósito — aquele formato existe para
visualizadores de 3DGS vanilla. Testes T4 e o de compatibilidade retroativa.

A mesma classe de erro foi antecipada e evitada para a **iluminação**: o envmap é salvo em
`lighting.pth` junto com o PLY, e `load_lighting` falha alto se a representação salva não
bater com a configurada. Teste T6.

---

## A-4 · Checkpoint não persistia iluminação nem embeddings de aparência — **crítico, silencioso** (achado 2026-09-01)

**Local.** `scene/gaussian_model.py`, `capture()`/`restore()`.

**O que acontecia.** A mesma classe de bug do A-2, não pega pela auditoria original
porque `self.lighting` (envmap/SH) e `self.appearance_network`/`_appearance_embeddings`
não existiam como conceito quando A-2 foi corrigido — só passaram a ser registrados no
otimizador (`training_setup`, grupo `"lighting"`) depois, na Fase 3. `capture()` salvava
`self.optimizer.state_dict()` (que só guarda o **momento** do Adam — `exp_avg`/`exp_avg_sq`
— não o valor dos parâmetros) e o dicionário de `MATERIAL_PARAMS`, mas nunca os valores de
`self.lighting` nem de `self.appearance_network`/`_appearance_embeddings`. Ao retomar de
`--start_checkpoint`:

1. `Scene.__init__` cria `self.lighting` **do zero**, com o ruído inicial (`init_std=0.02`)
   de sempre.
2. `restore()` recompunha o material corretamente (A-2 seguia correto), mas nunca tocava
   `self.lighting` — o objeto continuava sendo o recém-criado no passo 1.
3. `self.optimizer.load_state_dict(opt_dict)` reaplicava o **momento do Adam calculado
   para o envmap/SH treinado** sobre esse envmap/SH recém-inicializado — combinação
   inconsistente: gradientes acumulados para um material, aplicados a outro.

Nenhuma exceção, nenhuma forma incorreta — só o envmap/SH aprendido, silenciosamente
descartado a cada retomada. Idêntico em espírito a A-2, e com o mesmo alcance: quem
comparasse um run interrompido-e-retomado com um run contínuo atribuiria a diferença a
qualquer coisa menos à causa real.

**Por que isto não apareceu em nenhum run até agora.** Todos os runs executados
(B1, SMOKE) usaram o baseline **sem** `--brdf`, e `Scene` só cria `self.lighting` quando
`args.brdf` é verdadeiro (`scene/__init__.py`). O bug estava dormente — mas E1/E2 (as
próximas runs planejadas, `04_PROTOCOLO.md`) são exatamente as que ligam `--brdf`, rodam
30k iterações, e nesta máquina sem no-break: o cenário de maior probabilidade de precisar
de retomada é também o único em que o bug se manifesta.

**Correção.** `capture()` agora salva `{"class": ..., "state_dict": self.lighting.state_dict()}`
quando `self.lighting is not None`, e o estado de `_appearance_embeddings`/
`appearance_network`. `restore()` aplica isso a `self.lighting` **antes** de
`training_setup()` re-registrar seus parâmetros no Adam, e falha alto (`RuntimeError`) se
o checkpoint tem iluminação e o modelo atual não (ou vice-versa), ou se a classe de
iluminação não bate (`EnvironmentMap` vs `SHLighting`) — mesmo princípio de
`load_lighting` (A-3): um checkpoint de iluminação só faz sentido restaurado com os
mesmos flags que o geraram.

Cobertura nova em `tests/test_material_lifecycle.py`:
`test_t5_checkpoint_roundtrip_preserves_lighting`,
`test_restore_lighting_mismatch_fails_loud`,
`test_restore_lighting_class_mismatch_fails_loud`.

**Reforço de processo.** `scripts/run_experiment.sh` passou a injetar
`--checkpoint_iterations 10000 20000 30000` por default quando o chamador não especifica
o flag, em vez de depender de alguém lembrar — a mesma lição do incidente
EXP-20260811-01-b1 (queda de energia sem checkpoint, 7000 iterações perdidas), agora
estrutural em vez de só documentada.

---

## B-1 · Tensores de CPU cruzando a fronteira CUDA

`diff_gaussian_rasterization/__init__.py` criava `torch.Tensor([])` — um tensor de **CPU**
— quando os argumentos eram `None`, e `rasterize_points.cu` fazia
`.contiguous().data<float>()` neles, entregando um ponteiro de *host* a um kernel CUDA.
Inofensivo apenas porque nada desreferenciava; viraria *illegal memory access* no instante
em que a Fase 3 original o lesse.

**Correção.** Reversão completa da plumbagem C++/CUDA (ADR-006), em vez de blindá-la.

---

## B-2 · `_residual_color` era view-independent

O artigo define $c_r(\omega_o)$ como função da direção de vista; a implementação era um
RGB fixo por Gaussiana, que apenas duplicava o difuso e agravava a ambiguidade
difuso/especular. Removido; o papel passa para os graus ≥1 das SH (ADR-003).

---

## B-3 · Inicializações que atrapalham a otimização

| Parâmetro | Antes | Agora | Motivo |
|---|---|---|---|
| `specular_tint` | 0.50 | **0.05** | 0.5 põe 50 % de especular em *todas* as Gaussianas na iteração 0, competindo com o difuso antes de existir geometria |
| `roughness` | 0.50 | **0.70** | lóbulo mais largo no início evita highlights espúrios |
| envmap / SH | — | ruído `init_std=0.02` | evita o ponto degenerado onde $\partial L/\partial\rho = 0$ (ADR-008) |

Somado ao warm-up (`brdf_from_iter=3000`).

---

## B-4 · Nenhuma regularização

Sem `L_sparse`/`L_reg`, a separação difuso/especular — que é mal-posta — fica sem prior.
Corrigido na Fase 4.

---

## Menores

- **C-1** `view2gaussian_precomp` estava comentado na chamada de `rasterizer(...)` dentro
  de `render()`. Sem efeito com os flags default (`compute_view2gaussian_python=False`),
  mas descartaria em silêncio a matriz pré-computada se o flag fosse ligado.
  `integrate()`, no mesmo arquivo, sempre a passou — evidência interna de que passar é o
  comportamento pretendido. Restaurado.
- **C-2** `.data<float>()` deprecado em favor de `.data_ptr<float>()`. Tornou-se irrelevante
  com a reversão B-1.
- **C-3** `safe_state()` comentado em `train.py`. As seeds eram refeitas inline, então só
  se perdia o timestamp no stdout — que é justamente o que torna o log utilizável como
  registro de experimento. Reativado.

---

## Riscos de processo (migração para `CV_GOF_SHADER`)

- **D-1 · Instalações editáveis apontando para a árvore antiga.** `diff_gaussian_rasterization`,
  `simple_knn` e `tetra-nerf` resolviam para `~/Documentos/gaussian-opacity-fields/submodules/...`.
  Sem reinstalar, o Python continuaria importando o código antigo **sem nenhum aviso** —
  a causa nº 1 de "resultado fantasma". **Resolvido:** as três reinstaladas e verificadas.
- **D-2 · Sem `.so` compilado** em `CV_GOF_SHADER` (`*.so` está no `.gitignore`).
  Resolvido no mesmo passo.
- **D-3 · `.gitignore` descartava artefatos de experimento** (`output`, `*.json`, `*.png`,
  `*.ply`). Resolvido com exceção explícita para `docs/experiments/`.
- **D-4 · `tetra-triangulation`** existia com nomes diferentes nas duas árvores. Verificado
  como **conteúdo idêntico e vestigial**: `extract_mesh.py` importa `tetranerf`, que vem de
  `tetra-nerf-master`. Sem ação.
- **D-5 · Não-determinismo.** Os `atomicAdd` do rasterizador tornam o treino não
  bit-reprodutível mesmo com seeds fixas. Precisa constar do protocolo experimental:
  diferenças de PSNR abaixo de ~0,1 dB não são interpretáveis sem repetição.

---

## Achado durante a implementação

- **Rugosidade não-identificável sob luz uniforme.** Não é um bug herdado — foi descoberto
  ao escrever os testes da Fase 3. Ver ADR-008 e `03_FORMULACAO.md` §9. Vale registrar
  aqui porque revelou também um **teste fraco**: a versão do envmap "passava" apenas por
  ruído de ponto flutuante.
- **`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` não existe no torch 2.0.1** (foi
  adicionado no 2.1). Era uma das alavancas de VRAM previstas no plano; está indisponível
  neste ambiente e foi removida da lista.
