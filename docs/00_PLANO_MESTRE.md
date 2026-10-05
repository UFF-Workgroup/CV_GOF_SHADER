# Plano: GOF + Sombreamento Especular (GaussianShader) para Rocha Digital

> **Parcialmente superseded (2026-09-23, ADR-011 em `02_DECISOES.md`).** Este é o plano
> original — mantido por registro histórico. A partir de 2026-09-23 não há autorização de
> uso do dataset de rocha; tudo aqui que depende de B2/E2/cena de rocha (matriz de runs,
> marco de decisão, motivação) está fora do escopo ativo. Para o plano corrente, ver
> `04_PROTOCOLO.md`, `09_RESUMO_EXECUTIVO.md` e `02_DECISOES.md` ADR-011.

## Contexto

**Objetivo da pesquisa.** Integrar o sombreamento fisicamente baseado do *GaussianShader* na
arquitetura volumétrica do *Gaussian Opacity Fields* (GOF), para reconstruir gêmeos digitais de
testemunhos de rocha onde tanto a topologia quanto a resposta à luz importam. O GOF sozinho usa
apenas Harmônicas Esféricas (SH) para interpolar cor, o que falha em reflexos agudos; o
GaussianShader sozinho não tem a extração de malha por tetraedros nem o campo de opacidade do GOF.

**Estado atual.** As Fases 0–2 do plano anterior (`docs/GUIA_DA_APLICACAO.md`) foram executadas:
três tensores por-Gaussiana (`_specular_tint`, `_roughness`, `_residual_color`) foram criados em
`scene/gaussian_model.py` e canalizados por Python → PyBind11 → C++ → assinaturas CUDA. O `.so`
está compilado com as novas assinaturas e um treino completo de 30k rodou em Linux
(`output/fase2_linux_validacao`, cena Truck, `-r 2 --sh_degree 0 --eval`). **Nenhum dos três
tensores é lido por qualquer kernel**: hoje são peso morto.

**O que este plano muda.** A auditoria (§1) encontrou três bugs críticos e silenciosos, e a
revisão arquitetural (§2) mostrou que o caminho `colors_precomp` do rasterizador do GOF **já é
totalmente diferenciável**, o que torna as Fases 3 e 4 do plano anterior (BRDF em `forward.cu` +
derivadas manuais em `backward.cu`) desnecessárias. Isso remove semanas de trabalho de alto risco.

**Decisões travadas com o pesquisador:**

| # | Decisão | Escolha |
|---|---------|---------|
| 1 | Onde implementar o BRDF | **PyTorch via `colors_precomp`** — autograd, zero CUDA novo |
| 2 | Captura | **Mesa giratória, luz fixa na sala** (objeto gira, câmera/luz paradas) |
| 3 | Métricas do artigo | **NVS (PSNR/SSIM/LPIPS) + geometria da malha + relighting** |
| 4 | Repositório | **Trabalhar em `~/Documentos/CV_GOF_SHADER`** (git, GitHub UFF-Workgroup) |

**Hardware.** RTX 3050 **6 GB**, driver 595.71.05, CUDA 11.8, torch 2.0.1+cu118, Python 3.9,
env conda `gof`. Os 6 GB são a restrição dominante e moldam várias decisões abaixo.

---

## 1. Auditoria do código atual

### 1.1 Bugs críticos (falham em silêncio — quebram o artigo, não o programa)

**A-1 · `prune_points` desconecta os novos tensores do otimizador**
`scene/gaussian_model.py:678-680`

```python
self._specular_tint = nn.Parameter(self._specular_tint[valid_points_mask])   # ERRADO
```

`_prune_optimizer(mask)` (linha 643) já itera **todos** os grupos — incluindo `specular_tint`,
`roughness`, `residual_color` — e devolve tensores novos já podados em `optimizable_tensors`.
As três linhas acima descartam esse resultado e criam um `nn.Parameter` **diferente** a partir do
tensor antigo. A partir daí `self._specular_tint` e `optimizer.param_groups[6]["params"][0]` são
objetos distintos: o Adam passa a atualizar um tensor órfão que ninguém lê, e o parâmetro que o
modelo realmente usa nunca mais é atualizado. As formas continuam corretas, então **não há
exceção** — os parâmetros simplesmente congelam.

Dispara em `densify_and_prune` → `prune_points` (linha 813), ou seja **a partir da iteração 600**
(`densify_from_iter=500`, `densification_interval=100`), em todo treino.

*Correção:* usar `optimizable_tensors[...]`, exatamente como as seis linhas acima fazem.

**A-2 · `capture()` / `restore()` não incluem os novos tensores** — `gaussian_model.py:189-221`
Retomar de checkpoint (`--start_checkpoint`) restaura um modelo com `_specular_tint =
torch.empty(0)` e então chama `training_setup`, que monta o Adam sobre parâmetros vazios.
Retomada de treino está quebrada.

**A-3 · `save_ply()` / `load_ply()` não persistem os novos tensores**
`gaussian_model.py:468-502` e `580-624`
Todo o pipeline a jusante (`render.py`, `extract_mesh.py`, `metrics.py`, `create_fused_ply.py`,
`scene/__init__.py:81`) recarrega o modelo por `load_ply`. Assim que o BRDF estiver ativo, as
imagens de avaliação serão renderizadas com material default — **diferentes das imagens de
treino** — e ninguém perceberá pelo log. É o bug mais perigoso para a validade dos números.

### 1.2 Problemas de projeto (corretos em código, errados em física/estatística)

**B-1 · Tensores CPU vazios cruzando a fronteira CUDA.**
`diff_gaussian_rasterization/__init__.py:236-240` cria `torch.Tensor([])` (**CPU**) quando os
argumentos são `None`, e `rasterize_points.cu:111-113` faz `.contiguous().data<float>()` nesses
tensores, entregando um ponteiro de *host* a um kernel CUDA. Hoje é inofensivo porque nada o
desreferencia; viraria *illegal memory access* no instante em que a Fase 3 original o lesse.
Como o sombreamento migra para PyTorch, a correção certa é **reverter toda a plumbagem C++/CUDA**
(§3.1) em vez de blindá-la.

**B-2 · `_residual_color` é view-independent.** O artigo define $c_r(\omega_o)$ — uma função da
direção de vista. A implementação atual é um RGB fixo por Gaussiana, que apenas duplica o termo
difuso e agrava a ambiguidade difuso/especular. Correção em §2.3.

**B-3 · Inicializações ruins.** `specular_tint = 0.5` (sigmoid⁻¹) põe **50 % de especular em
todas as Gaussianas na iteração 0**, competindo com o difuso antes de a geometria existir.
`roughness = 0.5` produz lóbulos já estreitos, favorecendo highlights espúrios.

**B-4 · Nenhuma regularização.** A separação difuso/especular a partir de RGB é mal-posta. Sem
`L_sparse` / `L_reg`, o otimizador coloca o brilho no albedo (ou no resíduo) e o "material"
aprendido não tem significado físico — nem para relighting, nem para petrofísica.

### 1.3 Menores

- **C-1** `gaussian_renderer/__init__.py:111` — `view2gaussian_precomp` está comentado na chamada
  de `rasterizer(...)` em `render()`. Sem efeito com os flags default
  (`compute_view2gaussian_python=False`), mas descarta em silêncio a matriz pré-computada se o
  flag for ligado. Restaurar.
- **C-2** `.data<float>()` está deprecado em favor de `.data_ptr<float>()` (irrelevante se B-1 for
  revertido).
- **C-3** `train.py:365` — `safe_state()` comentado; as seeds são refeitas inline (367-370), então
  só se perde o timestamp no stdout. Reativar ajuda o registro de experimentos.

### 1.4 Riscos de processo (na migração para `CV_GOF_SHADER`)

- **D-1 · As três extensões estão instaladas em modo editável apontando para a árvore ANTIGA:**
  `diff_gaussian_rasterization`, `simple_knn` e `tetra-nerf` resolvem para
  `/home/geomesh/Documentos/gaussian-opacity-fields/submodules/...`. Sem reinstalar a partir de
  `CV_GOF_SHADER`, o Python continuará importando o código antigo **sem nenhum aviso**. Causa
  garantida de horas perdidas.
- **D-2 · `CV_GOF_SHADER` não tem `.so` compilado** (`*.so` está no `.gitignore`). Rebuild
  obrigatório.
- **D-3 · O `.gitignore` ignora `output`, `*.json`, `*.png`, `*.ply`** → os artefatos de
  experimento não seriam versionados. Precisa de exceções para `docs/experiments/`.
- **D-4 · `submodules/tetra-triangulation2` só existe na árvore de trabalho**; `CV_GOF_SHADER` tem
  `tetra-triangulation_OLD`. (Baixo impacto: `extract_mesh.py:13` importa `tetranerf`, que vem de
  `tetra-nerf-master`, presente nas duas.)
- **D-5 · Não-determinismo.** Os `atomicAdd` do rasterizador tornam o treino não bit-reprodutível
  mesmo com seeds fixas. Precisa ser declarado no protocolo experimental (§6).

---

## 2. Arquitetura escolhida

### 2.1 Por que PyTorch e não CUDA

O rasterizador do GOF aceita `colors_precomp` — uma cor RGB por Gaussiana, calculada fora — e o
backward já devolve o gradiente correspondente:

- `cuda_rasterizer/backward.cu:836` — `atomicAdd(&dL_dcolors[global_id*C+ch], ...)`
- `rasterize_points.cu:219` — devolve `dL_dcolors`
- `diff_gaussian_rasterization/__init__.py:175` — entrega como `grad_colors_precomp` ao autograd

Ou seja: **qualquer função diferenciável em PyTorch que produza `[N,3]` recebe gradiente correto
de graça.** Todo o BRDF (normal, reflexão, envmap, GGX, Fresnel, tint, rugosidade, SH) vira código
PyTorch comum. Consequências:

- Fases 3 e 4 do plano anterior deixam de existir. Zero derivada manual, zero risco de gradiente
  silenciosamente errado — que é a falha mais comum e mais cara nesse tipo de trabalho.
- O ciclo de iteração cai de "recompilar CUDA" (minutos) para "salvar arquivo" (segundos).
- O caminho `integrate` (usado por `extract_mesh.py`) aceita `colors_precomp` pela mesma via, então
  a extração de malha herda o sombreamento sem trabalho extra.

**Custo honesto:** o sombreamento passa a ser **por-Gaussiana** (uma cor por splat por vista), não
por-pixel. A nitidez do highlight fica limitada pela densidade de Gaussianas. É exatamente o que o
GaussianShader oficial faz, e funciona; mas é uma limitação real que o artigo deve declarar. A
variante por-pixel em CUDA fica registrada como trabalho futuro (§8).

### 2.2 Envmap no referencial da câmera — a adaptação para mesa giratória

Esta é a decisão física mais importante do plano, e é uma **contribuição metodológica publicável**.

Numa mesa giratória, o objeto gira e a luz fica parada na sala. No referencial do objeto — que é o
referencial que o COLMAP reconstrói — **a iluminação gira junto com a câmera**. Um environment map
ancorado no mundo (o que GaussianShader e todo o resto da literatura assume) é, portanto,
fisicamente inválido aqui: ele tentaria explicar um reflexo que se move como se fosse estático.

Derivação. Sejam $O$ o referencial do objeto (mundo do COLMAP), $R$ o da sala, $R_i$ a rotação da
mesa na imagem $i$, e $C$ a rotação sala→câmera (constante, pois a câmera está fixa). O COLMAP
entrega $W_i = C\,R_i$ (mundo→câmera). Uma direção $r_O$ em coordenadas do objeto vista da sala é

$$r_R = R_i\,r_O = C^{-1} W_i\, r_O = C^{-1} r_{\text{view}}$$

Logo $L(r_R) = L(C^{-1} r_{\text{view}}) =: L'(r_{\text{view}})$. Como $C$ é uma rotação global
constante, $L'$ é simplesmente $L$ pré-rotacionado — e é **exatamente o que a otimização aprende**
se indexarmos o envmap pela direção de reflexão em **espaço de vista**.

> **Não é preciso estimar o eixo da mesa nem o ângulo por imagem.** Amostrar o envmap com a direção
> de reflexão em espaço de vista é *exato* para esse setup, a menos de uma rotação global absorvida
> pelo próprio mapa aprendido.

Duas consequências práticas:

- Implementa-se como um flag `--light_frame {world,view}`. `world` reproduz o GaussianShader
  padrão e serve de **controle** (válido para Truck: objeto parado, câmera orbitando, sol fixo).
  `view` é o modo correto para os testemunhos.
- **Bônus de consistência:** as normais renderizadas pelo GOF (canais 3:6) já saem em espaço de
  vista (`forward.cu:504-549`), então as perdas de consistência de normal ficam no mesmo
  referencial, sem transformação.
- *Suposição declarada:* uma única estação de câmera. Com $K$ alturas/anéis de câmera, $C$ muda por
  estação; mitigação prevista é uma rotação aprendível por estação (`--num_light_stations K`).
- *Para relighting:* o mapa aprendido vive no referencial da sala. A demo de relighting substitui
  esse mapa; para renderizar sob luz "de mundo" basta fixar um ângulo de mesa e compor $C$.

### 2.3 As SH aparecem em dois lugares (o "extra" pedido)

**(a) `c_r(ω_o)` = SH de graus 1..3, reusando `_features_rest`.**
O termo residual do artigo é view-dependent. A base SH que o 3DGS já carrega é exatamente isso.
Então:

| Termo do artigo | Mapeamento no código | Situação |
|---|---|---|
| $c_d$ (difuso) | `_features_dc` via `SH2RGB` | já existe, já inicializa da nuvem SfM |
| $c_r(\omega_o)$ (residual) | `_features_rest` via `eval_sh` (graus ≥ 1) | já existe, `utils/sh_utils.py:57` |
| $s$ (tom especular) | `_specular_tint` | existe |
| $\rho$ (rugosidade) | `_roughness` | existe |
| $\Delta n$ (resíduo de normal) | **novo**, opcional | a criar |

`_residual_color` é **removido** — é redundante, fisicamente errado (B-2) e custa 3 floats +
3 estados de Adam por Gaussiana, o que importa em 6 GB.

**Propriedade valiosa:** com $s = 0$, $\Delta n = 0$ e `sh_degree=0`, a equação de sombreamento
colapsa em `clamp(SH2RGB(f_dc), 0, 1)` — **exatamente** o que o caminho
`pipe.convert_SHs_python=True` já calcula hoje (`gaussian_renderer/__init__.py:87-95`). O modelo
se reduz ao baseline do GOF de forma exata e testável (teste T1, §5).

**(b) SH como base de iluminação — ablação `--light_repr {envmap,sh}`.**
Além do envmap pré-filtrado, implementar iluminação em SH com convolução por lóbulo dependente da
rugosidade:

$$L_s(r,\rho) = \sum_{l=0}^{L}\; a_l(\rho) \sum_{m=-l}^{l} c_{lm}\, Y_{lm}(r),
\qquad a_l(\rho) = e^{-\,l(l+1)\,\rho^2/2}$$

É barato (27–48 parâmetros no total), diferenciável, e produz uma ablação cientificamente limpa:
*quanto de alta frequência a iluminação de um testemunho de rocha realmente exige?* Reusa
`eval_sh` de `utils/sh_utils.py`.

### 2.4 Equação de sombreamento final

$$c(\omega_o) \;=\; \gamma\Big( \underbrace{c_d}_{\text{albedo}} \;+\; \underbrace{F(\,n\!\cdot\!\omega_o,\, s)\odot L_s(r,\rho)}_{\text{especular}} \;+\; \underbrace{c_r(\omega_o)}_{\text{residual SH}} \Big)$$

com, **tudo em espaço de vista** (`R_{w2v} = cam.world_view_transform.T[:3,:3]`, padrão já usado em
`train.py:177-179`):

- $n$: eixo mais curto do elipsoide, $n = \text{normalize}(R[:,k] + \Delta n)$, $k=\arg\min$ das
  escalas, sinal escolhido para $n\!\cdot\!\omega_o > 0$.
  *(Nota: usar `get_scaling` cru para o `argmin`; o filtro 3D é monótono em $s$, logo preserva o
  argmin, mas altera a razão de anisotropia — vale documentar.)*
- $\omega_o$: direção do ponto para a câmera.
- $r = 2(n\!\cdot\!\omega_o)\,n - \omega_o$.
- $F$: Fresnel de Schlick, $F = s + (1-s)(1 - n\!\cdot\!\omega_o)^5$ — uma linha, fisicamente
  fundamentada, e o brilho rasante é muito visível em testemunho. Ablação `--use_fresnel`.
- $L_s$: envmap equirretangular HDR $128\times256$ com pirâmide de mips por `avg_pool` sucessivo
  (diferenciável, sem parâmetros extras); nível $= \text{clamp}(\sqrt{\rho}\,(L{-}1),0,L{-}1)$ com
  mistura trilinear. Amostragem por `F.grid_sample`. **Sem dependências externas.**
- $\gamma$: tone mapping. **Identidade no Marco 1** (para preservar a equivalência exata com o
  baseline e isolar o efeito do especular); linear→sRGB com albedo em espaço linear no Marco 2,
  quando o relighting HDR entra. Ablação `--use_tonemap`.

**Inicializações revisadas:** $s = \text{sigmoid}^{-1}(0.05)$ (quase sem especular),
$\rho = \text{sigmoid}^{-1}(0.7)$ (lóbulo largo → sem highlight espúrio), $\Delta n = 0$,
envmap = constante cinza. O ramo especular liga só a partir de `--brdf_from_iter` (default 3000),
depois de a geometria assentar.

---

## 3. Plano de implementação

### Fase 0 — Migração e linha de base auditável

1. Copiar `docs/` e `submodules/tetra-triangulation2` para `CV_GOF_SHADER`; passar a trabalhar lá.
2. Ajustar `.gitignore` (D-3): exceções `!docs/experiments/**` para `*.json`/`*.png`.
3. **Reinstalar as três extensões a partir da nova árvore** (D-1) e conferir que
   `diff_gaussian_rasterization.__file__` aponta para `CV_GOF_SHADER`. **Este passo não é opcional
   e é a causa nº 1 de resultados fantasma.**
4. Commit `baseline: estado auditado (fases 0-2)` — congela o ponto de partida.
5. Rodar o baseline B0 (§6) e arquivar as métricas.

### Fase 1 — Correções da auditoria (`scene/gaussian_model.py`)

Corrigir A-1, A-2, A-3, C-1, C-3 e remover `_residual_color` (B-2). Padrão a seguir — o próprio
arquivo já o usa corretamente para os seis parâmetros originais:

| Correção | Local | Ação |
|---|---|---|
| A-1 | `prune_points:678-680` | `self._X = optimizable_tensors["X"]` |
| A-2 | `capture`/`restore:189-221` | incluir `_specular_tint`, `_roughness`, `_normal_residual` |
| A-3 | `construct_list_of_attributes:468` + `save_ply`/`load_ply` | novos atributos, **com fallback**: se ausentes no PLY, inicializar no default (compatibilidade com checkpoints antigos) |
| B-2 | vários | remover `_residual_color` do modelo, do otimizador, da densificação e de `arguments/__init__.py:92` |
| B-3 | `create_from_pcd:417-419` | novas inicializações (§2.4) |
| C-1 | `gaussian_renderer/__init__.py:111` | descomentar `view2gaussian_precomp` |

**Invariante permanente a instituir:** um assert (ou teste T3) de que, após cada
densificação/poda, `id(self._X) == id(optimizer.param_groups[k]["params"][0])` para todo parâmetro.
É a única defesa estrutural contra a classe de bug A-1.

### Fase 2 — Reverter a plumbagem C++/CUDA

Remover `specular_tint`/`roughness`/`residual_color` das assinaturas de
`rasterize_points.{h,cu}`, `cuda_rasterizer/rasterizer.h`, `rasterizer_impl.cu`, `forward.{h,cu}` e
do wrapper `diff_gaussian_rasterization/__init__.py`. Recompilar.

Justificativa: é código morto que carrega risco de ABI (B-1), obriga a recompilar a cada mexida e
confunde revisores. O rasterizador volta a ser **idêntico ao GOF upstream** — o que fortalece o
artigo: *"nosso método não modifica o rasterizador"*.

Critério de aceite: `_C.rasterize_gaussians.__doc__` volta a 22 argumentos e o baseline reproduz as
métricas da Fase 0.

### Fase 3 — Módulo de sombreamento (o núcleo)

**Arquivos novos:**

- `scene/brdf.py` — funções puras e testáveis: `shortest_axis_normal`, `orient_towards_camera`,
  `reflect`, `schlick_fresnel`, `tonemap`.
- `scene/lighting.py` — `EnvironmentMap` (equirretangular + pirâmide de mips + `grid_sample`) e
  `SHLighting` (§2.3b), atrás de uma interface comum `sample(dirs, roughness)`.

**Alterações:**

- `scene/gaussian_model.py`: `_normal_residual` (opcional, `--use_normal_residual`); método
  `get_shading_colors(camera, light, opt)` que devolve `[N,3]`; registrar o envmap/SH no otimizador
  (LR própria, `--envmap_lr`).
- `gaussian_renderer/__init__.py`: em `render()` e `integrate()`, quando o BRDF está ativo,
  `colors_precomp = pc.get_shading_colors(...)` e **remover** os três argumentos antigos do
  rasterizador. Adicionar `--brdf` para alternar entre baseline e BRDF.
- `arguments/__init__.py`: `brdf`, `brdf_from_iter`, `light_frame`, `light_repr`, `use_fresnel`,
  `use_tonemap`, `use_normal_residual`, `envmap_lr`, `envmap_res`, `lambda_specular_sparse`,
  `lambda_normal_residual`, `lambda_shading_normal`, `num_light_stations`.

O envmap é salvo/carregado junto com o modelo (`scene/__init__.py:88`) — sem isso, relighting e
avaliação ficam inconsistentes (mesma classe de erro que A-3).

### Fase 4 — Perdas e regularização (`train.py`, `utils/loss_utils.py`)

$$\mathcal{L} = \underbrace{\mathcal{L}_{rgb} + \lambda_{d}\mathcal{L}_{dist} + \lambda_{dn}\mathcal{L}_{depth\text{-}normal}}_{\text{já existe no GOF}} + \lambda_{s}\mathcal{L}_{sparse} + \lambda_{r}\mathcal{L}_{reg} \;(+\, \lambda_{sn}\mathcal{L}_{shading\text{-}normal})$$

- $\mathcal{L}_{sparse}$: L1 sobre $s$ — a maioria das Gaussianas deve ser não-especular
  ($\lambda_s = 0.001$).
- $\mathcal{L}_{reg} = \lVert\Delta n\rVert^2$ ($\lambda_r = 0.001$), só com `--use_normal_residual`.
- $\mathcal{L}_{shading\text{-}normal}$ (**opcional, default OFF**): consistência entre a normal de
  sombreamento por-Gaussiana e a normal por-raio do GOF, via uma segunda rasterização com
  `override_color = n_{view}`. Custa +1 forward/backward por iteração — pesado nos 6 GB. A hipótese
  de trabalho é que a `depth_normal_loss` que o GOF **já tem** (`train.py:169-185`, junto com a
  `distortion_loss` que achata as Gaussianas) já acopla implicitamente as duas definições de
  normal; a perda explícita entra como ablação para testar essa hipótese, não como default.

Nota conceitual para o artigo: o GOF define a normal **por raio, em espaço de vista**, como
gradiente do level-set na interseção (`forward.cu:504-549`); o GaussianShader define **por
Gaussiana, no mundo**, pelo eixo mais curto. São objetos diferentes. Explicitar essa distinção — e
medir a divergência entre elas — é uma contribuição própria deste trabalho.

### Fase 5 — Otimização para 6 GB

Orçamento medido para Truck `-r 2` (~2,0 M Gaussianas, do PLY de 144,3 MB / 18 floats por ponto):

| Configuração | Params/Gauss. | VRAM (param+grad+2 Adam) |
|---|---|---|
| GOF baseline, `sh_degree=0` | 14 | ~448 MB |
| + $s,\rho,\Delta n$ | +7 | ~672 MB |
| + `sh_degree=3` para $c_r$ | +45 | **~2,1 GB** ← inviável |
| + `sh_degree=1` para $c_r$ | +9 | ~816 MB |

**Alavancas, em ordem de custo-benefício:**

1. **`sh_degree ≤ 1` como default do modelo BRDF.** Não é só engenharia: com o especular explícito,
   a necessidade de SH de alta ordem cai — e isso é uma **hipótese testável** (ablação E3), não uma
   concessão. Provar que $s,\rho$ + SH grau 1 batem SH grau 3 é resultado de artigo.
2. `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` contra fragmentação.
3. `--lambda_shading_normal 0` (evita a segunda rasterização).
4. `torch.utils.checkpoint` no `get_shading_colors` (~10 tensores `[N,3]` intermediários ≈ 240 MB)
   se faltar folga; troca memória por ~15 % de tempo.
5. Poda mais agressiva / teto de Gaussianas para as cenas de rocha.
6. `--resolution 4` como fallback se as fotos dos testemunhos forem de alta resolução.

Instrumentar `torch.cuda.max_memory_allocated()` no `training_report` e registrar em toda run.

---

## 4. Arquivos afetados

| Arquivo | Natureza |
|---|---|
| `scene/gaussian_model.py` | correções A-1/A-2/A-3, remoção de `_residual_color`, `_normal_residual`, `get_shading_colors` |
| `scene/brdf.py`, `scene/lighting.py` | **novos** |
| `gaussian_renderer/__init__.py` | `colors_precomp` = sombreamento; limpeza dos args antigos |
| `arguments/__init__.py` | novos flags/LRs |
| `train.py` | perdas novas, warm-up do BRDF, log de VRAM, `safe_state` |
| `utils/loss_utils.py` | `L_sparse`, `L_reg` |
| `scene/__init__.py` | persistir a iluminação |
| `submodules/diff-gaussian-rasterization/**` | **reversão** para upstream |
| `docs/**` | §7 |

---

## 5. Testes de validação (executáveis, `tests/`)

| ID | Teste | Critério |
|---|---|---|
| **T1** | **Redução exata ao baseline**: `--brdf` com $s{=}0,\Delta n{=}0$, `sh_degree=0` vs. caminho `--convert_SHs_python` | `max|Δpixel| < 1e-5` |
| **T2** | Fluxo de gradiente: 1 `backward` | `.grad` não-nulo em `_specular_tint`, `_roughness`, envmap |
| **T3** | **Invariante do otimizador** após 700 its (pós-densificação/poda) | `id(model._X) == id(optim.param_groups[k]['params'][0])` ∀X — pega A-1 |
| **T4** | Round-trip PLY: `save_ply` → `load_ply` → render | imagens idênticas — pega A-3 |
| **T5** | Round-trip checkpoint: `capture` → `restore` → 1 iteração | perda idêntica — pega A-2 |
| **T6** | Round-trip do envmap (salvar/carregar iluminação) | idêntico |
| **T7** | Especular sintético: esfera com highlight conhecido | $s,\rho$ recuperados dentro de tolerância |
| **T8** | `extract_mesh.py` roda no modelo BRDF | malha gerada sem erro |
| **T9** | Regressão de VRAM/tempo | pico < 5,5 GB; tempo/it dentro de +25 % do baseline |

T1 e T3 são os mais valiosos: T1 prova que a nova via de cor é equivalente à antiga, e T3 é o
guarda permanente contra a classe de bug que já ocorreu.

---

## 6. Protocolo experimental

**Cenas.** Truck (Tanks&Temples, `-r 2 --eval`) como cena de controle e comparação com a
literatura — tem GT de malha e já existe uma baseline treinada. Testemunhos de rocha como cena
alvo, com o modo `--light_frame view`.

**Matriz de runs:**

| ID | Configuração | Pergunta que responde |
|---|---|---|
| B0 | GOF upstream, sem alterações | número de referência da literatura |
| B1 | Árvore atual pós-Fase 2 | a reversão foi neutra? |
| E1 | BRDF, `light_frame=world`, SH 0 | o especular ajuda em cena de mundo fixo (Truck)? |
| E2 | BRDF, `light_frame=view`, SH 0 | **contribuição principal** — ganho no setup de mesa giratória |
| E3 | E2 × `sh_degree ∈ {0,1,2,3}` | quanto de $c_r$ é preciso com especular explícito? (§5 da otimização) |
| E4 | E2 com `light_repr=sh` | envmap pré-filtrado vs. iluminação SH |
| E5 | E2 sem Fresnel | o brilho rasante importa em rocha? |
| E6 | E2 sem $\Delta n$ | o resíduo de normal se paga? |
| E7 | E2 + `L_shading_normal` | a `depth_normal_loss` do GOF já basta? |

**Métricas.** PSNR/SSIM/LPIPS (`metrics.py`), malha (Chamfer/F1 via `eval_tnt/`), nº de Gaussianas,
pico de VRAM, tempo/iteração, tempo total. Relighting: renders qualitativos com envmap substituído.

**Rigor.** Toda run registra `git rev-parse HEAD`, comando completo, `cfg_args`, hardware e versões.
**Declarar explicitamente** que o rasterizador usa `atomicAdd` e portanto o treino **não é
bit-reprodutível** mesmo com seeds fixas (D-5); rodar os números principais (B0, E2) com ≥ 2 seeds e
reportar média ± desvio. Sem isso, diferenças de PSNR abaixo de ~0,1 dB não são interpretáveis.

**Marco de decisão.** Se E2 não superar B1 em NVS **e** em geometria, parar e diagnosticar antes de
continuar as ablações — a hipótese de que o especular ajuda pode não valer para rocha fosca, e isso
também é um resultado publicável (com evidência, não com abandono).

---

## 7. Plano de documentação científica

Estrutura em `docs/`, escrita em paralelo ao código (não depois), didática o suficiente para ser
lida por um coautor que não conhece o repositório:

| Arquivo | Conteúdo | Quando |
|---|---|---|
| `00_VISAO_GERAL.md` | Problema, contribuições, roadmap, estado atual. Substitui `GUIA_DA_APLICACAO.md` (que é arquivado, não apagado) | Fase 0 |
| `01_FUNDAMENTOS.md` | Didático: 3DGS → GOF (campo de opacidade, normal por raio, marching tetrahedra) → GaussianShader (BRDF, split-sum) → a fusão. Com intuição antes de cada equação | Fases 0–3 |
| `02_DECISOES.md` | **Log de ADRs** (append-only, numerado): contexto / opções / decisão / consequências. ADR-001 sombreamento em PyTorch; ADR-002 envmap em espaço de vista; ADR-003 $c_r$ como SH; ADR-004 redução exata ao baseline; ADR-005 orçamento de SH em 6 GB; ADR-006 reversão do CUDA | contínuo |
| `03_FORMULACAO.md` | Notação, derivação completa, **a prova da mesa giratória (§2.2)**, diagrama do fluxo de gradiente, as duas definições de normal | Fase 3 |
| `04_PROTOCOLO.md` | Datasets, splits, seeds, hardware, comandos exatos, definição de cada métrica, matriz de ablações, política de não-determinismo | Fase 0, revisado na 5 |
| `05_EXPERIMENTOS.md` | Log corrido: uma linha por run — ID, data, commit, comando, diff de config, métricas, VRAM, tempo, observação | contínuo |
| `06_AUDITORIA.md` | Os achados da §1 e as correções, com rastreabilidade do que estava errado e por quê | Fase 1 |
| `07_LIMITACOES.md` | Sombreamento por-Gaussiana, suposição de estação única, ambiguidade difuso/especular, trabalho futuro | Fase 5 |
| `CHANGELOG.md` | Por fase | contínuo |
| `templates/` | Modelo de ADR e de relatório de experimento | Fase 0 |

**Convenções de rastreabilidade:**
- ID de run: `EXP-AAAAMMDD-NN-<slug>`; diretório `docs/experiments/<RUN_ID>/` com `cfg_args`,
  `commit.txt`, `metrics.json`, `env.txt` (versões) e 3 renders de amostra.
- `scripts/run_experiment.sh` carimba commit + config + ambiente no diretório de saída
  automaticamente — registro manual é registro que não acontece.
- Nenhum número entra em `05_EXPERIMENTOS.md` sem um `RUN_ID` com commit correspondente.

---

## 8. Riscos e trabalho futuro

| Risco | Mitigação |
|---|---|
| Ambiguidade difuso/especular (problema mal-posto) | warm-up do BRDF, $s$ inicial baixo, `L_sparse`, ablação E3 limitando $c_r$ |
| Rocha fosca → especular quase nulo | E1/E2 medem isso; resultado negativo com evidência ainda é publicável |
| 6 GB insuficientes na cena de rocha | alavancas §3 Fase 5, em ordem; `-r 4` como último recurso |
| Múltiplas estações de câmera quebram a suposição de §2.2 | rotação aprendível por estação (`--num_light_stations`) |
| Highlight limitado pela densidade de Gaussianas | declarado como limitação; variante CUDA por-pixel fica como trabalho futuro |

**Trabalho futuro registrado:** sombreamento por-pixel em CUDA acoplado à normal por raio do GOF;
estimativa explícita do eixo da mesa giratória; validação dos mapas de material contra medidas
petrofísicas.

---

## 9. Verificação de ponta a ponta

Sequência para validar tudo depois da implementação (env `gof`, dentro de `CV_GOF_SHADER`):

1. **Sanidade do ambiente** — confirmar que `diff_gaussian_rasterization.__file__` aponta para
   `CV_GOF_SHADER` (D-1) e que `_C.rasterize_gaussians.__doc__` tem 22 args (Fase 2).
2. **Testes unitários** — `pytest tests/` (T1–T8); T1 e T3 são bloqueantes.
3. **Baseline curto** — `train.py -s ~/Documentos/Truck -m output/EXP-...-b1 -r 2 --sh_degree 0
   --eval --iterations 7000`, comparar PSNR@7k com `output/fase2_linux_validacao`.
4. **BRDF curto** — mesma coisa com `--brdf --light_frame view --iterations 7000`; verificar que
   $s$ e $\rho$ saem dos valores iniciais (prova viva de que A-1 foi corrigido) e que o pico de
   VRAM cabe (T9).
5. **Run completa** — 30k, `render.py` + `metrics.py`, e `extract_mesh.py` (T8).
6. **Relighting** — renderizar a mesma vista com 2–3 envmaps distintos e arquivar em
   `docs/experiments/<RUN_ID>/relight/`.
7. **Registro** — preencher `05_EXPERIMENTOS.md` e commitar.

---

## Após aprovação

Além da implementação, na Fase 0 eu também: copio este plano para `docs/` no repositório
`CV_GOF_SHADER` (para consulta fora do Claude Code) e gravo na memória do projeto os fatos
não-deriváveis do código — setup de captura em mesa giratória, restrição de 6 GB, repositório
canônico, e a decisão arquitetural de sombrear em PyTorch.
