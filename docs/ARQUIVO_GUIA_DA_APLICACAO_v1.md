> ⚠️ **DOCUMENTO ARQUIVADO — SUPERSEDIDO EM 2026-08-10.**
> Mantido apenas como registro histórico das Fases 0–2. **Não seguir as Fases 3, 4 e 5 daqui.**
> O plano vigente é [`00_PLANO_MESTRE.md`](./00_PLANO_MESTRE.md), que diverge deste em três pontos:
> 1. O BRDF é implementado **em PyTorch via `colors_precomp`** (autograd), não em `forward.cu`.
>    As Fases 3 e 4 abaixo (matemática CUDA + derivadas manuais em `backward.cu`) foram
>    **canceladas** — o caminho `colors_precomp` do GOF já é diferenciável ponta a ponta.
> 2. A plumbagem C++/CUDA descrita na Fase 2 abaixo será **revertida** (vira código morto).
> 3. O environment map é ancorado no **espaço de vista**, não no mundo, porque a captura é em
>    mesa giratória (objeto gira, luz fixa na sala).

# Documentação de Integração: GaussianShader & Gaussian Opacity Fields (GOF)

## 1. O Objetivo
Integrar o modelo de sombreamento fisicamente baseado (BRDF) introduzido pelo *GaussianShader* na arquitetura volumétrica do *Gaussian Opacity Fields* (GOF). O objetivo prático é substituir a formulação original baseada puramente em Harmônicas Esféricas (SH) por uma aproximação da equação de renderização que suporta superfícies refletivas[cite: 3]. A junção das normais de alta fidelidade extraídas via triangulação tetraédrica do GOF com as propriedades de material do GaussianShader viabilizará a pesquisa acadêmica na reconstrução de gêmeos digitais com visão computacional de testemunhos de rochas digitais, onde a precisão topológica e a resposta à iluminação são críticas.

---

## 2. As Fontes
* **Repositório Base (Geometria):** `autonomousvision/gaussian-opacity-fields`[cite: 1].
* **Referência Matemática (Sombreamento):** `asparagus15/gaussianshader`[cite: 2] e artigo *"GaussianShader: 3D Gaussian Splatting with Shading Functions for Reflective Surfaces"*[cite: 3].

---

## 3. Status Atual
**Avanço:** Implementação das **Fases 0, 1 e 2** concluída na camada de código. 
**Pendente (Imediato):** Recompilação do submódulo `diff-gaussian-rasterization` em ambiente Windows via *x64 Native Tools Command Prompt for VS 2022* e execução do teste de sanidade (*baseline*) para garantir a integridade da ponte PyBind11.

---

## 4. O Problema Original
O 3DGS padrão e o GOF não modelam explicitamente propriedades de aparência (interação luz-superfície)[cite: 3]. Eles utilizam apenas SH para interpolação de cor, falhando na representação de reflexos agudos[cite: 3]. O desafio central é introduzir a Equação de Sombreamento simplificada:

$$c(\omega_o) = \gamma(c_d + s \odot L_s(\omega_o, n, \rho) + c_r(\omega_o))$$

Onde devem ser introduzidas as variáveis de material na otimização de cada splat: cor residual ($c_r$), rugosidade ($\rho$) e tom especular ($s$)[cite: 3]. Arquiteturalmente, o obstáculo é propagar esses novos tensores da VRAM gerenciada pelo Python/PyTorch de forma contígua através da interface C++ (PyBind11) até os *kernels* CUDA subjacentes do rasterizador e do integrador volumétrico (exclusivo do GOF), sem causar corrupção de memória ou *segmentation faults*.

---

## 5. A Solução Encontrada
Estruturar a modificação de fora para dentro. Primeiro, expandiu-se a classe `GaussianModel` no PyTorch para instanciar, densificar e otimizar os novos tensores físicos usando ativações apropriadas (`sigmoid` para limites de 0 a 1). Segundo, mapeou-se uma rota segura de passagem de argumentos no wrapper Python `diff_gaussian_rasterization/__init__.py`. Terceiro, corrigiu-se o desalinhamento de assinaturas nas funções C++ `IntegrateGaussiansToPointsCUDA` e na classe `Rasterizer`, garantindo que o motor backend do GOF receba exatamente o que o Python está a enviar.

---

## 6. Todo-List
- [x] **Fase 0:** Preparação do Ambiente Windows (CUDA 11.8 + MSVC) e Validação de Baseline (`-r 2 --sh_degree 0`).
- [x] **Fase 1:** Estruturas de Dados e Alocação de Memória no Python (`gaussian_model.py`).
- [x] **Fase 2:** Passagem Bidirecional de Dados (Pipelines Python $\rightarrow$ C++).
- [ ] **Fase 3:** O Forward Pass (Matemática do BRDF, iluminação e reflexão em `forward.cu` e `rasterizer_impl.cu`).
- [ ] **Fase 4:** O Backward Pass (Derivadas parciais analíticas e Regra da Cadeia em `backward.cu`).
- [ ] **Fase 5:** Funções de Perda (Losses) e Fine-Tuning (Suavidade espacial e restrições sobre o $c_r$) em `train.py`.

---

## 7. Passo a Passo Completo (Fases 1 e 2)

### FASE 1: Estruturas de Dados no Modelo Python

**Arquivo: `scene/gaussian_model.py`**
1. Instanciar tensores no `__init__` e adicionar propriedades:
```python
self._specular_tint = torch.empty(0)
self._roughness = torch.empty(0)
self._residual_color = torch.empty(0)

@property
def get_specular_tint(self):
    return torch.sigmoid(self._specular_tint)

@property
def get_roughness(self):
    return torch.sigmoid(self._roughness)

@property
def get_residual_color(self):
    return self._residual_color


2. Inicializar na GPU no `create_from_pcd`:

```python
tensor_specular = inverse_sigmoid(0.5 * torch.ones((fused_point_cloud.shape[0], 3), dtype=torch.float, device="cuda"))
tensor_roughness = inverse_sigmoid(0.5 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))
tensor_residual = torch.zeros((fused_point_cloud.shape[0], 3), dtype=torch.float, device="cuda")

self._specular_tint = nn.Parameter(tensor_specular.requires_grad_(True))
self._roughness = nn.Parameter(tensor_roughness.requires_grad_(True))
self._residual_color = nn.Parameter(tensor_residual.requires_grad_(True))

```

3. Adicionar aos parâmetros rastreados pelo Adam no `training_setup`:

```python
{'params': [self._specular_tint], 'lr': training_args.specular_tint_lr, "name": "specular_tint"},
{'params': [self._roughness], 'lr': training_args.roughness_lr, "name": "roughness"},
{'params': [self._residual_color], 'lr': training_args.residual_color_lr, "name": "residual_color"},

```

4. Tratar densificação e poda (split, clone, prune) no `densification_postfix` e relacionados:

```python
# Em densification_postfix
d = {...,
     "specular_tint": new_specular,
     "roughness": new_roughness,
     "residual_color": new_residual}

# Em densify_and_clone
new_specular = self._specular_tint[selected_pts_mask]
# ... [repetir para roughness e residual]

# Em densify_and_split
new_specular = self._specular_tint[selected_pts_mask].repeat(N,1)
# ... [repetir para roughness e residual]

# No _prune_optimizer
self._specular_tint = nn.Parameter(self._specular_tint[valid_points_mask])
# ... [repetir para roughness e residual]

```

**Arquivo: `arguments/__init__.py**`
5. Declarar as novas *learning rates* em `OptimizationParams`:

```python
self.specular_tint_lr = 0.005
self.roughness_lr = 0.005
self.residual_color_lr = 0.001

```

### FASE 2: Canalização PyTorch para C++/CUDA

**Arquivo: `gaussian_renderer/__init__.py**`

1. Na função `render(...)` extrair e enviar variáveis:

```python
specular_tint = pc.get_specular_tint
roughness = pc.get_roughness
residual_color = pc.get_residual_color

# Dentro de rasterizer(...) e rasterizer.integrate(...)
specular_tint = specular_tint,
roughness = roughness,
residual_color = residual_color,

```

**Arquivo: `submodules/diff-gaussian-rasterization/diff_gaussian_rasterization/__init__.py**`
2. Em `RasterizeGaussians.forward` (e no respectivo `.integrate` e `backward`), ajustar as chamadas C++:

```python
# Na tupla args para _C.rasterize_gaussians e _C.integrate_gaussians_to_points
cov3Ds_precomp,
specular_tint,
roughness,
residual_color,
view2gaussian_precomp,

# No ctx.save_for_backward
ctx.save_for_backward(..., cov3Ds_precomp, specular_tint, roughness, residual_color, view2gaussian_precomp, ...)

```

**Arquivo: `submodules/diff-gaussian-rasterization/rasterize_points.h**`
3. Alterar as assinaturas `RasterizeGaussiansCUDA` e `IntegrateGaussiansToPointsCUDA`:

```cpp
const torch::Tensor& cov3D_precomp,
const torch::Tensor& specular_tint,
const torch::Tensor& roughness,
const torch::Tensor& residual_color,
const torch::Tensor& view2gaussian_precomp,

```

**Arquivo: `submodules/diff-gaussian-rasterization/rasterize_points.cu**`
4. Replicar assinaturas e mapear o `.data<float>()` na invocação do rasterizador nativo (`CudaRasterizer::Rasterizer::integrate` e `forward`):

```cpp
cov3D_precomp.contiguous().data<float>(), 
specular_tint.contiguous().data<float>(),
roughness.contiguous().data<float>(),
residual_color.contiguous().data<float>(),
view2gaussian_precomp.contiguous().data<float>(), 

```

**Arquivos: `cuda_rasterizer/rasterizer.h` e `cuda_rasterizer/rasterizer_impl.cu**`
5. Em `Rasterizer::integrate` e `Rasterizer::forward`, injetar as constantes flutuantes:

```cpp
const float* cov3D_precomp,
const float* specular_tint,
const float* roughness,
const float* residual_color,
const float* view2gaussian_precomp,

```

### FASE 3: O Forward Pass (Matemática do BRDF em CUDA)

**Objetivo:** Substituir a simples interpolação de cor via Harmônicas Esféricas (SH) pela avaliação da Equação de Renderização simplificada[cite: 3]. 

**Modelo Mental (Física da Imagem):** No rasterizador tradicional, a cor do pixel é um somatório de opacidades. Na nossa nova arquitetura, antes de o splat contribuir para o pixel, a sua cor base (difusa) será somada ao produto do tom especular ($s$) com a luz incidente ($L_s$) amostrada na direção de reflexão ($r$) num *environment map*, e por fim ajustada pela cor residual ($c_r$)[cite: 3].

**Arquivos de Intervenção:** 
* `submodules/diff-gaussian-rasterization/cuda_rasterizer/forward.h`
* `submodules/diff-gaussian-rasterization/cuda_rasterizer/forward.cu`

1. **Injeção de Parâmetros:** A assinatura do kernel `render` e `integrate` precisará de receber os ponteiros para `specular_tint`, `roughness` e `residual_color`.
2. **Cálculo da Normal e Vetor de Reflexão:** 
   O GaussianShader define a normal $n$ a partir do eixo mais curto do elipsoide $v$, somado a um resíduo otimizável $\Delta n$[cite: 3]. Em `forward.cu`, é necessário extrair a rotação e escala do gaussiano, identificar o menor vetor e calcular a reflexão:
   $$r = 2(\omega_o \cdot n)n - \omega_o$$
3. **Amostragem de Luz e Composição da Cor:**
   Onde a cor era computada pelas SH (`computeColorFromSH`), a lógica deve ser substituída para implementar a Equação 3 do artigo:
   $$c(\omega_o) = \gamma(c_d + s \odot L_s(\omega_o, n, \rho) + c_r(\omega_o))$$[cite: 3].

---

### FASE 4: O Backward Pass (Diferenciação Analítica em CUDA)

**Objetivo:** Computar os gradientes exatos das novas variáveis em relação à *Loss* do pixel e devolvê-los ao PyTorch[cite: 3].

**Modelo Mental (Algoritmos e Otimização):** Em PyTorch, a diferenciação (Autograd) é automática. Em CUDA nativo, o encadeamento de derivadas (Chain Rule) exige código explícito. O gradiente da cor final do pixel ($\frac{\partial L}{\partial C}$) está disponível no *backward pass*. Temos de derivar essa cor em relação a cada componente física que inserimos na Fase 3. 

**Arquivos de Intervenção:** 
* `submodules/diff-gaussian-rasterization/cuda_rasterizer/backward.h`
* `submodules/diff-gaussian-rasterization/cuda_rasterizer/backward.cu`

1. **Estrutura de Gradientes:** Adicionar os buffers de gradiente `dL_dspecular`, `dL_droughness`, `dL_dresidual` nas assinaturas de `RasterizeGaussiansBackwardCUDA`.
2. **Derivação Manual no Kernel (`backward.cu`):**
   * $\frac{\partial L}{\partial c_r}$: Gradiente direto em relação à cor residual.
   * $\frac{\partial L}{\partial s}$: Derivada parcial em relação ao tom especular, que depende fortemente do resultado da amostragem do mapa de ambiente $L_s$[cite: 3].
   * $\frac{\partial L}{\partial \rho}$: A derivada mais complexa, pois a rugosidade afeta o NDF (Normal Distribution Function) de GGX[cite: 3]. O gradiente terá de ser propagado através da interpolação do nível de detalhe (LOD) do *environment map* ou do MLP de iluminação.
3. **Escrita na Memória:** Utilizar operações atômicas (`atomicAdd`) para acumular os gradientes de forma segura na memória global da GPU, evitando condições de corrida (race conditions) entre *threads*.

---

### FASE 5: Funções de Perda (Losses) e Restrições Físicas

**Objetivo:** Estruturar a otimização matemática em Python para impedir que o otimizador contorne a física real, evitando que "trapaceie" utilizando a cor residual para modelar especularidade[cite: 3].

**Modelo Mental (Problemas Inversos):** A separação de materiais (difuso vs. especular) a partir de imagens RGB é um problema mal posto (ill-posed). Sem restrições, o otimizador pode colocar a textura do brilho diretamente na cor difusa ($c_d$) ou no resíduo ($c_r$). As *losses* funcionam como regularizadores severos do solver.

**Arquivos de Intervenção:** 
* `train.py`
* `utils/loss_utils.py`

1. **Consistência Normal-Geometria ($\mathcal{L}_{normal}$):**
   O GaussianShader introduz uma penalidade para garantir que a normal prevista do Gaussiano ($\hat{n}$) corresponda à normal derivada do mapa de profundidade ($\overline{n}$)[cite: 3]:
   $$\mathcal{L}_{normal} = ||\overline{n} - \hat{n}||^2$$[cite: 3].
   *Implementação:* No ciclo de treino em `train.py`, calcular a diferença entre as normais renderizadas pelo GOF e as normais preditas pelas propriedades do material, aplicando a função de perda MSE[cite: 3].

2. **Regularização do Resíduo da Normal ($\mathcal{L}_{reg}$):**
   Para evitar que a normal divirja severamente da orientação real da malha, penaliza-se o resíduo da normal calculando a sua norma L2[cite: 3]:
   $$\mathcal{L}_{reg} = ||\Delta n||^2$$[cite: 3].

3. **Loss Total de Otimização:**
   Consolidar a formulação final da *Loss* no final da iteração em `train.py` somando os pesos definidos pelo artigo[cite: 3]:
   $$\mathcal{L} = \mathcal{L}_{color} + \lambda_{n}\mathcal{L}_{normal} + \lambda_{s}\mathcal{L}_{sparse} + \lambda_{r}\mathcal{L}_{reg}$$[cite: 3].
   *Implementação:* Onde $\lambda_n = 0.01$, $\lambda_s = 0.001$ e $\lambda_r = 0.001$, injetando estas constantes e executando o `loss.backward()`[cite: 3].