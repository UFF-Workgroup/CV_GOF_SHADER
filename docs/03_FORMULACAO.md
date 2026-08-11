# Formulação Matemática

Este documento é autocontido: define a notação, deriva a equação de sombreamento e prova
o resultado da mesa giratória. A intenção é que um coautor consiga verificar cada passo
sem ler o código.

---

## 1. Notação

| Símbolo | Significado | Onde vive no código |
|---|---|---|
| $\mathbf{x}_i$ | centro da Gaussiana $i$ | `_xyz` |
| $R_i, \mathbf{s}_i$ | rotação (quatérnion) e escalas | `_rotation`, `_scaling` |
| $\mathbf{n}_i$ | normal de sombreamento | derivada, §3 |
| $\Delta\mathbf{n}_i$ | resíduo aprendível de normal | `_normal_residual` |
| $\omega_o$ | direção da superfície **para** a câmera | derivada |
| $\mathbf{r}$ | direção de reflexão especular | derivada, §4 |
| $c_d$ | albedo difuso | `_features_dc` (SH grau 0) |
| $c_r(\omega_o)$ | cor residual dependente da vista | `_features_rest` (SH graus ≥1) |
| $\mathbf{s}$ | tom especular (= $F_0$) | `_specular_tint`, via sigmoide |
| $\rho$ | rugosidade | `_roughness`, via sigmoide |
| $L_s$ | radiância incidente pré-filtrada | `EnvironmentMap` / `SHLighting` |
| $W_i$ | rotação mundo → câmera da imagem $i$ | `world_view_transform.T[:3,:3]` |

Convenção de referencial: **tudo em espaço de vista** quando `light_frame="view"`.

---

## 2. Equação de sombreamento

$$c(\omega_o) \;=\; \gamma\Big( \underbrace{c_d + c_r(\omega_o)}_{\text{avaliado por } \texttt{eval\_sh}} \;+\; \underbrace{F(\mathbf{n}\cdot\omega_o,\ \mathbf{s}) \odot L_s(\mathbf{r},\rho)}_{\text{especular}} \Big)$$

**Por que $c_d$ e $c_r$ aparecem juntos.** Não é economia de código: é a condição que
torna a ablação interpretável (ADR-004). `eval_sh` sobre todas as SH devolve exatamente
$c_d + c_r(\omega_o)$, porque o grau 0 é constante em $\omega_o$ (é o difuso) e os graus
$\ell \geq 1$ são precisamente a parte dependente da vista (é o residual). Com
$\mathbf{s}=0$ e $\gamma=\mathrm{id}$, a expressão colapsa em
$\mathrm{clamp}(\texttt{eval\_sh}(\cdot)+0.5,\,0)$ — literalmente o baseline do GOF.

$\gamma$ é a identidade no Marco 1, justamente para preservar essa propriedade. O
tone mapping linear→sRGB entra no Marco 2, com o relighting HDR.

---

## 3. As duas normais — e por que isso importa

Este trabalho junta dois métodos que definem "normal" de formas **diferentes**. Confundi-las
seria um erro sutil, então vale explicitar.

**Normal do GOF: por raio, em espaço de vista.** Em `renderCUDA`, para cada pixel e cada
Gaussiana, o GOF calcula o ponto de interseção do raio com o elipsoide e toma o gradiente
do level-set ali:

$$\mathbf{n}^{\text{GOF}} = -\frac{Q\,\mathbf{p}}{\lVert Q\,\mathbf{p}\rVert},
\qquad Q = \text{forma quadrática da Gaussiana em espaço de vista},\ \mathbf{p} = \text{ponto do raio}$$

Depende do raio, portanto varia **dentro** de um mesmo splat. É a normal de alta
fidelidade que dá ao GOF sua qualidade geométrica.

**Normal do GaussianShader: por Gaussiana, no mundo.** O eixo de menor variância do
elipsoide — a direção em que a Gaussiana é achatada:

$$\mathbf{v}_i = R_i[:,\,k],\qquad k = \arg\min_j\ s_{ij}$$

$$\mathbf{n}_i = \mathrm{normalize}(\mathbf{v}_i + \Delta\mathbf{n}_i),
\qquad \text{sinal escolhido tal que } \mathbf{n}_i\cdot\omega_o > 0$$

**Uma sobre o `argmin`.** O GOF infla as escalas com o filtro 3D,
$\tilde{s} = \sqrt{s^2 + f^2}$. Como $\sqrt{\cdot}$ é monótona crescente, o $\arg\min$ é
**preservado** — a escolha do eixo não muda. Mas a *razão de anisotropia* muda, então
usamos `get_scaling` cru para deixar isso explícito.

**Como se relacionam.** As duas coincidem quando a Gaussiana é bem achatada, que é
exatamente o regime que a `distortion_loss` do GOF induz. Por isso a hipótese de trabalho
é que a `depth_normal_loss` que o GOF **já tem** basta para acoplá-las, e a perda
explícita de consistência entra apenas como ablação (E7). Medir a divergência entre as
duas definições é uma contribuição própria deste trabalho.

---

## 4. Reflexão e Fresnel

$$\mathbf{r} = 2(\mathbf{n}\cdot\omega_o)\,\mathbf{n} - \omega_o$$

Reflexão é uma involução ($\mathbf{r}$ refletido de volta devolve $\omega_o$) e preserva
norma — ambas verificadas em teste.

Fresnel de Schlick, com o tom especular fazendo o papel de $F_0$:

$$F(\cos\theta, \mathbf{s}) = \mathbf{s} + (1-\mathbf{s})(1-\cos\theta)^5,
\qquad \cos\theta = \mathbf{n}\cdot\omega_o$$

Em incidência normal $F = \mathbf{s}$; em ângulo rasante $F \to 1$ **independentemente de
$\mathbf{s}$**. É o brilho de borda, muito visível em testemunho de rocha e em qualquer
dielétrico. É também a razão de a redução exata ao baseline (§2) valer apenas com o
Fresnel desligado.

---

## 5. Pré-filtragem por rugosidade (split-sum)

Integrar o lóbulo especular a cada amostra é caro. A aproximação split-sum pré-filtra a
iluminação em níveis progressivamente mais borrados e escolhe o nível pela rugosidade.

**Envmap.** Pirâmide por `avg_pool` sucessivo, reconstruída a cada forward (diferenciável,
sem parâmetros extras). Nível $= \mathrm{clamp}(\sqrt{\rho}\,(L-1),\,0,\,L-1)$, com mistura
trilinear. A raiz distribui os níveis de forma mais uniforme perceptualmente: a maior
parte da mudança visual de nitidez acontece em $\rho$ baixo.

**SH.** Convolução da SH com um lóbulo de largura crescente em $\rho$:

$$L_s(\mathbf{r},\rho) = \sum_{\ell=0}^{L} a_\ell(\rho) \sum_{m=-\ell}^{\ell} c_{\ell m} Y_{\ell m}(\mathbf{r}),
\qquad a_\ell(\rho) = e^{-\ell(\ell+1)\rho^2/2}$$

Modos altos morrem primeiro, então rugosidade alta produz iluminação suave — o
comportamento físico correto, e o análogo contínuo dos mips.

---

## 6. Mesa giratória: por que o envmap vai no espaço de vista

Esta é a adaptação central do trabalho ao setup de captura, e o resultado é **exato**, não
aproximado.

**Setup.** O testemunho gira sobre uma mesa; a câmera e as luzes ficam paradas na sala.

**Referenciais.**
- $O$: referencial do **objeto**. É o que o COLMAP reconstrói, porque o COLMAP vê o objeto
  parado e a câmera orbitando.
- $R$: referencial da **sala**, onde as luzes de fato estão.
- $R_i$: rotação da mesa na imagem $i$ (desconhecida).
- $C$: rotação sala → câmera. **Constante**, porque a câmera não se move.

**O problema.** O envmap representa a iluminação. Se o ancorarmos em $O$ (o que a
literatura faz), estamos afirmando que a luz é fixa em relação ao objeto — falso: no
referencial do objeto, a luz gira. O modelo tentaria explicar como estático um reflexo
que se move, e falharia.

**Derivação.** O COLMAP entrega, para cada imagem, a rotação mundo→câmera

$$W_i = C\,R_i$$

Seja $\mathbf{r}_O$ uma direção de reflexão em coordenadas do objeto. Sua expressão no
referencial da sala é

$$\mathbf{r}_R = R_i\,\mathbf{r}_O
= (C^{-1}C)\,R_i\,\mathbf{r}_O
= C^{-1}\,(C R_i)\,\mathbf{r}_O
= C^{-1}\,W_i\,\mathbf{r}_O
= C^{-1}\,\mathbf{r}_{\text{view}}$$

onde $\mathbf{r}_{\text{view}} = W_i \mathbf{r}_O$ é simplesmente a direção de reflexão em
**espaço de vista** — que sabemos calcular.

Portanto, para o envmap verdadeiro $L$ definido na sala:

$$L(\mathbf{r}_R) = L(C^{-1}\mathbf{r}_{\text{view}}) =: L'(\mathbf{r}_{\text{view}})$$

$L'$ é apenas $L$ pré-rotacionado por $C^{-1}$. Como $C$ é uma **rotação global
constante**, otimizar $L'$ diretamente — isto é, indexar o envmap pela direção de reflexão
em espaço de vista — é equivalente a otimizar $L$. A rotação desconhecida é absorvida pelo
próprio mapa aprendido.

> **Consequência prática.** Não é preciso estimar o eixo da mesa nem o ângulo por imagem.
> Uma única linha (`reflect_dirs @ R_w2v.T`) resolve exatamente um erro de modelagem que
> invalidaria todos os reflexos.

**Suposição, e quando ela quebra.** A prova usa que $C$ é constante, ou seja, **uma única
estação de câmera**. Com $K$ alturas ou anéis de câmera, existem $C_1,\dots,C_K$ distintos
e um único $L'$ não serve para todos. Mitigação prevista: uma rotação aprendível por
estação (`--num_light_stations`). Antes de rodar as cenas de rocha é preciso saber quantas
posições de câmera a captura usa.

**Caso de controle.** Em Truck (Tanks&Temples) o objeto está parado e a câmera orbita, com
o sol fixo no mundo — o caso oposto. Lá o correto é `--light_frame world`, que recupera o
GaussianShader padrão. Ter os dois modos permite verificar que cada um vence no seu regime.

---

## 7. Perdas

$$\mathcal{L} = \underbrace{\mathcal{L}_{rgb} + \lambda_d \mathcal{L}_{dist} + \lambda_{dn}\mathcal{L}_{depth\text{-}normal}}_{\text{já existentes no GOF}} + \lambda_s\mathcal{L}_{sparse} + \lambda_r\mathcal{L}_{reg}$$

- $\mathcal{L}_{sparse} = \mathbb{E}[\lVert \mathbf{s}\rVert_1]$. Separar difuso de
  especular só a partir de RGB é **mal-posto**: um brilho pode ser explicado como albedo
  claro ou como reflexo. Sem esta penalidade o otimizador não tem razão para preferir uma
  das explicações, e o material aprendido não serve nem para relighting nem para leitura
  petrofísica. A L1 codifica o prior "poucas Gaussianas especulares", correto para rocha.
- $\mathcal{L}_{reg} = \mathbb{E}[\lVert\Delta\mathbf{n}\rVert^2]$, ativo só com
  `--use_normal_residual`.

**Warm-up.** Os regularizadores e o ramo especular só entram após `brdf_from_iter`
(default 3000). Ligar o especular na iteração 0 o põe competindo com o difuso antes de
existir geometria para refletir.

---

## 8. Fluxo de gradiente

```
                    imagem renderizada
                            │  dL/dpixel
                            ▼
        ┌───────── rasterizador CUDA (GOF, não modificado) ─────────┐
        │  backward.cu: atomicAdd(&dL_dcolors[i*C+ch], ...)         │
        └───────────────────────────┬───────────────────────────────┘
                                    │ grad_colors_precomp   [N,3]
                                    ▼
                    ┌── get_shading_colors (PyTorch) ──┐
                    │        autograd daqui p/ baixo    │
                    └───┬───────┬────────┬─────────┬────┘
                        ▼       ▼        ▼         ▼
                  _features  _specular  _rough  lighting
                  _dc/_rest   _tint     ness    (envmap ou SH)
                                    │
                                    └──► _normal_residual, _rotation, _scaling
```

O ponto de ADR-001: a fronteira CUDA→PyTorch já existia e já era diferenciável. Tudo
abaixo dela é autograd, então **não há nenhuma derivada escrita à mão para errar**.

---

## 9. Identificabilidade: o que os dados conseguem determinar

Registrado porque afeta a interpretação dos resultados, não só a implementação.

**Rugosidade sob luz uniforme.** Se toda direção tem a mesma radiância, borrar o lóbulo
especular não muda a imagem:

$$L_s(\mathbf{r},\rho) = \text{const} \;\Longrightarrow\; \frac{\partial \mathcal{L}}{\partial\rho} = 0$$

$\rho$ é **não-identificável**. Verificado empiricamente: com `SHLighting` inicializado
só com o DC, o gradiente da rugosidade é exatamente zero; com `EnvironmentMap` constante,
é ruído de ponto flutuante (2e-05 com variância de saída 3e-17).

Consequências: as classes de iluminação inicializam com ruído pequeno para não partir de
um ponto degenerado (ADR-008); e, numa cena de iluminação quase uniforme — caixa de luz
difusa, comum em bancada de rocha — **o valor final de $\rho$ deve ser lido com
ceticismo**. Isso precisa constar do artigo antes de qualquer interpretação petrofísica.

**Ambiguidade difuso/especular.** Mal-posta por construção; mitigada por
$\mathcal{L}_{sparse}$, pelo warm-up, e pela limitação do grau de SH (ablação E3).
