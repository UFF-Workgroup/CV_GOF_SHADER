# Artigos de Referência: Fórmulas e Metodologia

Este documento mapeia os três artigos em `docs/artigos/` — as fontes teóricas por trás de
`01_FUNDAMENTOS.md` e `03_FORMULACAO.md` — equação por equação, com número de
seção/equação/página, para que qualquer fórmula usada no projeto seja **rastreável até a
fonte primária** em vez de citada de memória.

| Arquivo | Artigo | O que empresta ao projeto |
|---|---|---|
| `2311.17977v1.pdf` | **GaussianShader** — Jiang, Tu, Liu, Gao, Long, Wang, Ma. *3D Gaussian Splatting with Shading Functions for Reflective Surfaces.* arXiv:2311.17977v1, 29 nov 2023. | A equação de sombreamento base, a normal por eixo de menor escala + resíduo, o environment map treinável, a perda de consistência normal-geometria. |
| `2112.03907v1.pdf` | **Ref-NeRF** — Verbin, Hedman, Mildenhall, Zickler, Barron, Srinivasan. *Structured View-Dependent Appearance for Neural Radiance Fields.* arXiv:2112.03907v1, 7 dez 2021. | A reparametrização por direção de reflexão $\mathbf r=2(\mathbf n\cdot\omega_o)\mathbf n-\omega_o$; a fórmula de atenuação de banda SH por rugosidade $A_\ell(\kappa)\approx e^{-\ell(\ell+1)/2\kappa}$ — **fonte real** de $a_\ell(\rho)$ em `03_FORMULACAO.md` §5. |
| `2201.01487v1.pdf` | **Harmonics Virtual Lights (HVL)** — Mézières, Desrichard, Vanderhaeghe, Paulin (IRIT, Univ. Toulouse). *Fast Projection of Luminance Field on Spherical Harmonics for Efficient Rendering.* arXiv:2201.01487v1, 5 jan 2022. | A forma fechada luz×BRDF como produto escalar de coeficientes SH; projeção de fontes esféricas via Zonal Harmonics; evidência empírica de quantas bandas SH um reflexo especular exige. |

> **Achado principal desta varredura, antes dos detalhes:** ao ler os três artigos
> integralmente e comparar equação por equação com `03_FORMULACAO.md`, duas peças que os docs
> do projeto atribuem implicitamente ao GaussianShader **não estão nele**. Ver §4 (Tabela de
> correspondência e alertas de atribuição) — é a seção mais importante deste documento para
> quem for escrever o artigo final, porque uma citação errada é o tipo de erro que um revisor
> pega.

---

## 1. GaussianShader (arXiv 2311.17977)

### 1.1 Equação de sombreamento

**Eq. 3** (Seção 3.1, p.3):

$$c(\omega_o) = \gamma\big(c_d + s \odot L_s(\omega_o, \mathbf n, \rho) + c_r(\omega_o)\big)$$

| Termo | Significado | Natureza |
|---|---|---|
| $\gamma$ | tone mapping gama (sRGB) | **fixo, não identidade** — diferente do Marco 1 do projeto |
| $c_d\in[0,1]^3$ | cor difusa | constante por Gaussiana |
| $s\in[0,1]^3$ | tom especular ("specular tint") | atributo treinável por Gaussiana |
| $L_s(\omega_o,\mathbf n,\rho)$ | luz especular direta incidente | Eq. 4, split-sum |
| $c_r(\omega_o)\in\mathbb R^3$ | cor residual | SH de **3ª ordem fixa**, não é a mesma expansão de $c_d$ |

$s$ multiplica $L_s$ **diretamente** — não há termo de Fresnel $(1-\cos\theta)^5$ na Eq. 3.
$c_d$ e $c_r$ são dois termos **separados**, não a soma de todos os graus de uma única
expansão SH (ver §4 sobre a diferença com a leitura do projeto).

### 1.2 Normal por-Gaussiana

**Eixo de menor escala** (Seção 3.3, p.4): observação empírica de que a razão de aspecto do
elipsoide cresce durante o treino (Fig. 4) motiva usar o eixo mais curto do elipsoide,
$\mathbf v$, como normal aproximada. O artigo não escreve um $\arg\min$ formal — é descrito em
prosa, não em fórmula.

**Resíduo e ambiguidade de sinal — Eq. 5** (p.5):

$$\mathbf n = \begin{cases} \mathbf v + \Delta\mathbf n_1 & \text{se } \omega_o\cdot\mathbf v > 0 \\ -(\mathbf v + \Delta\mathbf n_2) & \text{caso contrário} \end{cases}$$

O artigo usa **dois resíduos aprendíveis separados** ($\Delta\mathbf n_1,\Delta\mathbf n_2$),
um por lado da ambiguidade, e testa o sinal contra o eixo cru $\mathbf v$ — não contra a
normal final $\mathbf n$. (`03_FORMULACAO.md` §3 usa um único $\Delta\mathbf n$ com sinal
escolhido contra $\mathbf n\cdot\omega_o$: simplificação deliberada do projeto, não um erro,
mas vale sabê-la como tal.)

### 1.3 Environment map e split-sum

**Representação** (Seção 3.2, p.4): **cubemap treinável $6\times64\times64$** — não
equirretangular/esférico.

**Integral de reflexão — Eq. 4** (p.4):

$$L_s(\omega_o,\mathbf n,\rho) = \int_\Omega L(\omega_i)\,D(\mathbf r,\rho)\,(\omega_i\cdot\mathbf n)\,d\omega_i$$

com $\Omega$ = hemisfério superior, $D$ = NDF **GGX** [Walter et al. 2007], $\mathbf r =
2(\omega_o\cdot\mathbf n)\mathbf n - \omega_o$.

Mecanismo declarado em prosa: pré-filtragem em múltiplos mip maps, um por nível de
rugosidade; interpolação entre eles para uma rugosidade arbitrária. **O artigo não dá a
fórmula de seleção de nível** — nenhum equivalente textual a "nível =
$\sqrt\rho\,(L{-}1)$". O artigo compara explicitamente essa escolha com a Integrated
Directional Encoding do Ref-NeRF, justificando mip maps por serem mais baratos em treino.

### 1.4 Fresnel

**Não existe Fresnel de Schlick (nem nenhum) na Eq. 3.** $s$ é um tint direto, sem
dependência angular. O comportamento "borda brilha independente de $F_0$" que
`03_FORMULACAO.md` §4 descreve **não vem deste artigo**.

### 1.5 Perdas

**Sparse loss — Eq. 8** (p.5), citando [30,50]:

$$\mathcal L_{sparse} = \frac{1}{|\alpha|}\sum_{\alpha_i}\big[\log(\alpha_i)+\log(1-\alpha_i)\big]$$

Atua sobre a **opacidade $\alpha$** (empurra para $\{0,1\}$, ajuda a geometria a convergir
para "placa fina") — **não sobre o tom especular $s$**. É uma perda homônima à
$\mathcal L_{sparse}$ de `03_FORMULACAO.md` §7 ($\mathbb E[\lVert s\rVert_1]$), mas atua
sobre um tensor diferente e resolve um problema diferente (esparsidade geométrica vs.
esparsidade de material).

**Consistência normal-geometria — Eq. 7** (p.5): $\mathcal L_{normal} =
\lVert\bar{\mathbf n}-\hat{\mathbf n}\rVert^2$, onde $\bar{\mathbf n}$ é o mapa de normais
renderizado e $\hat{\mathbf n}$ vem de um operador tipo-Sobel sobre o mapa de profundidade
renderizado. Motivação: sem essa perda, cada Gaussiana aprende seu resíduo isolada da
vizinhança (KNN explícito seria caro demais no treino).

**Regularização do resíduo — Eq. 6**: $\mathcal L_{reg} = \lVert\Delta\mathbf n\rVert^2$.

**Perda total — Eq. 9**: $\mathcal L = \mathcal L_{color} + \lambda_n\mathcal L_{normal} +
\lambda_s\mathcal L_{sparse} + \lambda_r\mathcal L_{reg}$, com $\lambda_n=0{,}01$,
$\lambda_s=0{,}001$, $\lambda_r=0{,}001$ nos experimentos do artigo (ponto de partida útil,
não diretamente portável — as perdas não são idênticas às do projeto).

### 1.6 Implementação e treino

- **O residual $c_r$ (SH 3ª ordem) começa desligado** e só entra depois — o oposto do
  `brdf_from_iter` do projeto, que atrasa o *especular*, não o *residual*
  (Suplementar §6, p.12: *"introduced in the later training stages for refinement"*).
  Candidato a ablação futura do projeto.
- Adam, 30.000 iterações, RTX 3090 (Seção 4.3, p.6).
- Grau de SH do residual **fixo em 3ª ordem** — o artigo não faz dele um eixo de ablação
  (isso é original ao projeto, E3).

### 1.7 Resultados principais

| Dataset | 3DGS | GaussianShader | Δ |
|---|---|---|---|
| NeRF Synthetic (majoritariamente difuso) | 33,30 dB | 33,38 dB | +0,08 |
| Glossy Synthetic | 26,26 dB | 27,36 dB | **+1,10** |
| Tanks and Temples (cenas reais, difusas) | 29,54 dB | 29,73 dB | +0,19 |
| Shiny Blender | — | ligeiramente **abaixo** de Ref-NeRF/ENVIDR | — |

Custo (Tabela 3): Ref-NeRF 23h/0,03 FPS; ENVIDR 6h/1,33 FPS; 3DGS 0,25h/274 FPS;
**GaussianShader 0,58h/97 FPS**.

Ablação (Tabela 4, Shiny Blender): maior queda isolada ao remover $\mathcal L_{normal}$
(−1,16 dB) e ao trocar o cubemap por MLP de luz estilo Ref-NeRF (−2,36 dB) — a
representação explícita de luz é o componente de maior impacto.

### 1.8 Limitações declaradas

1. Fica atrás de ENVIDR/Ref-NeRF em superfícies muito especulares — atribuído à falta de
   uma representação de superfície contínua tipo SDF.
2. Ganho pequeno em cenas do mundo real predominantemente difusas (+0,19 dB em
   Tanks and Temples) — *"the objects... are predominantly diffuse, which does not fully
   leverage the strengths of our approach."*
3. Não discute identificabilidade de $\rho$ sob luz uniforme (contribuição própria do
   projeto, ADR-008).
4. Não discute captura com objeto girando/luz parada — assume implicitamente envmap fixo
   no mundo com câmera orbitando (a lacuna que o ADR-002 do projeto preenche).

**Relevância para rocha fosca:** o ganho do método escala com a especularidade da cena
(+1,10 dB em objetos vítreos/metálicos vs. +0,19 dB em cenas reais difusas). Isso é
consistente com o "Marco de decisão" de `04_PROTOCOLO.md` — se E2 não superar B2 por muito,
não é necessariamente falha de implementação: o próprio artigo-fonte mostra retornos
decrescentes em baixa especularidade.

---

## 2. Ref-NeRF (arXiv 2112.03907)

### 2.1 Reparametrização por direção de reflexão

**Eq. 4** (Seção 3.1, p.4):

$$\hat\omega_r = 2(\hat\omega_o\cdot\hat n)\hat n - \hat\omega_o$$

— **fonte formal** da fórmula de reflexão usada em `03_FORMULACAO.md` §4. Justificativa
(Eq. 5, p.4): para BRDFs rotacionalmente simétricas em torno de $\hat\omega_r$, ignorando
interreflexão e auto-oclusão, a radiância de saída é função só de $\hat\omega_r$:

$$L_{out}(\hat\omega_o) \propto \int L_{in}(\hat\omega_i)\,p(\hat\omega_r\cdot\hat\omega_i)\,d\hat\omega_i = F(\hat\omega_r)$$

Consultar a MLP direcional por $\hat\omega_r$ em vez de $\hat\omega_o$ mantém a função-alvo
aproximadamente **constante ao longo da superfície** sob luz distante — em vez de variar
rápido e de forma irregular com a orientação local (Fig. 2). É a raiz do "problema
off-specular" (§2.5).

O artigo também injeta $\hat n\cdot\hat\omega_o$ como feature extra da MLP direcional, para
permitir efeitos tipo-Fresnel — mas **não escreve uma fórmula de Fresnel fechada**; deixa a
rede aprender a dependência angular implicitamente.

### 2.2 Integrated Directional Encoding (IDE) — fonte da atenuação por rugosidade

**Eq. 6-8** (Seção 3.2, p.5). Codifica a distribuição vMF (von Mises–Fisher) de reflexões
possíveis, centrada em $\hat\omega_r$ com concentração $\kappa=1/\rho$:

$$\mathrm{IDE}(\hat\omega_r,\kappa)=\{\mathbb E_{\hat\omega\sim\mathrm{vMF}(\hat\omega_r,\kappa)}[Y_\ell^m(\hat\omega)]\}_{(\ell,m)\in\mathcal M_L}$$

Forma fechada (Eq. 7, prova completa no Apêndice A):

$$\mathbb E_{\hat\omega\sim\mathrm{vMF}}[Y_\ell^m(\hat\omega)] = A_\ell(\kappa)\,Y_\ell^m(\hat\omega_r)$$

com a **função de atenuação por banda — Eq. 8** (p.5):

$$A_\ell(\kappa) \approx \exp\!\left(-\frac{\ell(\ell+1)}{2\kappa}\right)$$

**Esta é a fórmula de origem** de $a_\ell(\rho)=e^{-\ell(\ell+1)\rho^2/2}$ em
`03_FORMULACAO.md` §5 — com $\kappa=1/\rho$ trocado por $\rho^2$ multiplicando (escolha de
parametrização do projeto; a estrutura $\ell(\ell+1)$ e o formato exponencial são idênticos
ao artigo). A forma exata (não aproximada) é a Eq. S8 do Apêndice A, e a aproximação
exponencial é exata a menos de $O(1/\kappa^2)$ (Claim 3, p.12).

Fisicamente: rugosidade maior → $\kappa$ menor → atenuação mais forte de harmônicos de alta
ordem → interpolação mais borrada — o análogo direto de um mipmap de rugosidade, agora em
domínio SH em vez de espacial.

### 2.3 Separação difuso/especular

**Eq. 9** (Seção 3.3, p.5): $\mathbf c = \gamma(\mathbf c_d + \mathbf s\odot\mathbf c_s)$,
com $\mathbf c_d,\mathbf s$ saída da MLP espacial (função só de posição) e $\mathbf c_s$
saída da MLP direcional (recebe IDE + $\hat n\cdot\hat\omega_o$ + bottleneck). Sem Fresnel
de Schlick explícito — ver nota em §2.1.

### 2.4 Normais

Normal "verdadeira" por gradiente de densidade (Eq. 3): $\hat n(\mathbf x) =
-\nabla\tau(\mathbf x)/\lVert\nabla\tau(\mathbf x)\rVert$.

- **Perda de predição de normal — Eq. 10** (p.6): $\mathcal R_p = \sum_i w_i\lVert\hat n_i -
  \hat n_i'\rVert^2$, amarra a normal predita pela MLP à normal por gradiente, ponderada
  pelos pesos de compositing.
- **Perda de orientação — Eq. 11** (p.6): $\mathcal R_o = \sum_i w_i
  \max(0,\hat n_i'\cdot\hat{\mathbf d})^2$, penaliza normais "de costas" para a câmera em
  amostras visíveis — evita que o modelo explique especularidades com emissores ocultos sob
  uma casca semitransparente ("foggy shell").

Estruturalmente análogo à `depth_normal_loss` que o GOF já traz e à
$\mathcal L_{normal}$ do GaussianShader (§1.5) — mesma família de ideia (consistência entre
normal predita e normal geométrica), três implementações distintas.

### 2.5 O problema off-specular

Perto de um realce especular, a radiância verdadeira muda muito rápido e de forma não-linear
com a direção de vista mesmo para geometria simples — a MLP condicionada em $\hat\omega_o$
fica mal-condicionada nessa região: reflexos "aparecem e desaparecem" entre vistas em vez de
se mover suavemente, e o campo de densidade tende a ficar "nebuloso" porque o otimizador
esconde emissores dentro do objeto para simular o brilho. A reparametrização por
$\hat\omega_r$ resolve isso ao tornar a função-alvo aproximadamente constante ao longo da
superfície; a perda de orientação resolve o problema associado da casca nebulosa.

### 2.6 Hiperparâmetros

Adam ($\beta_1{=}0{,}9,\beta_2{=}0{,}999,\varepsilon{=}10^{-6}$), 250k iterações, batch
$2^{14}$, LR log-linear $2\times10^{-3}\to2\times10^{-5}$ com warm-up de 512 iterações,
grad clipping $10^{-3}$. Pesos: $\mathcal R_o$ = 0,1; $\mathcal R_p$ = $3\times10^{-4}$
(sintético) / $10^{-3}$ (real). Ruído gaussiano $\sigma=0{,}1$ no bottleneck durante o
treino.

### 2.7 Resultados

Dataset próprio "Shiny Blender" (6 objetos brilhantes/metálicos, 100 treino/200 teste):
Ref-NeRF 35,96 dB / MAE normal 18,38° vs. mip-NeRF 31,59 dB / MAE 58,07° — melhoria de
~4,3 dB e ~35% menos erro de normal. Blender padrão: 33,99 vs. 33,09 dB. Ablação confirma
cada componente: sem reparametrização por reflexão, 35,96→29,47 dB; sem $\mathcal R_o$,
→31,62 dB (MAE piora para 52,56°).

### 2.8 Limitações declaradas

1. Custo computacional: IDE + backprop por densidade para normais deixa o modelo ~25% mais
   lento que mip-NeRF. **Não se aplica ao projeto** — o GOF já dá normais por raio via
   level-set, sem precisar de backprop por densidade, e o projeto usa Gaussianas explícitas,
   não MLP.
2. **Assume BRDF rotacionalmente simétrica e iluminação distante, sem interreflexão nem
   auto-oclusão** (explícito na derivação da Eq. 5). Esta é exatamente a mesma suposição
   simplificadora que `03_FORMULACAO.md` herda ao tratar $L_s$ como envmap "de fundo" sem
   ray tracing de sombras — vale citar este precedente ao declarar a suposição em
   `07_LIMITACOES.md`.
3. Não faz decomposição inversa física completa (BRDF+iluminação com significado físico
   garantido) — os componentes "se comportam intuitivamente" o bastante para edição, sem
   mais garantia que isso.

---

## 3. Harmonics Virtual Lights (arXiv 2201.01487)

### 3.1 Equação de rendering em SH

Luminância de saída via HVLs (Eq. 1-2, p.3): soma sobre luzes virtuais harmônicas, cada
contribuição sendo a equação de rendering padrão $\int_\Omega L_j(\omega_i)F(\omega_i,
\omega_o)\,d\omega_i$ com $F=\cos\theta_i\, f_s$. Com luminância aproximadamente constante no
ângulo sólido da HVL (Eq. 3) e projetando $L$ e a BRDF $F$ em SH como vetores de
coeficientes, a integral vira **produto interno** (Eq. 4, Seção 4.1):

$$\int_\Omega L(\omega_i)F(\omega_i)\,d\omega_i = \mathbf L\cdot\mathbf F$$

Forma final usada pelo método (Eq. 19, p.7): $L_{j,o} \approx L_j(x,\omega_j)\,\mathbf
L\cdot\mathbf F_{\omega_o,x}$ — luz × BRDF vira produto escalar de dois vetores SH em vez de
integral numérica.

### 3.2 Rugosidade na convolução SH — não há fórmula analítica

**Achado importante:** o artigo **não** modela rugosidade como atenuação analítica por
banda. A BRDF (GGX ou dados MERL, já contendo a rugosidade do material) é **tabulada
numericamente e projetada em SH por direção de saída discretizada** (Eq. 16-18, grade
90×360 ou 90 amostras se isotrópica) — não existe um $A_\ell(\rho)$ fechado. O único
controle explícito de frequência é uma **janela sobre os coeficientes SH da BRDF** para
mitigar ringing de band-limit, aplicada no pré-processamento, remetendo a [Slo08] sem dar a
fórmula.

**Conclusão para o projeto:** $a_\ell(\rho)=e^{-\ell(\ell+1)\rho^2/2}$ **não vem deste
artigo** — vem do Ref-NeRF (§2.2). HVL trata rugosidade via tabulação numérica, não via
atenuação de banda fechada.

### 3.3 Projeção de fontes esféricas via Zonal Harmonics

Fonte = calota esférica de ângulo sólido $Q$, luminância constante dentro, zero fora.
Alinhando o eixo $z$ do referencial local à direção da luz $\omega_{light}$, só sobrevivem
coeficientes de Zonal Harmonics ($m=0$). Resolvendo a integral de Legendre (Eq. 13),
com $a$ = meio-ângulo subtendido e $\alpha=\cos a$:

$$\tilde{\mathbf L}_l = \begin{cases}\sqrt{\dfrac{\pi}{2l+1}}\big(P_{l-1}(\alpha)-P_{l+1}(\alpha)\big) & l\neq 0\\[4pt] \sqrt\pi\,(1-\alpha) & l=0\end{cases}\qquad\text{(Eq. 14)}$$

Rotação para SH completo (Eq. 15, método de Sloan [SLS05]):

$$\mathbf L_l^m = \sqrt{\frac{4\pi}{2l+1}}\,Y_l^m(\omega_{light})\,\tilde{\mathbf L}_l$$

— evita rotação geral de SH (custosa) ao integrar a rotação diretamente na projeção. Quando
a convolução se restringe a lóbulos circularmente simétricos (ZH-only), a complexidade cai
de $\mathcal O(n^2)$ para $\mathcal O(n)$ em número de bandas $n$ (Seção 6.3, comparação com
VSGL) — a origem do "$O(n)$ vs $O(n^2)$" do abstract.

**Padrão estrutural reutilizável:** em vez de rotacionar o objeto complexo (BRDF/envmap
completo), rotaciona-se/ancora-se o termo mais simples (a direção da luz) e a matemática
absorve a rotação — o mesmo princípio do ADR-002 do projeto (não estimar o ângulo da mesa;
deixar o mapa aprendido absorver a rotação desconhecida).

### 3.4 Bandas SH necessárias — evidência empírica relevante à ablação E3/E4

Recomendação prática dos autores: 0-5 bandas para emissão da HVL, 0-10 para a convolução no
fragmento, escalando a 0-20 para milhares de HVLs. Em cena com reflexo especular forte no
chão (Fig. 13, Sponza), precisaram de **15 bandas** para captar o reflexo de forma
convincente — evidência direta de que grau baixo borra reflexos nítidos. Custo de gathering
cresce ~2,8-6× de 3 para 9 bandas (quadrático no número de coeficientes, $(N{+}1)^2$).

**Relevância direta:** com `sh_degree ≤ 1` imposto pela VRAM de 6 GB (ADR-005), a literatura
sugere que um reflexo realmente especular exigiria ordens muito mais altas do que o
orçamento permite. Reforça tratar isso como **hipótese testável** (E3/E4), não suposição —
e sugere que, se a rocha for majoritariamente fosca (reflexo largo, baixa frequência), o
grau baixo pode bastar; se houver brilho pontual agudo, é esperado que não baste.

### 3.5 Limitações declaradas

- **Kernel angular abrupto** (fronteira dura da calota esférica, Eq. 9) produz *ringing* em
  alta frequência — os próprios autores reconhecem que um kernel suave ajudaria, sem
  resolver. Confirma que SH de banda limitada degrada reflexos nítidos de forma estrutural,
  não incidental.
- Materiais anisotrópicos custam ~360× mais armazenamento — não aplicável ao projeto (1
  material aprendido por Gaussiana, não biblioteca de materiais).
- Visibilidade aproximada só pelo centro da fonte, sem sombras suaves — não relevante ao
  projeto (sem GI/HVLs, só a representação SH da luz incidente).

---

## 4. Tabela de correspondência e alertas de atribuição

Comparação direta entre o que `03_FORMULACAO.md`/`01_FUNDAMENTOS.md` descrevem e o que os
artigos realmente dizem — para uso na hora de escrever as citações do artigo final.

| Peça do projeto | Onde está nos docs | Atribuição correta | Nota |
|---|---|---|---|
| $\mathbf r=2(\mathbf n\cdot\omega_o)\mathbf n-\omega_o$ | `03_FORMULACAO.md` §4 | **Ref-NeRF**, Eq. 4 | Confirmado, citação correta possível |
| $a_\ell(\rho)=e^{-\ell(\ell+1)\rho^2/2}$ | `03_FORMULACAO.md` §5 | **Ref-NeRF**, Eq. 8 ($A_\ell(\kappa)$, $\kappa=1/\rho$) | ⚠️ Os docs atuais não citam a fonte explicitamente perto da fórmula — vale adicionar `[Verbin et al. 2022, Eq. 8]` |
| $F=s+(1-s)(1-\cos\theta)^5$ (Fresnel-Schlick) | `03_FORMULACAO.md` §4 | **Nenhum dos três artigos** — Schlick 1994 é a fonte clássica; nem GaussianShader nem Ref-NeRF usam essa forma fechada explícita | ⚠️ Extensão própria do projeto sobre o GaussianShader. Correto tecnicamente, mas não atribuir ao GaussianShader se citado |
| Normal = eixo de menor escala + $\Delta n$ | `03_FORMULACAO.md` §3 | **GaussianShader**, Eq. 5 | Correto em espírito; artigo usa dois resíduos ($\Delta n_1,\Delta n_2$), projeto usa um só — simplificação declarável |
| $c_d+c_r(\omega_o)$ = `eval_sh` sobre todos os graus | `02_DECISOES.md` ADR-003 | **Reinterpretação do projeto** — GaussianShader trata $c_d$ e $c_r$ como termos separados (Eq. 3), não a mesma expansão SH | Fisicamente equivalente e é uma decisão de engenharia defensável (ADR-003 já argumenta por que), mas não é literalmente o que o artigo escreve |
| Split-sum: nível = $\mathrm{clamp}(\sqrt\rho(L{-}1),0,L{-}1)$ | `03_FORMULACAO.md` §5 | **Nenhum dos três artigos** dá essa fórmula exata — GaussianShader descreve mip maps em prosa sem fórmula de seleção de nível | Compatível com a literatura padrão de split-sum (Karis/UE4), mas não citável a nenhum destes três papers |
| $\mathcal L_{sparse}$ sobre $s$ (tom especular) | `03_FORMULACAO.md` §7 | **Divergência deliberada do GaussianShader** — lá, $\mathcal L_{sparse}$ (Eq. 8) atua sobre a **opacidade** $\alpha$, não sobre $s$ | Nomes iguais, tensores e propósitos diferentes — desambiguar se citado |
| $\mathcal L_{reg}=\mathbb E[\lVert\Delta n\rVert^2]$ | `03_FORMULACAO.md` §7 | **GaussianShader**, Eq. 6 | Confirmado |
| Iluminação distante, sem interreflexão/sombra | premissa implícita em `03_FORMULACAO.md` | **Ref-NeRF**, explícita na derivação da Eq. 5, e **HVL** também assume BRDF sem GI complexa no termo direto | Vale citar como precedente da literatura ao declarar a suposição em `07_LIMITACOES.md` |
| Grau de SH baixo → reflexo borrado | ADR-005, E3/E4 | **HVL**, Seção 4.2 (empírico: 15 bandas para reflexo nítido em Sponza) + **Ref-NeRF** (mesma lógica via $A_\ell(\kappa)$) | Reforça, não contradiz, a decisão do projeto |

**Recomendação prática.** Nenhuma dessas divergências é um erro de implementação — o código
não foi auditado aqui, só a prosa dos docs de formulação contra os artigos-fonte. Mas ao
escrever o artigo final, ADR-003, ADR-004 e `03_FORMULACAO.md` §4-5 devem citar Ref-NeRF (não
GaussianShader) para a fórmula de atenuação por banda, e declarar o Fresnel de Schlick e a
fórmula de seleção de nível do split-sum como escolhas de engenharia do projeto, não como
citações. Se for do interesse, um ADR novo (append-only, não se edita os existentes) pode
registrar essa correção de atribuição — avise se quiser que eu redija.
