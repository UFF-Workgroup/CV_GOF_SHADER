# Fundamentos: de 3DGS a GOF, de GaussianShader à fusão

Documento didático. Vai da ideia mais simples até a arquitetura deste trabalho, com a
intuição antes de cada fórmula. Escrito para ser lido por um coautor que conhece visão
computacional mas não este repositório.

A matemática formal está em `03_FORMULACAO.md`; aqui o objetivo é *entender por quê*.

---

## 1. 3D Gaussian Splatting: a ideia de base

**O problema.** Dadas fotos de um objeto de vários ângulos, reconstruir uma representação
que permita renderizar ângulos novos.

**A ideia.** Representar a cena como uma nuvem de **elipsoides gaussianos** semi-transparentes.
Cada Gaussiana tem posição, orientação, três escalas (é um elipsoide, pode ser achatado ou
alongado), opacidade e cor. Renderizar é projetar todos os elipsoides na tela e
compor por ordem de profundidade — daí "splatting", como jogar tinta na tela.

O que torna isso poderoso é ser **diferenciável**: dá para calcular como a imagem muda
quando se mexe cada parâmetro, e portanto otimizar milhões de Gaussianas por gradiente
descendente até que as renderizações batam com as fotos.

**Cor dependente da vista: harmônicas esféricas.** Uma superfície real não tem a mesma cor
de todo ângulo — madeira envernizada brilha de um lado, foscos não mudam. O 3DGS captura
isso guardando, em vez de um RGB, uma pequena função sobre a esfera de direções, expandida
numa base de **harmônicas esféricas (SH)**.

Vale a intuição: SH são para a esfera o que série de Fourier é para o círculo. O grau 0 é
uma constante (a cor média — o "difuso"); graus mais altos adicionam variação
progressivamente mais fina com a direção. Grau 3 usa 16 coeficientes por canal, 48 no
total por Gaussiana.

**A limitação que motiva tudo o mais.** SH são uma base de **baixa frequência**. Um
reflexo especular agudo — o ponto de luz numa esfera polida — é altíssima frequência.
Aproximá-lo com SH é como aproximar um pico estreito com poucos senos: sai borrado, e a
otimização compensa espalhando artefatos pela cena. É por isso que 3DGS e GOF falham em
superfícies refletivas.

---

## 2. Gaussian Opacity Fields: geometria de verdade

O GOF parte do 3DGS e ataca um problema diferente: extrair **superfície** confiável.

**Campo de opacidade.** Em vez de tratar a nuvem só como algo a renderizar, o GOF a
interpreta como um campo escalar de opacidade no espaço 3D. A superfície do objeto é um
**level set** desse campo — o conjunto de pontos onde a opacidade cruza um limiar. Isso dá
uma definição de superfície matematicamente limpa, em vez de heurística.

**A normal por raio.** É a peça mais importante do GOF para este trabalho, e a mais fácil
de confundir. Para cada pixel, o GOF acha onde o raio da câmera intersecta o elipsoide e
calcula ali o gradiente do level set. Essa é a normal.

Note o que isso significa: a normal **varia dentro de um mesmo splat**, porque depende do
raio. Não é uma propriedade da Gaussiana — é uma propriedade do par (Gaussiana, raio).
E ela sai em **espaço de vista** (coordenadas da câmera). Guardar esses dois fatos evita
um erro sutil na §4.

**Regularizadores.** O GOF adiciona duas perdas que herdamos de graça:
- *distortion loss*: empurra as Gaussianas a se concentrarem em torno da superfície, o que
  na prática as **achata**;
- *depth-normal consistency*: força a normal renderizada a concordar com a normal derivada
  do mapa de profundidade.

**Extração de malha.** Tetraedrização do espaço (Marching Tetrahedra) para extrair a malha
do level set, adaptativa e compacta.

**O que falta.** Nada em tudo isso modela **material**. A cor continua sendo SH — com a
limitação da §1.

---

## 3. GaussianShader: material em vez de só cor

O GaussianShader troca "guardar a cor observada" por "modelar por que a cor é aquela".

**A equação de renderização, simplificada.** A cor que sai de um ponto na direção da
câmera é:

$$c(\omega_o) = \underbrace{c_d}_{\text{difuso}} + \underbrace{s \odot L_s(\omega_o, n, \rho)}_{\text{especular}} + \underbrace{c_r(\omega_o)}_{\text{residual}}$$

Em português:

- **$c_d$, difuso.** A cor "própria" do material, igual de todo ângulo. É a cor de um
  tijolo, de uma rocha fosca.
- **$s$, tom especular.** *Quanto* e *de que cor* o material reflete o ambiente. Metais
  refletem colorido; dielétricos (rocha, plástico, água) refletem quase branco e fraco.
- **$\rho$, rugosidade.** *Quão nítido* é o reflexo. Espelho tem $\rho \approx 0$ e reflete
  uma imagem; rocha rugosa tem $\rho$ alto e reflete um borrão.
- **$L_s$, a luz incidente.** De onde vem a luz. Guardada como um **environment map** — uma
  foto panorâmica do ambiente ao redor.
- **$c_r(\omega_o)$, residual.** O que a física simplificada não explica.

**Como o reflexo é calculado.** Dada a normal $n$ e a direção da câmera $\omega_o$,
calcula-se a direção de reflexão $r$ (espelhar $\omega_o$ em torno de $n$) e consulta-se o
environment map naquela direção. É exatamente o que um espelho faz.

**O truque da pré-filtragem (split-sum).** Integrar corretamente o lóbulo especular a cada
amostra seria caríssimo. Em vez disso, pré-borra-se o environment map em vários níveis e
escolhe-se o nível pela rugosidade: liso lê o nível nítido, rugoso lê um borrado. É o mesmo
princípio dos mipmaps de textura.

**A normal do GaussianShader.** Uma Gaussiana achatada é praticamente um disco, e a normal
do disco é o **eixo de menor escala** do elipsoide. Como o achatamento nem sempre é
perfeito, adiciona-se um resíduo aprendível $\Delta n$, mantido curto por regularização.

---

## 4. A fusão — e as três decisões que a definem

### 4.1 Onde o sombreamento é calculado

O plano original mandava reescrever a matemática dentro dos kernels CUDA e derivar todos os
gradientes à mão. A auditoria mostrou que isso é desnecessário: o rasterizador do GOF já
aceita **cores pré-computadas por Gaussiana** (`colors_precomp`) e já devolve o gradiente
delas.

Ou seja, a fronteira diferenciável já existia. Basta calcular a cor em PyTorch e entregá-la
— o autograd propaga sozinho para tom especular, rugosidade, normal e environment map.
**Nenhuma derivada escrita à mão, portanto nenhuma derivada para errar.**

O preço, declarado: o sombreamento é **por Gaussiana** (uma cor por splat por vista), não
por pixel. A nitidez do reflexo fica limitada pela densidade de Gaussianas. É o que o
GaussianShader oficial faz, e funciona — mas é uma limitação real.

### 4.2 Onde as SH entram (e por que isso conserta a física)

A implementação herdada criou um tensor `_residual_color`: um RGB fixo por Gaussiana. Mas
o artigo define o residual como $c_r(\omega_o)$ — uma **função da direção**. Um RGB fixo
não é função de $\omega_o$: ele só duplica o difuso.

A correção é elegante porque a peça certa já estava lá: **as SH que o 3DGS carrega são
exatamente uma função da direção**. Então:

- grau 0 (o termo constante) **é** o difuso $c_d$;
- graus ≥ 1 (a parte que varia com a direção) **são** o residual $c_r(\omega_o)$.

Nenhum tensor novo, física correta, e um parâmetro redundante a menos — o que importa numa
GPU de 6 GB.

Isso também produz uma propriedade valiosa: com o especular zerado, a equação **colapsa
exatamente** no baseline do GOF. Sem isso, uma diferença de PSNR poderia vir de uma
mudança acidental na cor difusa em vez do termo especular, e a comparação não significaria
nada. A propriedade é verificada por teste, não assumida.

E abre uma pergunta científica de verdade: *com o especular modelado explicitamente, de
quanta SH ainda se precisa?* Vira ablação, não chute.

### 4.3 A mesa giratória

Aqui está a adaptação mais específica deste trabalho, e ela vem da física da captura.

Os testemunhos são fotografados numa **mesa giratória**: o objeto gira, a câmera e as luzes
ficam paradas na sala.

Agora pense no referencial. O COLMAP reconstrói tudo no referencial do **objeto** — do
ponto de vista dele, o objeto está parado e a câmera é que orbita. Mas nesse referencial,
**a luz gira junto com a câmera**. Um environment map ancorado no mundo — que é o que o
GaussianShader e toda a literatura assumem — afirmaria que a luz é fixa em relação ao
objeto. É falso, e o modelo tentaria explicar como estático um reflexo que se move.

A correção parece que exigiria estimar o eixo da mesa e o ângulo de cada foto. Não exige.
A derivação (`03_FORMULACAO.md` §6) mostra que **consultar o environment map na direção de
reflexão expressa em espaço de vista é exato** para esse setup — a rotação desconhecida
entre sala e câmera é constante e o próprio mapa aprendido a absorve.

Uma linha de código resolve um erro de modelagem que invalidaria todos os reflexos.

**A suposição embutida:** uma única posição de câmera. Se a captura usar vários anéis ou
alturas, a rotação deixa de ser constante e o modelo precisa de uma por estação.

### 4.4 As duas normais

Vale fechar com o ponto mais sutil da fusão. Os dois métodos definem "normal"
diferentemente:

| | GOF | GaussianShader |
|---|---|---|
| Granularidade | por **raio** (varia dentro do splat) | por **Gaussiana** |
| Referencial | espaço de vista | mundo |
| Origem | gradiente do level set na interseção | eixo de menor escala do elipsoide |

Não são a mesma coisa. Coincidem quando a Gaussiana é bem achatada — que é exatamente o
regime que a *distortion loss* do GOF induz. Por isso a hipótese de trabalho é que a
consistência já existente no GOF basta para acoplá-las, e a perda explícita entra apenas
como ablação. **Medir a divergência entre as duas definições é uma contribuição própria
deste trabalho.**

---

## 5. O que se espera medir

Três eixos, definidos com o pesquisador:

1. **Síntese de novas vistas** — PSNR/SSIM/LPIPS. O especular melhora a fidelidade?
2. **Geometria da malha** — Chamfer/F1. Modelar reflexo melhora as normais e a superfície,
   ou só a aparência?
3. **Relighting** — trocar o environment map e re-renderizar. É a demonstração de que o
   material aprendido tem significado, e não é apenas um ajuste de cor disfarçado.

E uma advertência de identificabilidade, registrada em ADR-008: **sob iluminação uniforme,
a rugosidade é indeterminável**. Se o testemunho for fotografado em caixa de luz difusa —
comum em bancada — o valor de $\rho$ aprendido deve ser lido com ceticismo, porque os dados
simplesmente não o determinam.
