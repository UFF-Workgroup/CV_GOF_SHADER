# Resumo Executivo: do Problema ao Resultado

Este documento existe para uma pessoa que não acompanhou nada do processo. Não pressupõe
leitura prévia de `01_FUNDAMENTOS.md`, `02_DECISOES.md` ou dos demais — eles continuam
sendo a referência técnica completa; aqui a meta é dar o filme inteiro, do problema ao
número final, em uma sentada.

---

## 1. O problema, em uma frase

**3D Gaussian Splatting (3DGS)** e sua variante geométrica, **Gaussian Opacity Fields
(GOF)**, representam a cor de cada ponto da cena com harmônicas esféricas (SH) — uma base
matemática de **baixa frequência**. Isso funciona bem para superfícies foscas, mas falha
sistematicamente em **reflexos especulares** (o brilho num metal, num verniz, na borda
molhada de uma rocha): um reflexo agudo é alta frequência, e aproximá-lo com poucos termos
de SH produz borrão e artefatos espalhados pela cena.

O **GaussianShader** resolve isso trocando "guardar a cor observada" por "modelar por que a
cor é aquela" — separa a aparência em difuso + especular (com Fresnel, rugosidade e um
mapa de iluminação), mas não tem a extração de malha nem o campo de opacidade do GOF.

**A proposta deste trabalho:** fundir os dois — a geometria confiável do GOF com o
sombreamento físico do GaussianShader — para reconstruir **gêmeos digitais de testemunhos
de rocha**, onde tanto a forma quanto a resposta à luz importam (a rocha tem brilho de
borda, veios minerais reflexivos, texturas que um modelo puramente difuso apaga).

---

## 2. Como a fusão foi feita

**A decisão que evitou o trabalho mais arriscado (ADR-001).** O plano original mandava
reescrever a equação de sombreamento dentro dos kernels CUDA do rasterizador e derivar à
mão todos os gradientes para o *backward pass* — semanas de trabalho de alto risco, porque
um gradiente errado escrito à mão **não gera erro nenhum**, só faz o treino convergir mal
de um jeito difícil de diagnosticar. A auditoria do código descobriu algo melhor: o
rasterizador do GOF **já aceita** uma cor RGB pré-computada por Gaussiana
(`colors_precomp`) e **já devolve** o gradiente dela corretamente. Ou seja: bastava
calcular a cor do sombreamento em PyTorch comum — normal, reflexão, Fresnel, rugosidade,
consulta ao mapa de luz — e entregá-la por esse caminho. O autograd do PyTorch cuida de
propagar o gradiente sozinho, para tudo: tom especular, rugosidade, normal, mapa de
iluminação. **Zero derivada escrita à mão, zero linha de CUDA nova.** O rasterizador
continua sendo, byte a byte, o do GOF original — o que é bom para o método (mais simples,
mais fácil de auditar) e bom para o artigo ("nosso método não modifica o rasterizador").

**A equação de sombreamento** (`03_FORMULACAO.md` tem a derivação completa):

$$c(\omega_o) = \gamma\Big(\underbrace{c_d + c_r(\omega_o)}_{\text{já existia: SH do 3DGS}} \;+\; \underbrace{F(n\cdot\omega_o, s)\odot L_s(r,\rho)}_{\text{especular, novo}}\Big)$$

O truque elegante aqui (ADR-003): o termo residual $c_r(\omega_o)$ — a parte da cor que
muda com o ângulo de vista — **já existia** no 3DGS, escondido nos graus ≥1 das harmônicas
esféricas que todo splat carrega. Reaproveitá-lo em vez de criar um tensor novo tem duas
vantagens: economiza memória (importante numa GPU de 6 GB) e, principalmente, garante uma
propriedade crucial (ADR-004): **com o tom especular zerado, a equação inteira colapsa
exatamente na cor do GOF original** — testado por código, não assumido (`max|Δpixel| <
1e-5`). Sem essa garantia, uma diferença de qualidade entre "com especular" e "sem
especular" poderia vir de qualquer mudança acidental, não do especular em si — e a
comparação não significaria nada.

**A adaptação mais específica e publicável do trabalho (ADR-002).** Os testemunhos de
rocha são fotografados numa **mesa giratória**: o objeto gira, a câmera e as luzes ficam
paradas na sala. Isso parece um detalhe de bancada, mas quebra uma suposição que toda a
literatura de sombreamento em 3DGS faz: que o mapa de iluminação é fixo *no mundo*. Numa
mesa giratória, no referencial que o COLMAP reconstrói (o do objeto), **a luz gira junto
com a câmera** — um mapa fixo no mundo estaria tentando explicar como parado um reflexo que
na verdade se move, e falharia. A correção ingênua seria estimar o ângulo exato da mesa em
cada foto; a demonstração matemática em `03_FORMULACAO.md` §6 mostra que isso é
desnecessário: **consultar o mapa de luz pela direção de reflexão em espaço de vista (da
câmera) é exato**, porque a rotação desconhecida entre a sala e a câmera é global e
constante — o próprio mapa aprendido a absorve. Uma linha de código resolve um erro de
modelagem que, sem ela, invalidaria todos os reflexos da cena de rocha.

---

## 3. Os cinco bugs silenciosos — e por que cada um importava

O padrão que se repete nos cinco é o mesmo, e é o que torna cada um perigoso: **nenhum
gerava exceção**. As formas dos tensores continuavam corretas, o treino rodava até o fim,
os números saíam — só que errados, de um jeito que só aparece comparando com o que
*deveria* ter acontecido.

| # | O que era | Por que era perigoso | Quando foi pego |
|---|---|---|---|
| **A-1** | Ao podar Gaussianas de baixa opacidade, o código recriava o tensor de material a partir do valor *antigo*, descartando o que o otimizador já tinha podado. O modelo passava a ler um tensor; o Adam atualizava outro. | O material (`specular_tint`, `roughness`) **congelava exatamente na iteração 600** — a primeira poda — e nunca mais se movia, sem nenhum aviso. | Auditoria inicial do código herdado |
| **A-2** | `capture()`/`restore()` (usados por `--start_checkpoint`) não incluíam o material. | Retomar um treino de um checkpoint reconstruía o modelo com material **vazio**, e o Adam era montado sobre parâmetros de tamanho zero. | Auditoria inicial |
| **A-3** | O arquivo `.ply` salvo ao fim do treino não gravava o material. | Todo o pipeline de avaliação (`render.py`, `metrics.py`, `extract_mesh.py`) recarrega o modelo pelo `.ply` — as imagens de **avaliação** sairiam com material *default*, diferentes das de **treino**, sem nada no log denunciar. Era o bug com maior potencial de corromper silenciosamente os números do artigo. | Auditoria inicial |
| **A-4** | A mesma falha do A-2, só que para a **iluminação** (envmap/SH) e para os embeddings de aparência — descobertos só quando o checkpoint passou a ser levado a sério nesta sessão. | Retomar um treino com `--brdf` de um checkpoint reiniciaria o mapa de luz **aprendido** para ruído, enquanto reaplicava o momento do Adam calculado para os valores antigos — silenciosamente descartando toda a iluminação aprendida numa queda de energia. Ficou dormente até agora porque nenhum treino anterior tinha usado `--brdf`. | Auditoria do checkpoint, nesta sessão, **antes** de qualquer run longo o atingir |
| **A-5** | A conversão de direção de reflexão em coordenadas do mapa de luz usa `atan2`/`acos`, que têm gradiente que **diverge exatamente nos polos** do mapa (uma forma matemática `0/0`). Com mais de 100 mil Gaussianas por iteração, bastava uma cair perto o bastante de um polo. | O treino real (E1, ver §4) foi de fato a `NaN` na iteração 3020 — dez iterações depois do ramo especular ligar — e ficou `NaN` por mais de mil iterações sem se recuperar, silenciosamente destruindo o modelo. | **Ao vivo**, monitorando o primeiro run real com BRDF |

Os três primeiros (A-1 a A-3) foram herdados do código recebido no início do projeto e
corrigidos antes de qualquer experimento contar. Os dois últimos (A-4, A-5) foram achados
**durante esta sessão**, exatamente ao levar a sério os dois pedidos que motivaram o
trabalho recente: ter checkpoints confiáveis, e rodar o primeiro experimento de verdade. Se
não tivessem sido pegos agora, cada um invalidaria silenciosamente resultados futuros — A-4
numa eventual queda de energia em um run de 30k, A-5 em qualquer treino com `--brdf` e
`--light_repr envmap` (o modo default), ou seja, **todos** os runs BRDF planejados (E1, E2,
E4).

Todos os cinco têm teste de regressão dedicado — a suíte de testes do projeto foi de 25
para **36 testes passando**, e cada bug tem pelo menos um teste que teria pego o problema
original.

---

## 4. O experimento: E1, Truck, `--brdf --light_frame world`

Com a arquitetura pronta e os bugs corrigidos, rodou-se o primeiro experimento real da
matriz definida em `04_PROTOCOLO.md`: **E1**, na cena de controle **Truck**
(Tanks&Temples — objeto parado, câmera orbitando, sol fixo no mundo, o regime em que
`--light_frame world` é fisicamente correto), comparado contra **B1** — a mesma cena, a
mesma configuração, **sem** o BRDF ligado. Essa comparação controlada (mudando só uma
coisa) é o que existe para isolar o efeito do especular de qualquer outro fator.

Pelo caminho, a primeira tentativa foi justamente onde o bug A-5 apareceu e foi corrigido
ao vivo (ver tabela acima). A segunda tentativa, sobre o código corrigido, rodou limpa:

| | B1 (sem `--brdf`) | **E1** (`--brdf --light_frame world`) |
|---|---|---|
| PSNR | 25,2200 | **25,6254** |
| SSIM | 0,88632 | **0,88943** |
| LPIPS | 0,13379 | 0,13627 |
| Gaussianas finais | 2 089 655 | 2 084 149 |
| Pico de VRAM real | ~3 GB | 3,99 GB |
| Tempo total (30k it) | 6h50min | 8h22min |
| `Loss=nan` no log | — | **zero ocorrências** |

**PSNR subiu 0,41 dB, SSIM melhorou levemente, LPIPS piorou levemente.** O ganho de PSNR
está **acima do piso de ruído** de ~0,1 dB que o próprio treino tem por não ser
bit-reprodutível (`atomicAdd` no rasterizador) — não é atribuível só a variação aleatória
de execução.

---

## 5. Isso é uma contribuição científica válida?

**Resposta direta: ainda não — mas é um sinal real, na direção certa, e nada quebrou.**

O que o número **sustenta**:
- O especular **ajuda** e não atrapalha, mesmo numa cena majoritariamente fosca/metálica
  sem brilho intenso. O ganho de PSNR é maior que o piso de ruído, e SSIM concorda na mesma
  direção.
- A arquitetura funciona ponta a ponta: 30 000 iterações, zero `NaN`, VRAM dentro do
  orçamento, checkpoints confiáveis, redução exata ao baseline comprovada por teste.
- É consistente com a literatura: o próprio artigo do GaussianShader relata ganho pequeno
  em cenas reais predominantemente difusas (+0,19 dB em Tanks&Temples) e ganho maior só em
  datasets desenhados para especularidade (+1,1 dB) — `08_ARTIGOS_REFERENCIA.md` §1.7.
  Um ganho modesto em Truck é, portanto, o resultado **esperado**, não uma surpresa.

O que falta para chamar isso de resultado citável:

1. **Uma única execução.** O próprio protocolo do projeto (`04_PROTOCOLO.md`) exige **duas
   ou mais seeds** para qualquer número de destaque, justamente porque o rasterizador não é
   bit-reprodutível. Este é um resultado de **uma** seed — indicativo, não definitivo.
2. **A cena errada para a pergunta principal.** Truck é a cena de *controle*, escolhida
   porque tem `--light_frame world` fisicamente correto e permite comparar com a
   literatura — não é um teste duro de especularidade (é um caminhão, majoritariamente
   fosco). A pergunta que este trabalho existe para responder — *o sombreamento especular
   ajuda a reconstruir testemunhos de rocha fotografados em mesa giratória?* — só é
   respondida por **E2** (`--light_frame view`, cena de rocha), que ainda não rodou.
3. **E2 está bloqueado**, não por falta de dados (a cena `~/Documentos/rocha_fs16_16cm`
   já existe, com 72 fotos capturadas e verificadas — 8 faixas × 9 poses, confirmado por
   correlação de fase, ver ADR-010), mas porque a reconstrução COLMAP dela saiu
   **fragmentada em 4 sub-modelos** (`distorted/sparse/{0,1,2,3}`) em vez de um único
   modelo — algo que precisa ser investigado e resolvido antes de treinar.
4. **Sem comparação de geometria.** A extração de malha para E1 falhou por limite de VRAM
   (`07_LIMITACOES.md` §7) — não é possível ainda comparar Chamfer/F1 entre B1 e E1 nesta
   escala de cena neste hardware.

**Em uma frase:** há evidência preliminar, honesta e encorajadora de que a metodologia
funciona e ajuda — mas a validação científica completa depende de repetir E1 com mais
seeds e, principalmente, de rodar E2 na cena que de fato importa para este trabalho.

---

## 6. Próximos passos, em ordem

1. **Resolver a fragmentação do COLMAP** da cena de rocha (`rocha_fs16_16cm`) — bloqueio
   número um para qualquer progresso na pergunta central do trabalho.
2. **Rodar E2** (`--brdf --light_frame view`, rocha) assim que o COLMAP estiver
   consolidado — esta é a run que responde à pergunta que dá título ao projeto.
3. **Repetir E1 com ≥2 seeds**, para que o +0,41 dB vire um número citável com
   média ± desvio, conforme o protocolo já exige.
4. Se E2 confirmar o ganho, seguir para as ablações (E3–E7) que isolam *por que* o
   especular ajuda: grau de SH, envmap vs. SH, Fresnel, resíduo de normal.
5. Endereçar a limitação de VRAM em `extract_mesh.py` (processar pontos tetra em lotes)
   para viabilizar comparação geométrica.

---

*Este documento reflete o estado do projeto em 2026-09-02, commit `b06bb05`. Para a
matemática completa, ver `03_FORMULACAO.md`; para o log de decisões, `02_DECISOES.md`;
para o registro de todos os bugs com detalhe técnico, `06_AUDITORIA.md`; para a matriz de
experimentos e seus resultados, `05_EXPERIMENTOS.md`.*
