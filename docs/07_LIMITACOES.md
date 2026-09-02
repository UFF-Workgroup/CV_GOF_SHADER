# Limitações e Trabalho Futuro

Limitações conhecidas e declaradas. Estão aqui para entrar no artigo, não para serem
resolvidas antes dele.

---

## 1. Sombreamento por Gaussiana, não por pixel

A cor é avaliada **uma vez por Gaussiana por vista** e entra no rasterizador como
`colors_precomp`. A nitidez de um reflexo especular fica portanto limitada pela densidade
de Gaussianas: um highlight menor que um splat não pode ser representado.

É a mesma escolha do GaussianShader oficial, e ela funciona — mas cria uma assimetria
interessante com o GOF, cuja normal é calculada **por raio**. Sombreamos com uma
quantidade por-primitiva enquanto o GOF renderiza normais por-pixel.

*Mitigação parcial:* a densificação adaptativa tende a colocar mais Gaussianas onde o erro
fotométrico é alto, o que inclui highlights.

*Trabalho futuro:* variante em CUDA que avalie o BRDF por pixel usando a normal por raio do
GOF. Custo alto (derivadas manuais) e por isso deliberadamente adiada — ver ADR-001.

---

## 2. Rugosidade é não-identificável sob luz uniforme

Se a iluminação é uniforme, borrar o lóbulo especular não muda a imagem, logo
$\partial\mathcal{L}/\partial\rho = 0$ e $\rho$ **não pode ser estimada a partir dos
dados**. Não é limitação da implementação: é do problema.

**Isto importa diretamente para as cenas-alvo.** Testemunhos de rocha costumam ser
fotografados em caixa de luz difusa, justamente para eliminar sombras — o que produz
iluminação quase uniforme. Nesse regime $\rho$ fica mal condicionado.

*Consequência para o artigo:* qualquer interpretação petrofísica dos mapas de rugosidade
precisa vir acompanhada de uma medida de quão não-uniforme era a iluminação. Reportar
$\rho$ sem isso seria reportar ruído.

*Mitigação implementada:* as classes de iluminação inicializam com ruído pequeno
(`init_std=0.02`) para não partir de um ponto exatamente degenerado. Isso resolve o
arranque, não a identificabilidade.

Ver ADR-008 e `03_FORMULACAO.md` §9.

---

## 3. Uma única estação de câmera

A prova de que o envmap pode ser ancorado no espaço de vista (`03_FORMULACAO.md` §6) usa
que a rotação sala→câmera $C$ é **constante** — ou seja, a câmera não se move durante a
captura.

Se o protocolo usar várias alturas ou anéis de câmera, existem $C_1,\dots,C_K$ distintos e
um único envmap em espaço de vista deixa de ser exato.

*Mitigação prevista, não implementada:* uma rotação aprendível por estação
(`--num_light_stations K`), com as imagens agrupadas por posição de câmera.

**Pendência ativa:** é preciso confirmar com o pesquisador quantas estações a captura dos
testemunhos usa. É informação de bancada, não derivável do código.

---

## 4. Ambiguidade difuso/especular

Separar material de iluminação a partir apenas de imagens RGB é um problema **mal-posto**:
um pixel claro pode ser albedo alto sob luz fraca ou albedo baixo sob luz forte.

*Mitigações implementadas:* $\mathcal{L}_{sparse}$ sobre o tom especular, warm-up de 3000
iterações, inicialização quase difusa, e a limitação do grau de SH (que reduz a capacidade
de $c_r$ "absorver" o especular).

*O que continua verdade:* a decomposição obtida é **uma** explicação consistente com as
fotos, não necessariamente a física. Por isso a validação da decomposição de material
ficou **fora** das métricas-alvo. Relighting é a evidência indireta: se o material for
apenas ajuste de cor disfarçado, trocar a iluminação produzirá resultados implausíveis.

---

## 5. Treino não é bit-reprodutível

Os `atomicAdd` do rasterizador fazem a ordem de acumulação dos gradientes variar entre
execuções. Seeds fixas não bastam.

*Consequência:* diferenças de PSNR abaixo de ~0,1 dB não são interpretáveis sem repetição
por seed. Ver `04_PROTOCOLO.md`.

---

## 6. Restrição de 6 GB molda o espaço de configurações

`sh_degree=3` com ~2 M Gaussianas custa ~2,1 GB só em parâmetros e estados do Adam —
inviável. O default do modelo com BRDF é `sh_degree ≤ 1`.

Isso está convertido em hipótese testável (ablação E3) em vez de escondido como concessão,
mas continua sendo verdade que **não conseguimos testar todas as configurações neste
hardware**. Um resultado "grau 1 basta" precisa ser lido lembrando que grau 3 não foi
medido nesta cena.

Também indisponível: `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`, que só existe a
partir do torch 2.1.

---

## 7. `extract_mesh.py` não cabe em 6 GB para cenas com muitas Gaussianas

Achado ao tentar extrair a malha do run E1 (2 084 149 Gaussianas): `CUDA
OutOfMemoryError` dentro de `evaluage_alpha`, mesmo após liberar cache entre vistas —
o OOM ocorre já na primeira vista, então não é acúmulo, é o volume de pontos mantido
em VRAM o tempo todo. `get_tetra_points` gera **9 pontos por Gaussiana** (8 vértices de
caixa + 1 centro): com ~2 M Gaussianas isso são ~18,8 M pontos antes de processar
qualquer vista, sem flag de subamostragem disponível.

**Não é específico ao BRDF.** B1 (baseline, sem `--brdf`) tem 2 089 655 Gaussianas —
praticamente igual a E1 — e provavelmente bateria no mesmo teto; nenhuma extração de
malha na escala de 30k havia sido tentada antes desta sessão (o teste T8 usou o
checkpoint do *smoke test*, com só 117 892 Gaussianas).

**Consequência.** Comparação geométrica (Chamfer/F1) entre modelos na escala completa
fica **indisponível neste hardware** até `evaluage_alpha`/`integrate()` serem
reescritos para processar os pontos tetra em lotes, em vez de todos de uma vez. Enquanto
isso, a validação de reconstrução fica limitada a NVS (PSNR/SSIM/LPIPS) e inspeção
qualitativa de malhas em cenas menores (como o *smoke test* que validou T8).

---

## Trabalho futuro

1. Sombreamento por pixel em CUDA, acoplado à normal por raio do GOF (§1).
2. Estimativa explícita do eixo da mesa giratória, removendo a suposição de estação única (§3).
3. Rotação aprendível de iluminação por estação de câmera.
4. Validação dos mapas de material contra medidas petrofísicas independentes — o que
   exigiria ground-truth de reflectância que hoje não existe.
5. Medir quantitativamente a divergência entre as duas definições de normal (por raio vs.
   por Gaussiana) ao longo do treino: é uma contribuição própria ainda não explorada.
6. Reescrever `evaluage_alpha`/`integrate()` em `extract_mesh.py` para processar pontos
   tetra em lotes, para viabilizar Chamfer/F1 em cenas com muitas Gaussianas em 6 GB (§7).
