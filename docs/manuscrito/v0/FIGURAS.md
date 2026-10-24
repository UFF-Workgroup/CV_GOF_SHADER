# Figuras do manuscrito: guia de uso

Cada figura do artigo tem o seu próprio script em `scripts/figuras/`. Para mudar uma figura, abra o
script dela, altere o bloco `AJUSTES` no topo, rode o script e recompile o PDF. Nenhum número das
figuras é digitado à mão: tudo vem de `dados/` ou dos renders dos treinos.

## Uso rápido

```bash
cd docs/manuscrito/v0
conda activate gof                      # ou use ~/miniconda3/envs/gof/bin/python no lugar de python

python scripts/figuras/nero_metricas.py # refaz figuras/nero_metricas.pdf
./compilar.sh                           # remonta o PDF do artigo com a figura nova
```

Para refazer todas:

```bash
PY_GOF=~/miniconda3/envs/gof/bin/python ./scripts/figuras/todas.sh           # todas
PY_GOF=~/miniconda3/envs/gof/bin/python ./scripts/figuras/todas.sh so-dados  # só as que não precisam de output/
```

Os scripts podem ser rodados de qualquer pasta, porque os caminhos são resolvidos a partir do próprio
arquivo. A geração é determinística: rodar de novo sem mudar nada produz o mesmo arquivo, byte a
byte. Assim, o `git status` só acusa uma figura quando ela mudou de fato.

## Mapa das figuras

| Figura no PDF | `\label` | Script | Saída | Precisa de `output/`? |
|---|---|---|---|---|
| Fig. 1, pipeline | `fig:pipeline` | `pipeline.sh` (compila `figuras/pipeline.tex`) | `pipeline.pdf` | não (precisa de Docker) |
| Fig. 2, Truck | `fig:truck` | `truck.py` | `truck.pdf` | **sim**, renders de B1 e E1 |
| Fig. 3, NeRO qualitativo | `fig:nero_qualitativo` | `nero_qualitativo.py` | `nero_qualitativo.pdf` | **sim**, renders do NeRO |
| Fig. 4, métricas NeRO | `fig:nero_metricas` | `nero_metricas.py` | `nero_metricas.pdf` | não |
| Fig. 5, efeitos pareados | `fig:efeitos` | `efeitos_pareados.py` | `efeitos_pareados.pdf` | não |
| Fig. 6, custo × qualidade | `fig:custo` | `custo_qualidade.py` | `custo_qualidade.pdf` | não |
| Fig. 7, mecanismo | `fig:mecanismo` | `mecanismo.py` | `mecanismo.pdf` | não (usa os PNG de `envmaps.py`) |
| Fig. S1, potência | `fig:potencia` | `potencia.py` | `potencia.pdf` | não |
| Fig. S2, por vista | `fig:por_vista` | `por_vista_ajustado_gs.py` | `por_vista_ajustado_gs.pdf` | não |
| insumo da Fig. 7 | — | `envmaps.py` | `envmaps/<cena>_<método>.png` | **sim**, `lighting.pth` |

Os números das figuras são os da v0. Se a ordem mudar no texto, os scripts continuam valendo, porque
cada um tem o nome do arquivo que gera.

## De onde vêm os dados

```
saídas dos treinos (output/, fora do git)
 ├─ scripts/extrair_dados.py ───► dados/           (métricas, estatística, custo, material, potência)
 ├─ scripts/figuras/envmaps.py ─► figuras/envmaps/ (mapas de ambiente em PNG)
 └─ renders de teste ───────────► truck.py, nero_qualitativo.py (lidos direto)

dados/ (no git) ───► nero_metricas, efeitos_pareados, custo_qualidade, mecanismo, potencia, por_vista
```

- `dados/` e `figuras/envmaps/` estão versionados. Por isso sete das nove figuras (as seis em Python
  que leem só `dados/` e o diagrama) podem ser refeitas a partir de um clone, sem os treinos.
- `truck.py`, `nero_qualitativo.py` e `envmaps.py` leem `output/` na raiz do repositório. Se as
  saídas estiverem em outro lugar, aponte com `GOF_REPO=<pasta que contém output/>`.
- `envmaps.py` precisa de `torch`; as outras, só de `matplotlib`, `numpy` e `Pillow`.
- Rode `envmaps.py` antes de `mecanismo.py` quando os mapas mudarem. O `todas.sh` já segue essa ordem.

## O que ajustar em cada figura

Todo script tem um bloco assim no topo; é o único lugar que precisa ser editado para um ajuste comum:

```python
# --- AJUSTES -------------------------------------------------------------------------------------
VISTA = 24
...
# -------------------------------------------------------------------------------------------------
```

### Fig. 1: `pipeline.sh` e `figuras/pipeline.tex`

Diagrama TikZ do método: Gaussianas com material, sombreamento em PyTorch, rasterizador do GOF
inalterado, perdas e malha. Cores e textos se editam direto no `.tex`. As cores ficam no topo
(`novo` marca o que este trabalho acrescenta) e os textos em cada `\node`. Depois de editar, rode
`./scripts/figuras/pipeline.sh`.

### Fig. 2: `truck.py`

Linha de cima: uma vista de teste com a referência, o GOF, o GOF+BRDF e o mapa Δ erro. Esse mapa é
o erro do GOF+BRDF menos o do GOF: azul onde o GOF+BRDF erra menos, vermelho onde erra mais. Linha
de baixo: Δ PSNR de cada uma das 32 vistas, em ordem crescente, com a média tracejada.

| Ajuste | Padrão | Efeito |
|---|---|---|
| `VISTA` | 24 | vista mostrada; o PSNR do título é lido de `per_view.json` |
| `RUN_GOF`, `RUN_BRDF` | B1, E1 | treinos comparados |
| `LIM_DIF` | 0.1 | saturação da escala do mapa Δ erro |
| `CMAP_DIF` | `RdBu_r` | mapa de cores divergente, centrado em zero |
| `TAMANHO` | 7,2 × 3,2 pol. | dimensões |

### Fig. 3: `nero_qualitativo.py`

Grade com 6 linhas e 5 colunas: para cada cena, uma linha de renders e uma de erro absoluto médio por
pixel. O recorte é achado automaticamente em volta do objeto (os pixels que não são fundo branco na
referência). O PSNR da vista aparece no canto de cada render.

| Ajuste | Padrão | Efeito |
|---|---|---|
| `VISTA` | 5 | vista de teste (a mesma nas três cenas) |
| `VMAX_ERRO` | 0.25 | topo da escala de erro; menor realça diferenças pequenas |
| `CMAP_ERRO` | `inferno` | mapa sequencial (magnitude) |
| `MARGEM` | 24 | pixels em volta do objeto no recorte |
| `LIMIAR_FUNDO` | 0.97 | o que conta como fundo para achar o objeto |

### Fig. 4: `nero_metricas.py`

PSNR, SSIM e LPIPS médios dos quatro métodos em cada cena. Os losangos vazados são os números
publicados pelo GaussianShader (Tabela 5 do artigo deles, em `GS_PUBLICADO` no `comum.py`). Os
eixos não começam em zero, por isso a figura usa **pontos** e não barras: com eixo truncado, o
comprimento de uma barra mentiria sobre a razão entre valores.

| Ajuste | Efeito |
|---|---|
| `YLIM` | limites de cada painel |
| `TITULOS` | títulos, com a seta indicando o sentido melhor |
| `MOSTRAR_PUBLICADO` | liga ou desliga os losangos do GaussianShader publicado |

### Fig. 5: `efeitos_pareados.py`

Gráfico de floresta: cada linha é um contraste (Truck e os três contrastes de cada cena do NeRO).
O ponto é a diferença média por vista e a barra é o IC 95% por bootstrap pareado. Se a barra cruza
o zero, o efeito não se distingue de zero. O contraste "ajustado − GaussianShader" é invertido em
relação ao JSON (que guarda "GaussianShader vs ajustado") para que valor positivo sempre signifique
vantagem do nosso método no PSNR e no SSIM.

### Fig. 6: `custo_qualidade.py`

PSNR contra a energia da GPU no treino. Quanto mais para cima e para a esquerda, melhor. `XLIM` fixa
o mesmo eixo de energia nas três cenas, para comparar entre painéis.

### Fig. 7: `mecanismo.py` (e `envmaps.py`)

Duas linhas de curvas acumuladas, ponderadas pela opacidade: o tom especular e a rugosidade
aprendidos, com a configuração do Truck e com o ajuste. Embaixo, os mapas de ambiente da bell nas
duas resoluções. É a figura que refuta a hipótese de "rugosidade alta": com o ajuste, o tom sobe e
a rugosidade não cai.

- `mecanismo.py`: `CENA_ENVMAP` escolhe a cena dos mapas e `ROTULO_ENVMAP` define os rótulos.
- `envmaps.py`: o mapa é linear e tem alto alcance dinâmico. Para exibir, a radiância no percentil
  `PERCENTIL` (99,5) vira branco e a imagem recebe gama `GAMA` (2,2). Só a visualização muda; os
  dados não.

### Fig. S1: `potencia.py` e Fig. S2: `por_vista_ajustado_gs.py`

- **S1:** potência da GPU ao longo do treino, amostrada pelo `nvidia-smi` a cada 2 s e reamostrada
  para no máximo 400 pontos por `extrair_dados.py`. A área sob cada curva é a energia da Fig. 6.
- **S2:** Δ PSNR por vista entre o ajustado e o GaussianShader, em ordem crescente. Mostra se a
  diferença média vem de muitas vistas ou de poucas. `A`, `B` e `METRICA` trocam o contraste e a
  métrica.

## Escolhas comuns a todas as figuras (`comum.py`)

Mudar algo em `comum.py` muda todas as figuras de uma vez.

- **Cor fixa por método:** GOF roxo `#7B6CB5`, GOF+BRDF ocre `#C98A2E`, ajustado ferrugem
  `#A1440C`, GaussianShader azul `#1F7AA0`. A cor segue o método em todas as figuras e a paleta foi
  validada para daltonismo. Texto, eixos e grade usam tinta neutra, nunca a cor de uma série.
- **Fonte:** Liberation Serif, embutida como TrueType (`pdf.fonttype 42`), para o PDF não depender
  de fontes do leitor. Tamanhos de 6 a 8,5 pt na figura.
- **Largura:** as figuras de página inteira são desenhadas com `LARG_PAG` = 7,2 pol. e, depois do
  recorte de margens (`bbox_inches="tight"`), ficam entre 5,8 e 7 pol. O artigo as inclui com
  `width=\textwidth`, que é 6,5 pol. (papel carta, margens de 1 pol.). Por isso as mais largas são
  reduzidas em cerca de 7%, e as mais estreitas, ampliadas. Para escala 1:1, use `LARG_PAG = 6.5`
  (isso muda a diagramação e os tamanhos relativos da fonte).
- **Vírgula decimal:** o locale `pt_BR.UTF-8` põe vírgula nos eixos, e `virgula()` faz o mesmo nos
  rótulos. Sem esse locale no sistema, os eixos saem com ponto (`sudo locale-gen pt_BR.UTF-8`
  resolve).
- **Mapas de cor:** sequencial (`inferno`) para magnitude de erro; divergente centrado em zero
  (`RdBu_r`) para diferenças.

## Acrescentar uma figura nova

1. Copie o script mais parecido para `scripts/figuras/<nome>.py`, mude a docstring, o bloco `AJUSTES`
   e o nome passado a `salvar(fig, "<nome>.pdf")`.
2. Acrescente `<nome>` em `SO_DADOS` ou `COM_OUTPUT` no `todas.sh`.
3. No `.tex`: `\includegraphics[width=\textwidth]{figuras/<nome>.pdf}` e um `\label{fig:<nome>}`.
4. Acrescente uma linha na tabela deste guia.

## Problemas comuns

| Sintoma | Causa e solução |
|---|---|
| `ModuleNotFoundError: matplotlib` | o `python3` do sistema não tem matplotlib; use o do ambiente `gof` |
| `FileNotFoundError: ... precisa das saidas dos treinos em output/` | a figura lê renders; rode na máquina com `output/` ou aponte `GOF_REPO` |
| aviso `findfont: Font family 'Liberation Serif' not found` | instale `fonts-liberation`; senão cai em outra serifada e o tamanho muda |
| eixos com ponto decimal | falta o locale `pt_BR.UTF-8` |
| a figura mudou mas o PDF do artigo não | rode `./compilar.sh` depois do script da figura |

## Tabelas

As tabelas seguem a mesma ideia, mas num script só: `scripts/gerar_tabelas.py` lê `dados/` e escreve
`tabelas/*.tex`, que o artigo inclui com `\inputtab`. Rode `python3 scripts/gerar_tabelas.py .` dentro
de `v0/` e depois `./compilar.sh`.
