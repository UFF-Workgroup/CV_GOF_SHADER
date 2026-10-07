# Resultados comparativos no Truck: GOF, GOF+BRDF e GaussianShader

Comparação ponta a ponta de quatro sistemas treinados no Truck (Tanks&Temples), 30 mil iterações, resolução 978x546, mesmas 32 vistas de teste. Dados e saídas em `docs/experiments/COMPARACAO-truck/`.

## 1. Origem dos dados

| Sistema | Origem | Configuração relevante | Gaussianas |
|---|---|---|---|
| GOF (B1) | nosso PC | `sh_degree 0`, sem aparência desacoplada | 2,09 M |
| GOF+BRDF (E1) | nosso PC | `sh_degree 0`, `--brdf --light_frame world` | 2,08 M |
| GaussianShader | outro PC | BRDF próprio, envmap 512, `sh_degree -1` | 2,71 M |
| GOF (outro PC) | outro PC | `sh_degree 3`, aparência desacoplada, 421 min de treino | 2,15 M |

Os resultados do outro PC estão em `output/resultados pequisa/{gof,shader} [truck]/` (ignorado pelo git). O ground truth das três origens é idêntico pixel a pixel (diferença média 0,0), então a comparação é válida em imagens e resolução.

## 2. Pipeline de análise

```bash
R="output/resultados pequisa"
python scripts/compare_preds.py --gt output/EXP-20260901-02-e1/test/ours_30000/gt_2 \
  -m "GOF(outro-pc)=$R/gof [truck]/test/ours_30000/test_preds_2" \
  -m "GaussianShader=$R/shader [truck]/test/ours_30000/renders" \
  -m "GOF-B1(nosso)=output/EXP-20260811-02-b1/test/ours_30000/test_preds_2" \
  -m "GOF+BRDF-E1(nosso)=output/EXP-20260901-02-e1/test/ours_30000/test_preds_2" \
  --out docs/experiments/COMPARACAO-truck
python scripts/analyze_comparison.py --per_view docs/experiments/COMPARACAO-truck/per_view.json \
  --out docs/experiments/COMPARACAO-truck
python scripts/make_comparison_figures.py ...   # montagens e grafico de ganho por vista
```

- `compare_preds.py`: PSNR, SSIM e LPIPS-VGG por vista e médias, com o mesmo código do `metrics.py`.
- `analyze_comparison.py`: IC 95% por bootstrap pareado (10 mil reamostras), Wilcoxon pareado e d de Cohen, para todos os pares.
- `make_comparison_figures.py`: montagens GT x métodos e gráfico de ganho por vista.

## 3. Resultados individuais (média das 32 vistas)

| Sistema | PSNR | SSIM | LPIPS |
|---|---|---|---|
| GOF (B1) | 25,220 | 0,8863 | 0,1338 |
| GOF+BRDF (E1) | 25,625 | 0,8894 | 0,1363 |
| GaussianShader | 23,836 | 0,8496 | 0,1829 |
| GOF (outro PC) | 23,393 | 0,8960 | 0,1182 |

Os valores do GaussianShader batem com o `results.json` que veio do outro PC (23,836 / 0,8496 / 0,1829).

## 4. Comparativo dois a dois (B menos A)

| Par | Δ PSNR (dB) [IC 95%] | Δ SSIM | Δ LPIPS | Vistas com PSNR maior em B |
|---|---|---|---|---|
| E1 - B1 (controlado) | +0,41 [+0,29; +0,53] | +0,0031 | +0,0025 | 28/32 |
| E1 - GaussianShader | +1,79 [+1,54; +2,04] | +0,0398 | -0,0466 | 32/32 |
| B1 - GaussianShader | +1,38 [+1,15; +1,62] | +0,0367 | -0,0491 | 32/32 |
| E1 - GOF (outro PC) | +2,23 [+1,92; +2,56] | -0,0066 | +0,0181 | 32/32 |
| B1 - GOF (outro PC) | +1,83 [+1,53; +2,15] | -0,0097 | +0,0156 | 32/32 |
| GaussianShader - GOF (outro PC) | +0,44 [+0,03; +0,85] | -0,0464 | +0,0647 | 19/32 |

Δ LPIPS: negativo é melhor. Tabela completa com p de Wilcoxon e d de Cohen em `docs/experiments/COMPARACAO-truck/analysis.md`.

## 5. Leitura

- **Único par controlado: E1 - B1.** Mesmo código e hiperparâmetros, só muda `--brdf`. O ganho é de +0,41 dB de PSNR (mediana +0,38; maior +1,66 na vista 24), SSIM maior nas 32 vistas e LPIPS 0,0025 pior (piora em 30 de 32 vistas). Em 4 vistas o E1 fica atrás, no máximo 0,14 dB.
- **Contra os sistemas de outro PC, a causa não está isolada.** O GOF do outro PC usa `sh_degree 3` e aparência desacoplada; o GaussianShader usa hiperparâmetros próprios. O GOF do outro PC tem o melhor SSIM e LPIPS, mas PSNR 1,8 dB abaixo do B1; não investigamos o motivo.
- O GaussianShader é pior que B1 e E1 nas três métricas em todas as 32 vistas, no protocolo acima.

## 6. Limites de validade

- Uma semente por sistema; o treino não é reprodutível bit a bit, e diferenças abaixo de cerca de 0,1 dB não são interpretáveis sem repetição.
- Os dois sistemas de outro PC não foram retreinados com o protocolo do E1.
- Só o Truck, com a luz no referencial `world`; `--light_frame view` (mesa giratória) segue sem validação empírica.
- Não há métricas de malha neste conjunto.

## 7. Próximos passos

1. Repetir E1 e B1 em pelo menos 3 sementes.
2. Retreinar GOF e GaussianShader com o mesmo protocolo (`sh_degree`, aparência, resolução, iterações).
3. Ablações E4 a E7 (SH contra envmap, Fresnel, resíduo de normal, consistência de normal).
4. Métricas de geometria da malha e validação do referencial de vista.

## 8. Apresentação

Apresentação em 16 slides gerada a partir destes números (proposta, método, correção, resultados individuais e dois a dois, exemplos visuais, limites e próximos passos): https://claude.ai/artifact/LPZP2rRa4zTj1u6WS6MMsP (privada).
