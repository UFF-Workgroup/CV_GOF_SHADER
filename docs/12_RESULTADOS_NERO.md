# Resultados no NeRO GlossySynthetic (bell, cat, teapot)

GOF + BRDF e GaussianShader treinados e avaliados nas mesmas cenas, com perfil de tempo, energia e VRAM. Uma semente por execucao.

## Como foi feito
- Dados: `GlossySynthetic.tar.gz` do NeRO, convertido por `nero2blender.py` do GaussianShader (96 vistas de treino, 32 de teste, 800x800, fundo branco). Para o GOF, `scripts/nero_to_gof.py` acrescenta `camera_angle_x`.
- GOF + BRDF: `--sh_degree 0 --brdf --light_frame world -w`, 30 mil iteracoes, inicializacao aleatoria de 100 mil pontos.
- GaussianShader: comando do README para cenas Blender (`--brdf_dim 0 --sh_degree -1 --lambda_predicted_normal 2e-1 --brdf_env 512 -w`), 30 mil iteracoes, ambiente `gs` separado.
- Fila serial (`scripts/queue_nero.sh`), perfil por `scripts/run_benchmark.py` (nvidia-smi a cada 2 s), renders por `scripts/render_nero.sh`, metricas por `scripts/compare_preds.py` e graficos por `scripts/nero_report.py`. GT identico nos dois (diferenca maxima 0).
- Hardware: RTX 3050 de 6 GB. Os treinos rodaram um por vez porque o GaussianShader usa ate 5,7 GB.

## Resultados (media das 32 vistas de teste)
| Cena | Metodo | PSNR | SSIM | LPIPS | Tempo (h) | Energia (Wh) | VRAM (GB) |
|---|---|---|---|---|---|---|---|
| bell | GOF + BRDF | 24,81 | 0,8928 | 0,1159 | 3,18 | 153 | 2,8 |
| bell | GaussianShader | **29,84** | **0,9405** | **0,0834** | 6,09 | 414 | 5,3 |
| cat | GOF + BRDF | **32,37** | **0,9665** | **0,0541** | 2,23 | 107 | 2,6 |
| cat | GaussianShader | 31,32 | 0,9607 | 0,0578 | 6,08 | 414 | 5,5 |
| teapot | GOF + BRDF | 23,08 | 0,8889 | 0,0982 | 3,77 | 180 | 2,6 |
| teapot | GaussianShader | **23,82** | **0,9021** | **0,0946** | 6,47 | 439 | 5,6 |

Diferenca GaussianShader menos GOF + BRDF: bell +5,03 dB (32/32 vistas), cat −1,05 dB (GaussianShader melhor em 1/32), teapot +0,75 dB (28/32).

## Leitura
- Qualidade depende da cena: o GaussianShader vence bell e teapot, e o GOF + BRDF vence cat.
- Custo: o GaussianShader leva 1,7 a 2,7 vezes o tempo, de 2,4 a 3,9 vezes a energia (potencia media 68 W contra 48 W) e o dobro de VRAM.
- Na cat o GOF + BRDF entrega mais qualidade com menos de 40% do tempo e da energia.
- O render das 32 vistas leva cerca de 22 s no GOF e 50 s no GaussianShader.

## Limites
- Nao foi treinado o GOF sem BRDF nestas cenas, entao nao ha como atribuir ao termo especular a diferenca entre os metodos.
- Os hiperparametros do GOF + BRDF nao foram ajustados para estas cenas (SH grau 0, luz no referencial do mundo).
- Uma semente por execucao; diferencas abaixo de cerca de 0,1 dB nao sao interpretaveis.
- Nao ha metricas de geometria (malha) neste conjunto.
- O primeiro treino de cat com o GaussianShader foi perdido por suspensao do computador e refeito; o tempo reportado e o da repeticao.

Arquivos: `docs/experiments/NERO/` (`resumo.csv`, `metricas_<cena>.json`, `qualidade.png`, `custo.png`, `diferenca_psnr_por_vista.png`, `potencia.png`).
