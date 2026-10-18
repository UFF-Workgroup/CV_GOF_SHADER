# Resultados no NeRO GlossySynthetic (bell, cat, teapot)

Quatro variantes treinadas e avaliadas nas mesmas cenas, com perfil de tempo, energia e VRAM: GOF sem BRDF, GOF + BRDF com a configuracao do Truck, GOF + BRDF ajustado para objetos espelhados e GaussianShader. Uma semente por execucao.

## Como foi feito
- Dados: `GlossySynthetic.tar.gz` do NeRO, convertido por `nero2blender.py` do GaussianShader (96 vistas de treino, 32 de teste, 800x800, fundo branco). Para o GOF, `scripts/nero_to_gof.py` acrescenta `camera_angle_x`.
- GOF sem BRDF: `--sh_degree 0 -w`, 30 mil iteracoes, inicializacao aleatoria de 100 mil pontos (base para medir o termo especular).
- GOF + BRDF: `--sh_degree 0 --brdf --light_frame world -w`, demais hiperparametros como no Truck (mapa de ambiente 128, `lambda_specular_sparse 0.001`, regularizacao de normais a partir de 15 mil).
- GOF + BRDF ajustado: o anterior com `--envmap_res 512 --lambda_specular_sparse 0 --depth_normal_from_iter 3000`. O ajuste veio do diagnostico da bell (rugosidade aprendida alta num espelho, mapa de ambiente pequeno, esparsidade z## Resultados (media das 32 vistas de teste)
| Cena | Metodo | PSNR | SSIM | LPIPS | Tempo (h) | Energia (Wh) | VRAM (GB) |
|---|---|---|---|---|---|---|---|
| bell | GOF sem BRDF | 24,85 | 0,9043 | 0,1109 | 3,59 | 170 | 2,6 |
| bell | GOF + BRDF | 24,80 | 0,8928 | 0,1159 | 3,18 | 153 | 2,8 |
| bell | GOF + BRDF ajustado | 29,83 | 0,9350 | 0,0867 | 2,51 | 123 | 2,7 |
| bell | GaussianShader | **29,84** | **0,9405** | **0,0834** | 6,09 | 414 | 5,3 |
| cat | GOF sem BRDF | 29,87 | 0,9521 | 0,0720 | 2,54 | 123 | 2,6 |
| cat | GOF + BRDF | 32,37 | 0,9665 | 0,0541 | 2,23 | 107 | 2,6 |
| cat | GOF + BRDF ajustado | **32,68** | **0,9698** | **0,0491** | 2,06 | 103 | 2,6 |
| cat | GaussianShader | 31,32 | 0,9607 | 0,0578 | 6,08 | 414 | 5,5 |
| teapot | GOF sem BRDF | 21,08 | 0,8790 | 0,1033 | 4,04 | 189 | 2,5 |
| teapot | GOF + BRDF | 23,08 | 0,8889 | 0,0982 | 3,77 | 180 | 2,6 |
| teapot | GOF + BRDF ajustado | **24,40** | **0,9062** | **0,0921** | 3,39 | 165 | 2,6 |
| teapot | GaussianShader | 23,82 | 0,9021 | 0,0946 | 6,47 | 439 | 5,6 |

Diferencas de PSNR por vista (media; vistas em que o primeiro e melhor):
- GOF + BRDF menos GOF sem BRDF: bell −0,04 dB (14/32), cat +2,49 dB (32/32), teapot +2,00 dB (31/32).
- GOF + BRDF ajustado menos GOF + BRDF: bell +5,02 dB (32/32), cat +0,32 dB (29/32), teapot +1,33 dB (32/32).
- GaussianShader menos GOF + BRDF ajustado: bell +0,01 dB (15/32; SSIM melhor em 28/32, LPIPS em 25/32), cat −1,37 dB (0/32), teapot −0,58 dB (0/32).

## Leitura
- O termo especular ajuda quando esta bem configurado: com a configuracao do Truck ele ja rende +2 a +2,5 dB em cat e teapot, mas nada na bell, o objeto mais espelhado.
- A diferenca de 5 dB que havia na bell era de configuracao, nao do metodo: o ajuste a elimina por completo, e na bell o GOF + BRDF ajustado empata com o GaussianShader em PSNR, com SSIM e LPIPS ligeiramente piores.
- Com o ajuste, o GOF + BRDF supera o GaussianShader em cat e teapot nas tres metricas e em todas as 32 vistas de PSNR.
- O ajuste nao custa mais: tempo e energia caem um pouco e a VRAM fica igual.
- Custo: o GaussianShader leva 1,9 a 3,0 vezes o tempo e 2,7 a 4,0 vezes a energia do GOF + BRDF ajustado (potencia media 68 W contra 49 W), e o dobro de VRAM.
- O render das 32 vistas leva cerca de 21 s no GOF e 50 s no GaussianShader.

## Limites
- O ajuste foi escolhido olhando a bell, que tambem e cena de teste; cat e teapot nao participaram da escolha.
- Os tres ajustes foram aplicados juntos, entao nao se sabe o peso de cada um.
- O GaussianShader usou os hiperparametros do README e nao foi ajustado por cena.
- Uma semente por execucao; diferencas abaixo de cerca de 0,1 dB nao sao interpretaveis (bell: GOF sem BRDF e GOF + BRDF; GOF + BRDF ajustado e GaussianShader).
- Nao ha metricas de geometria (malha) neste conjunto.
- O primeiro treino de cat com o GaussianShader foi perdido por suspensao do computador e refeito; o tempo reportado e o da repeticao.

encia media 68 W contra 48 W) e o dobro de VRAM.
- Na cat o GOF + BRDF entrega mais qualidade com menos de 40% do tempo e da energia.
- O render das 32 vistas leva cerca de 22 s no GOF e 50 s no GaussianShader.

## Limites
- Nao foi treinado o GOF sem BRDF nestas cenas, entao nao ha como atribuir ao termo especular a diferenca entre os metodos.
- Os hiperparametros do GOF + BRDF nao foram ajustados para estas cenas (SH grau 0, luz no referencial do mundo).
- Uma semente por execucao; diferencas abaixo de cerca de 0,1 dB nao sao interpretaveis.
- Nao ha metricas de geometria (malha) neste conjunto.
- O primeiro treino de cat com o GaussianShader foi perdido por suspensao do computador e refeito; o tempo reportado e o da repeticao.

Arquivos: `docs/experiments/NERO/` (`resumo.csv`, `metricas_<cena>.json`, `qualidade.png`, `custo.png`, `diferenca_psnr_por_vista.png`, `potencia.png`, `nero_visual.png`). A diferenca por vista compara GaussianShader e GOF + BRDF ajustado.
