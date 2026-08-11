# <RUN_ID> — <uma linha dizendo o que este run responde>

Copie para `docs/experiments/<RUN_ID>/README.md`. Os campos de rastreabilidade
(`commit.txt`, `command.txt`, `cfg_args`, `env.txt`, `train.log`) são gerados
automaticamente por `scripts/run_experiment.sh`.

## Pergunta

O que este run decide. Se não houver uma pergunta que possa ser respondida "sim" ou "não"
ao final, o run provavelmente não devia existir.

## Configuração

| | |
|---|---|
| Commit | `<hash>` (de `commit.txt`) |
| Cena | |
| Diferença em relação ao run de referência | **só isto** — se houver mais de uma variável mudando, a comparação não isola nada |
| Seeds | |

## Resultados

| Métrica | Valor | Referência | Δ |
|---|---|---|---|
| PSNR | | | |
| SSIM | | | |
| LPIPS | | | |
| Chamfer / F1 | | | |
| Nº de Gaussianas | | | |
| Pico de VRAM | | | |
| Tempo total | | | |

Saúde do material (se `--brdf`): médias e dispersão de `specular_tint` e `roughness`.
**Dispersão perto de zero significa material não aprendendo** — sintoma do bug A-1.

## Resposta

Responde a pergunta. Se o resultado for negativo, diga; resultado negativo com evidência
é publicável, resultado negativo escondido é dívida.

## Ressalvas

Seja explícito sobre o que este run **não** prova. Lembre:

- O treino **não é bit-reprodutível** (`atomicAdd`). Diferenças de PSNR abaixo de ~0,1 dB
  não são interpretáveis a partir de execução única.
- Se o modelo comparado tem mais parâmetros, parte de qualquer ganho vem de capacidade,
  não de física.
- Configuração de conveniência (`-r 4`, poucas iterações) não extrapola para o regime real.

## Artefatos

`docs/experiments/<RUN_ID>/` — 3 renders de amostra, `metrics.json`, e o que mais permita
outra pessoa refazer a leitura sem re-treinar.
