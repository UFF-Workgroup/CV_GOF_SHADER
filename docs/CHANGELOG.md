# Changelog

## 2026-09-01 (2) — Fix A-5: NaN no polo do envmap, achado na primeira tentativa de E1

**Achado.** A primeira tentativa do run E1 (Truck, `--brdf --light_frame world`) foi a
NaN na iteração 3020, dez iterações depois de `brdf_from_iter=3000` ligar o especular.
Causa: `direction_to_equirect_uv` (`scene/lighting.py`) usa `atan2`/`acos` para mapear
direção 3D em UV do envmap; ambos têm gradiente singular no polo do mapa (`x=z=0`,
`y=±1`) -- `atan2` degenera numa forma `0/0` ali, produzindo NaN direto. Com >1e5
Gaussianas por iteração, bastou uma cair perto o bastante do polo para contaminar o
otimizador inteiro. Ver `06_AUDITORIA.md` A-5.

**Correção.** Direção afastada do polo por epsilon antes de `atan2`/`acos`: viés
desprezível (~1e-4 rad, abaixo da resolução de um texel), gradiente perto do polo passa
a ser grande porém finito. Teste novo de regressão
(`test_a5_envmap_gradient_finite_near_poles`), verificado também com estresse de 2M
direções aleatórias. Suíte: 36/36 passando (era 35).

`scripts/run_experiment.sh` passou a gravar um `.log` por execução (timestampado, sem
sobrescrever tentativas anteriores) tanto em `output/<RUN_ID>/logs/` quanto em
`docs/experiments/<RUN_ID>/` -- a tentativa que crashou com NaN teria seu próprio
arquivo preservado ao lado da retomada, em vez de ser sobrescrita.

## 2026-09-01 — Auditoria de checkpoint: A-4

**Achado.** `capture()`/`restore()` (`scene/gaussian_model.py`) nunca persistiam
`self.lighting` (envmap/SH) nem `_appearance_embeddings`/`appearance_network` — mesma
classe do bug A-2, dormente até agora porque nenhum run executado usou `--brdf`. Retomar
de `--start_checkpoint` num treino com BRDF descartaria silenciosamente o envmap/SH
aprendido, reaplicando o momento do Adam antigo sobre um material reinicializado. Ver
`06_AUDITORIA.md` A-4.

**Correção.** `capture`/`restore` agora cobrem iluminação e embeddings de aparência, com
falha alta (`RuntimeError`) em caso de mismatch de flags entre checkpoint e retomada.
`scripts/run_experiment.sh` passou a injetar `--checkpoint_iterations 10000 20000 30000`
por default. Três testes novos em `tests/test_material_lifecycle.py`. Suíte completa:
35 testes passando (era 25).

## 2026-08-10 — Fases 0 a 5

**Auditoria.** Três bugs críticos e silenciosos no código herdado, todos produzindo formas
corretas e resultado errado: material desconectado do otimizador na poda (congelava a
partir da iteração 600), material ausente de capture/restore e do PLY. Ver `06_AUDITORIA.md`.

**Reversão.** Plumbagem C++/CUDA das fases anteriores removida — era código morto e o
rasterizador voltou a ser idêntico ao GOF upstream (22/24/23 args).

**Sombreamento.** `scene/brdf.py` e `scene/lighting.py`; BRDF inteiro em PyTorch via
`colors_precomp`, com autograd. Envmap equirretangular com mips e iluminação SH com
convolução por rugosidade. Envmap ancorado em espaço de vista para mesa giratória.

**Perdas.** L_sparse e L_reg com warm-up de 3000 iterações.

**Instrumentação.** Pico de VRAM e médias de material no tensorboard.

**Achado.** Rugosidade é não-identificável sob luz uniforme (ADR-008) — relevante para
testemunhos fotografados em caixa de luz difusa.

**Testes.** 25 passando (T1–T7). Smoke test ponta a ponta em Truck: material aprende com
dispersão por-Gaussiana, o que é a prova de produção de que o bug A-1 está corrigido.

**Documentação.** `docs/` completo: fundamentos, ADRs, formulação, protocolo, experimentos,
auditoria, limitações. `scripts/run_experiment.sh` carimba rastreabilidade.
