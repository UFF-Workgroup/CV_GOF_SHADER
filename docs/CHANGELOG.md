# Changelog

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
