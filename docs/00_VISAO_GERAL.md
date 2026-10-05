# GOF + Sombreamento Especular — Visão Geral

Integração do sombreamento fisicamente baseado do **GaussianShader** na arquitetura
volumétrica do **Gaussian Opacity Fields (GOF)**, sem modificar o rasterizador. A motivação
original era reconstrução de gêmeos digitais de **testemunhos de rocha**; desde 2026-09-23
(ADR-011) o escopo ativo está restrito a **Truck/Tanks&Temples**, por falta de autorização
de uso do dataset de rocha — a contribuição passou a ser metodológica: a fusão GOF+BRDF em
si, validada com múltiplas seeds, ablações e geometria numa cena pública. A motivação de
rocha permanece como trabalho futuro condicionado a autorização (ver ADR-011).

---

## Por onde começar

| Se você quer… | Leia |
|---|---|
| Entender o método do zero, sem conhecer o repo | [`01_FUNDAMENTOS.md`](01_FUNDAMENTOS.md) |
| Saber **por que** cada decisão foi tomada | [`02_DECISOES.md`](02_DECISOES.md) (ADRs) |
| Verificar a matemática | [`03_FORMULACAO.md`](03_FORMULACAO.md) |
| Rodar experimentos | [`04_PROTOCOLO.md`](04_PROTOCOLO.md) |
| Ver os números medidos | [`05_EXPERIMENTOS.md`](05_EXPERIMENTOS.md) |
| Saber o que estava quebrado no código herdado | [`06_AUDITORIA.md`](06_AUDITORIA.md) |
| Saber onde o método falha | [`07_LIMITACOES.md`](07_LIMITACOES.md) |
| Consultar as fórmulas dos artigos-fonte, com equação/seção/página | [`08_ARTIGOS_REFERENCIA.md`](08_ARTIGOS_REFERENCIA.md) |
| Entender tudo do zero, sem ler os outros documentos primeiro | [`09_RESUMO_EXECUTIVO.md`](09_RESUMO_EXECUTIVO.md) |
| O plano completo original | [`00_PLANO_MESTRE.md`](00_PLANO_MESTRE.md) |

`ARQUIVO_GUIA_DA_APLICACAO_v1.md` é o guia anterior, **arquivado e supersedido** — suas
Fases 3–5 foram canceladas.

---

## As três contribuições

**1. O rasterizador não é modificado.** O GOF já expõe um caminho de cor por-Gaussiana
(`colors_precomp`) cujo gradiente já era devolvido pelo backward em CUDA. Todo o BRDF vive
em PyTorch e o autograd cuida das derivadas. Zero derivada escrita à mão — que é a classe
de erro mais cara e mais difícil de detectar nesse tipo de trabalho. (ADR-001)

**2. Environment map ancorado no espaço de vista, para mesa giratória.** Na captura, o
objeto gira e a luz fica parada na sala; no referencial que o COLMAP reconstrói, a luz gira
junto com a câmera. Um envmap ancorado no mundo — o que a literatura assume — é
fisicamente inválido nesse setup. Provamos que consultar o mapa na direção de reflexão em
**espaço de vista** é *exato*, sem precisar estimar o eixo da mesa nem o ângulo por imagem.
(ADR-002, `03_FORMULACAO.md` §6)

**3. O residual view-dependent são as SH que já existiam.** $c_r(\omega_o)$ é, por
definição, uma função da direção de vista — exatamente o que a base SH do 3DGS já
representa: grau 0 é o difuso, graus ≥1 são o residual. Isso corrige a física, elimina um
tensor redundante, e transforma o grau de SH em pergunta científica: *com o especular
explícito, de quanta SH ainda se precisa?* (ADR-003)

---

## Estado

| Fase | Estado |
|---|---|
| 0 · Migração e rastreabilidade | concluída |
| 1 · Correções da auditoria | concluída — 3 bugs críticos silenciosos |
| 2 · Reversão da plumbagem CUDA | concluída — rasterizador == GOF upstream |
| 3 · Módulo de sombreamento | concluída |
| 4 · Perdas e regularização | concluída |
| 5 · Otimização/instrumentação de VRAM | concluída |
| Testes | 36 passando (5 bugs silenciosos corrigidos — ver `06_AUDITORIA.md`) |
| Validação ponta a ponta | B1 e E1 concluídos em Truck (1 seed, +0,41 dB PSNR) — ver `05_EXPERIMENTOS.md` |
| Escopo | restrito a Truck/Tanks&Temples desde 2026-09-23 (ADR-011) |
| Documentação | concluída |

**Próximo passo:** repetir E1 (e B1, se necessário) com ≥2 seeds, rodar as ablações E3–E7
sobre E1 e viabilizar a comparação geométrica (Chamfer/F1) — ver `04_PROTOCOLO.md` e
`09_RESUMO_EXECUTIVO.md` §6.

---

## Duas coisas que quem entrar no projeto precisa saber

**As extensões são instalações editáveis.** Se apontarem para outra árvore, o Python
importa código antigo **sem nenhum aviso**. Rode a verificação de ambiente do
`04_PROTOCOLO.md` antes de confiar em qualquer resultado.

**O treino não é bit-reprodutível.** `atomicAdd` no rasterizador. Diferenças de PSNR
abaixo de ~0,1 dB não são interpretáveis sem repetição por seed.
