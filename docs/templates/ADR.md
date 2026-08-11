# ADR-NNN — <título em uma linha, no imperativo ou como afirmação>

**Data:** AAAA-MM-DD · **Status:** proposto | aceito | supersedido por ADR-MMM

## Contexto

O que forçou a decisão. Fatos, não opiniões — número medido, restrição de hardware,
resultado de teste, exigência do artigo. Se a decisão veio de um achado, cite a evidência.

## Opções

1. …
2. …

Liste também a opção que **não** foi escolhida e parecia óbvia — é o que torna o ADR útil
seis meses depois.

## Decisão

Qual opção, em uma frase.

## Consequências

- *Positivas.*
- *Negativas, e declaradas.* Toda decisão tem custo. Um ADR sem consequência negativa
  normalmente é um ADR que não pensou o bastante.
- *O que passa a ser verdade no código.* Arquivos, flags, invariantes.

---

Regras do log:

- **Append-only.** Números de ADR nunca são reusados.
- Reverter uma decisão = escrever um ADR **novo** que a supersede. O original permanece:
  o raciocínio descartado é parte do registro científico.
- Se a decisão for testável, aponte o teste. `02_DECISOES.md` ADR-004 é o exemplo: a
  propriedade afirmada é verificada por `test_t1_*`, não assumida.
