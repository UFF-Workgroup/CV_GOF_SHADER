# Manuscrito v0: preprint para revisão interna

`gof_brdf_preprint.pdf` é a versão v0 do artigo, em português, sobre o template da família *Science* que veio em `../GOF___GaussianShader.zip`. O original está extraído, sem alterações, em `../template_original/`.

## Estrutura

| Caminho | Conteúdo |
|---|---|
| `gof_brdf_preprint.tex` | texto do manuscrito; o nome muda porque o readme do template exige |
| `referencias.bib` | gerado de `referencias/verificacao.json` por `scripts/gerar_bib.py`, não editar à mão |
| `REFERENCIAS.md` | catálogo para revisão: identificador, status da verificação, onde é citada, o que sustenta no texto e divergências do levantamento do Consensus |
| `referencias/` | `candidatas.json` (entrada), `verificacao.json` (Crossref/arXiv), `resumos.json` (resumos oficiais) |
| `dados/` | todos os números usados, copiados ou derivados das saídas dos treinos por `scripts/extrair_dados.py` |
| `tabelas/` | tabelas LaTeX geradas por `scripts/gerar_tabelas.py`; nenhum número das tabelas é digitado à mão |
| `figuras/` | figuras geradas por `scripts/gerar_figuras.py`, mais o diagrama `pipeline.tex` (TikZ) |
| `build/Dockerfile` | imagem TeX Live usada na compilação (`texlive/texlive:latest-medium` mais `standalone`, `newtx` e outros) |

## Como recompilar

```bash
./compilar.sh                                              # só o PDF
PY_GOF=~/miniconda3/envs/gof/bin/python ./compilar.sh tudo # dados, figuras, tabelas e PDF
```

O LaTeX roda num contêiner Docker, sem instalar nada no sistema. `tudo` lê as saídas dos treinos em `output/nero/`, `output/EXP-*` e `docs/experiments/COMPARACAO-truck/`.

Para reverificar as referências e refazer o catálogo, que precisam de rede:

```bash
python3 scripts/verificar_referencias.py referencias/candidatas.json referencias/verificacao.json
MAILTO=seu@email python3 scripts/buscar_resumos.py referencias/verificacao.json referencias/resumos.json  # MAILTO é opcional
python3 scripts/gerar_bib.py referencias/verificacao.json referencias.bib
python3 scripts/gerar_catalogo.py referencias/verificacao.json referencias/resumos.json gof_brdf_preprint.tex REFERENCIAS.md
```

## Decisões tomadas na v0

- **Idioma:** o texto está em português, como o rascunho original.
- **Pacotes além do template:** foram acrescentados `fontenc`, `babel`, `amsmath`, `booktabs`, `array`, `xcolor`, `lineno` e `hyperref`. Estão marcados com `[preprint]` no preâmbulo, porque o template da *Science* pede para não acrescentar pacotes; numa submissão à revista, rever.
- **Seções numeradas e linhas numeradas:** facilitam a revisão. A *Science* não numera seções.
- **Rascunho original reescrito:** três afirmações dele não se sustentam nos experimentos e foram removidas.
  - "estendeu-se a biblioteca CUDA": na verdade, o rasterizador não foi alterado.
  - "o termo residual demonstrou papel fundamental": todos os experimentos usam SH de grau 0, portanto sem residual.
  - Qualquer resultado em rocha ou minerais: os dados não foram autorizados.
- **Contribuições declaradas de forma restrita:** o levantamento bibliográfico mostra que a parametrização de material já existe no GaussianShader. A contribuição é a integração verificável, a avaliação controlada e o diagnóstico.

## O que falta preencher (marcado em vermelho no PDF)

- autores, afiliação e e-mail;
- repositório público do código;
- agradecimentos e financiamento.

## Pontos para a revisão científica

1. **Uma semente por configuração:** os intervalos de confiança cobrem a variação entre vistas, não entre treinos.
2. **Seleção do ajuste:** o ajuste foi escolhido na *bell*, que também é cena de teste; *cat* e *teapot* servem de evidência fora da amostra.
3. **Ajustes confundidos:** os três ajustes foram aplicados juntos, então não se sabe o peso de cada um.
4. **Esforço de ajuste assimétrico:** ajustamos o nosso método e usamos o GaussianShader na configuração dos autores.
5. **Diagnóstico refutado:** os dados de material não sustentam a hipótese inicial de rugosidade alta, como está na Seção 5.4. Essa hipótese estava na apresentação.
6. **Comparação não controlada:** os números do GaussianShader no Truck vêm de outra máquina e de outra configuração (Tabela S1).
7. **Sem métricas de geometria:** não há nesta versão.
