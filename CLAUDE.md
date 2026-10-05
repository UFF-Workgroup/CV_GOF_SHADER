# Organizacao de repositorios e branches

## Remotos
- `origin` = **DEV**: `git@github.com:UFF-Workgroup/CV_GOF_SHADER.git` (estudo/desenvolvimento, com `docs/`).
- `upstream` = **PRD**: `https://github.com/medialab-uff/CV_GOF_SHADER.git` (versao organizada para publicacao, **sem** `docs/`).

## Branches locais
- `feat/brdf-especular` (rastreia `origin`): branch de **desenvolvimento**. Tem `docs/`, logs e referencias a ele. Todo trabalho comeca aqui.
- `prd` (rastreia `upstream/main`): **unico ponto de contato com o PRD**. Nunca contem `docs/`; seu codigo nao pode referenciar `docs/`.
- `main` local: espelha `origin/main` ("Entrega final CV"); nao mexer.

## Regras
- O PRD comecou com um unico commit limpo (`be41a77`). Mensagens de commit curtas, sem coautor.
- Levar mudanca da dev ao PRD com `git cherry-pick` dos commits de codigo, a partir da `prd`. **Nunca fazer merge** da dev na `prd`: traria `docs/` de volta e conflitaria com as limpezas.
- Cherry-pick de commit que mexe em `docs/` ou cita `docs/` no codigo/README/scripts: ajustar para que a `prd` fique sem nenhuma referencia (`git grep -n "docs/"` deve vir vazio, salvo URLs externas).
- Na `prd`, metadados de run ficam em `output/<RUN_ID>/meta/` (nao em `docs/experiments/`).
- Imagens e resultados (`output/`, `release/`) nunca entram no git; sao enviados manualmente.
- Submodulos (`submodules/`) ficam vendored como pastas comuns, nao como git submodule.
- Push para `origin` e para `upstream`, e qualquer force push, so quando o usuario pedir. Force push no PRD com `--force-with-lease` apontando o commit esperado.
- Comentarios no codigo: no maximo uma frase.
- Este arquivo existe so na dev; nao levar para a `prd`.
