# CV_GOF_SHADER

Sombreamento especular fisicamente baseado (**GaussianShader**) dentro da arquitetura
volumétrica do **Gaussian Opacity Fields**, para reconstrução de gêmeos digitais de
**testemunhos de rocha**.

Fork de [`autonomousvision/gaussian-opacity-fields`](https://github.com/autonomousvision/gaussian-opacity-fields)
(SIGGRAPH Asia 2024), com a formulação de sombreamento de
[`asparagus15/GaussianShader`](https://github.com/asparagus15/GaussianShader) (CVPR 2024).

📖 **Comece por [`docs/00_VISAO_GERAL.md`](docs/00_VISAO_GERAL.md).**

---

## As três contribuições

1. **O rasterizador não é modificado.** Todo o BRDF vive em PyTorch, entregue ao GOF pelo
   caminho `colors_precomp`, que já era diferenciável. Zero derivada escrita à mão.
2. **Environment map ancorado no espaço de vista**, correção exata para captura em mesa
   giratória — onde o envmap de mundo assumido pela literatura é fisicamente inválido.
3. **O residual view-dependent são as SH que já existiam**: grau 0 é o difuso, graus ≥1 são
   $c_r(\omega_o)$. Corrige a física e transforma o grau de SH em pergunta científica.

Detalhes e justificativas em [`docs/02_DECISOES.md`](docs/02_DECISOES.md).

---

## Uso

```bash
conda activate gof

# Baseline (GOF puro)
python train.py -s <fonte> -m output/<RUN_ID> -r 2 --eval

# Com sombreamento BRDF
python train.py -s <fonte> -m output/<RUN_ID> -r 2 --eval \
    --sh_degree 1 --brdf --light_frame view --brdf_from_iter 3000

# Avaliação — NAO repetir os flags de BRDF: vêm do cfg_args
python render.py  -m output/<RUN_ID>
python metrics.py -m output/<RUN_ID>
python extract_mesh.py -m output/<RUN_ID> --iteration 30000
```

Para runs oficiais use `scripts/run_experiment.sh <RUN_ID> <args>`, que verifica o ambiente
e carimba commit, config e hardware em `docs/experiments/<RUN_ID>/`.

### Flags do BRDF

| Flag | Default | O que faz |
|---|---|---|
| `--brdf` | off | liga a equação de sombreamento |
| `--light_frame` | `view` | `view` = mesa giratória · `world` = câmera orbita |
| `--light_repr` | `envmap` | `envmap` pré-filtrado · `sh` baixa frequência |
| `--no_fresnel` | off | ablação: desliga o Fresnel de Schlick |
| `--use_normal_residual` | off | ablação: liga o resíduo $\Delta n$ |
| `--use_tonemap` | off | linear→sRGB (Marco 2, relighting HDR) |
| `--brdf_from_iter` | 3000 | warm-up antes de ativar o especular |

---

## Testes

```bash
pytest tests/ -q     # 25 testes; precisa de GPU
```

Dois são bloqueantes: **T1** (o BRDF reduz exatamente ao baseline quando o especular é
zero — é o que torna a ablação interpretável) e **T3** (o invariante de vínculo com o
otimizador, que protege contra a classe de bug descrita em
[`docs/06_AUDITORIA.md`](docs/06_AUDITORIA.md)).

---

## Duas armadilhas do ambiente

**As extensões CUDA são instalações editáveis.** Se apontarem para outra árvore, o Python
importa código antigo **sem nenhum aviso**. Verifique antes de confiar em qualquer número:

```bash
python -c "
import torch, simple_knn, diff_gaussian_rasterization as d, tetranerf
for m in (simple_knn, d, tetranerf): assert 'CV_GOF_SHADER' in m.__file__, m.__file__
print('ambiente OK')"
```

**O treino não é bit-reprodutível.** `atomicAdd` no rasterizador faz a ordem de acumulação
dos gradientes variar. Seeds fixas não bastam: diferenças de PSNR abaixo de ~0,1 dB não são
interpretáveis sem repetição por seed.

---

## Hardware de referência

RTX 3050 **6 GB** · CUDA 11.8 · PyTorch 2.0.1 · Python 3.9. Os 6 GB são a restrição
dominante e moldam o default de `--sh_degree`; ver
[`docs/04_PROTOCOLO.md`](docs/04_PROTOCOLO.md).

## Licença

Herda os termos dos projetos originais — uso não comercial, de pesquisa e avaliação.
Ver [`LICENSE.md`](LICENSE.md).
