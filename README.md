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

## Objetos brilhantes (NeRO GlossySynthetic)

Benchmark contra o GaussianShader em cenas sintéticas espelhadas, com tempo, energia e VRAM
medidos por treino (`nvidia-smi` a cada 2 s). Requer um clone do
[GaussianShader](https://github.com/Asparagus15/GaussianShader) com seu próprio ambiente.

```bash
# 1. Dados: GlossySynthetic do NeRO convertido pelo nero2blender.py do GaussianShader
#    (precisa de configs/synthetic_split_128.pkl do NeRO); depois a versão lida pelo GOF
python scripts/nero_to_gof.py --src <GlossySynthetic>/bell_blender --dst <GlossySynthetic>/bell_gof

# 2. Treinos em fila serial (pula os já concluídos); GS_DIR e PY_GS só para o GaussianShader
export NERO_DATA=<GlossySynthetic> GS_DIR=<clone do GaussianShader> PY_GS=<python do ambiente gs>
METODOS="gof gof_brdf gof_brdf_ajustado gaussianshader" ./scripts/queue_nero.sh bell cat teapot

# 3. Render das vistas de teste, métricas e gráficos
METODOS="gof gof_brdf gof_brdf_ajustado gaussianshader" ./scripts/render_nero.sh bell cat teapot
python scripts/compare_preds.py --gt output/nero/bell/gof/test/ours_30000/gt_-1 \
    -m GOF=output/nero/bell/gof/test/ours_30000/test_preds_-1 \
    -m GOF+BRDF=output/nero/bell/gof_brdf/test/ours_30000/test_preds_-1 \
    -m "GOF+BRDF ajustado=output/nero/bell/gof_brdf_ajustado/test/ours_30000/test_preds_-1" \
    -m GaussianShader=output/nero/bell/gaussianshader/test/ours_30000/renders \
    --out output/nero/bell/metricas
python scripts/nero_report.py --root output/nero --out output/nero/relatorio
```

`gof_brdf` usa a configuração do Truck. `gof_brdf_ajustado` acrescenta
`--envmap_res 512 --lambda_specular_sparse 0 --depth_normal_from_iter 3000`, para que o
reflexo de superfícies espelhadas não seja borrado nem zerado pela regularização. Variáveis
opcionais: `PY_GOF` (padrão `python`), `CUDA_HOME` (padrão `/usr/local/cuda-11.8`), `ITERS`
(padrão 30000) e `OUT` (padrão `output/nero`). `nero_report.py` espera as quatro variantes
nas três cenas.

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
