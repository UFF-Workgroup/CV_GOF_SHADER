"""Relighting: renderiza as mesmas vistas sob environment maps substituidos.

Este e o passo 6 da verificacao de ponta a ponta do plano mestre (secao 9). Serve a duas
finalidades distintas, e vale nao confundi-las:

1. **Demonstracao.** Se o material aprendido tem algum significado fisico, trocar a luz
   deve mudar os reflexos *sem* mudar o albedo. E a figura qualitativa do artigo.

2. **Diagnostico.** Renderizar sob luz uniforme e o teste mais direto de quanto da imagem
   o modelo atribui ao especular. Se a imagem sob luz uniforme for indistinguivel da
   renderizada com o mapa aprendido, o ramo especular nao esta fazendo trabalho nenhum --
   e nenhuma metrica de NVS revelaria isso sozinha.

Todos os mapas sao procedurais: nenhuma dependencia externa e nenhum HDR para baixar.

    python scripts/relight.py -m output/EXP-20260811-02-b1 --views 0 12 24

    python scripts/relight.py --dry_run          # so gera os mapas, nao toca na GPU

Saida em docs/experiments/<RUN_ID>/relight/ por padrao, onde RUN_ID e o nome do
diretorio do modelo.
"""
import math
import os
import sys
from argparse import ArgumentParser

import torch
import torchvision

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))


def equirect_directions(height):
    """Grade de direcoes unitarias [3,H,2H] na convencao de scene/lighting.py.

    Precisa ser a inversa exata de `direction_to_equirect_uv`, senao os mapas sinteticos
    aparecem rotacionados ou espelhados em relacao ao aprendido e a comparacao engana.
    La: u = atan2(x,-z)/2pi + 0.5 e v = acos(y)/pi.
    """
    width = height * 2
    v = (torch.arange(height, dtype=torch.float32) + 0.5) / height
    u = (torch.arange(width, dtype=torch.float32) + 0.5) / width
    theta = v * math.pi                      # angulo polar a partir de +y
    phi = (u - 0.5) * 2.0 * math.pi          # = atan2(x, -z)

    sin_t = torch.sin(theta)[:, None]
    y = torch.cos(theta)[:, None].expand(height, width)
    x = sin_t * torch.sin(phi)[None, :]
    z = -sin_t * torch.cos(phi)[None, :]
    return torch.stack([x, y, z], dim=0)


def _blob(dirs, direction, sharpness, color):
    """Fonte de luz gaussiana em torno de `direction`, sobre a esfera."""
    d = torch.tensor(direction, dtype=torch.float32)
    d = d / d.norm()
    cos = (dirs * d[:, None, None]).sum(0).clamp(-1.0, 1.0)
    falloff = torch.exp(-(1.0 - cos) * sharpness)
    return torch.tensor(color, dtype=torch.float32)[:, None, None] * falloff[None]


def build_envmaps(height=128):
    """Baterias de iluminacao sinteticas, como {nome: [3,H,2H]}.

    Escolhidas para variar em UMA dimensao por vez: direcao (lateral), numero de fontes
    (dupla), frequencia espacial (ceu/chao e suave; lateral e aguda) e presenca de
    estrutura (uniforme nao tem nenhuma).
    """
    dirs = equirect_directions(height)
    maps = {}

    # Luz de estudio: uma fonte quente acima e a esquerda, com preenchimento fraco.
    maps["estudio_lateral"] = (
        _blob(dirs, (-1.0, 0.6, -0.4), sharpness=12.0, color=(1.0, 0.93, 0.80)) * 3.0
        + 0.06
    )

    # Duas fontes opostas e de temperaturas diferentes: separa o que e reflexo do que e
    # albedo melhor que qualquer fonte unica, porque os dois reflexos se movem juntos.
    maps["contraluz_dupla"] = (
        _blob(dirs, (1.0, 0.3, 0.2), sharpness=18.0, color=(1.0, 0.85, 0.65)) * 2.5
        + _blob(dirs, (-1.0, 0.1, -0.3), sharpness=18.0, color=(0.60, 0.75, 1.0)) * 2.0
        + 0.05
    )

    # Ceu/chao: baixa frequencia pura. Um material rugoso deve responder a este mapa quase
    # como ao uniforme; um material polido, nao.
    y = dirs[1]
    t = (y * 0.5 + 0.5).clamp(0.0, 1.0)
    ceu = torch.tensor([0.45, 0.62, 1.00])[:, None, None] * t[None]
    chao = torch.tensor([0.30, 0.24, 0.18])[:, None, None] * (1.0 - t)[None]
    maps["ceu_chao"] = ceu + chao

    # Uniforme: o controle negativo. Sem estrutura, o termo especular vira uma constante.
    maps["uniforme"] = torch.ones(3, height, height * 2)

    return maps


def match_mean(source, target_mean):
    """Reescala um mapa para a mesma radiancia media do aprendido.

    Sem isto a comparacao fica confundida por brilho: um mapa mais luminoso produz uma
    imagem mais clara e a diferenca seria lida como "efeito da iluminacao" quando e so
    exposicao. Igualar a media deixa a *distribuicao* da luz como unica variavel.
    """
    mean = source.mean().clamp(min=1e-8)
    return source * (target_mean / mean)


def save_envmap_preview(tensor, path):
    torchvision.utils.save_image(tensor.clamp(0.0, 1.0), path)


def main():
    parser = ArgumentParser(description="Relighting com envmaps substituidos")
    from arguments import ModelParams, PipelineParams, get_combined_args

    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--views", nargs="+", type=int, default=[0, 12, 24],
                        help="indices das cameras de teste a renderizar")
    parser.add_argument("--split", default="test", choices=["test", "train"])
    parser.add_argument("--out", default=None,
                        help="default: docs/experiments/<RUN_ID>/relight/")
    parser.add_argument("--envmap_height", default=128, type=int)
    parser.add_argument("--dry_run", action="store_true",
                        help="so escreve os envmaps sinteticos; nao carrega o modelo")
    parser.add_argument("--quiet", action="store_true")

    if "--dry_run" in sys.argv:
        args = parser.parse_args()
        out = args.out or "docs/experiments/_relight_dry_run"
        os.makedirs(out, exist_ok=True)
        maps = build_envmaps(args.envmap_height)
        for name, m in maps.items():
            save_envmap_preview(match_mean(m, 0.5), os.path.join(out, f"envmap_{name}.png"))
            print(f"  {name:18s} media={m.mean():.4f}  max={m.max():.4f}  shape={tuple(m.shape)}")
        print(f"[relight] dry run: {len(maps)} mapas em {out}")
        return

    args = get_combined_args(parser)
    dataset = model.extract(args)
    pipe = pipeline.extract(args)

    if not getattr(dataset, "brdf", False):
        print("[relight] ERRO: este modelo foi treinado SEM --brdf. Nao ha material nem "
              "iluminacao para substituir; relighting nao se aplica.", file=sys.stderr)
        sys.exit(1)

    from scene import Scene
    from scene.lighting import EnvironmentMap
    from gaussian_renderer import GaussianModel, render
    from utils.general_utils import safe_state

    safe_state(args.quiet)

    run_id = os.path.basename(os.path.normpath(dataset.model_path))
    out_root = args.out or os.path.join("docs", "experiments", run_id, "relight")
    os.makedirs(out_root, exist_ok=True)

    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=args.iteration, shuffle=False)
        cameras = scene.getTestCameras() if args.split == "test" else scene.getTrainCameras()
        if len(cameras) == 0:
            print(f"[relight] ERRO: split '{args.split}' vazio. Treinou com --eval?",
                  file=sys.stderr)
            sys.exit(1)

        bg = torch.tensor([1, 1, 1] if dataset.white_background else [0, 0, 0],
                          dtype=torch.float32, device="cuda")

        learned = gaussians.lighting
        if learned is None:
            print("[relight] ERRO: modelo sem iluminacao carregada (lighting.pth ausente).",
                  file=sys.stderr)
            sys.exit(1)
        learned_mean = float(learned.as_image().mean())
        print(f"[relight] radiancia media do mapa aprendido: {learned_mean:.4f}")

        # O aprendido entra como controle na mesma grade de figuras: sem ele nao da para
        # dizer se uma diferenca vem da luz nova ou de o modelo simplesmente ser ruim.
        conditions = {"aprendido": None}
        for name, m in build_envmaps(args.envmap_height).items():
            conditions[name] = match_mean(m, learned_mean)

        preview_dir = os.path.join(out_root, "envmaps")
        os.makedirs(preview_dir, exist_ok=True)
        save_envmap_preview(learned.as_image().cpu(),
                            os.path.join(preview_dir, "aprendido.png"))

        indices = [i for i in args.views if 0 <= i < len(cameras)]
        skipped = sorted(set(args.views) - set(indices))
        if skipped:
            print(f"[relight] AVISO: vistas fora do intervalo ignoradas: {skipped} "
                  f"({len(cameras)} cameras em '{args.split}')")

        for name, tensor in conditions.items():
            if tensor is None:
                gaussians.lighting = learned
            else:
                # Substituir o modulo inteiro (em vez de so os pesos) faz o relighting
                # funcionar mesmo se o treino usou SHLighting: `sample()` e a unica
                # interface que get_shading_colors consome.
                sub = EnvironmentMap(resolution=args.envmap_height, init_std=0.0).cuda()
                sub.base.data.copy_(tensor.cuda())
                gaussians.lighting = sub
                save_envmap_preview(tensor, os.path.join(preview_dir, f"{name}.png"))

            cond_dir = os.path.join(out_root, name)
            os.makedirs(cond_dir, exist_ok=True)
            for i in indices:
                img = render(cameras[i], gaussians, pipe, bg,
                             kernel_size=dataset.kernel_size, brdf_args=dataset)["render"]
                torchvision.utils.save_image(
                    img[:3].clamp(0.0, 1.0),
                    os.path.join(cond_dir, f"{args.split}_{i:05d}.png"))
            print(f"[relight] {name:18s} -> {cond_dir}")

        gaussians.lighting = learned

    print(f"[relight] concluido. Compare visualmente 'aprendido' com 'uniforme': se forem "
          f"iguais, o ramo especular nao esta contribuindo.")


if __name__ == "__main__":
    main()
