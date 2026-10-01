"""Compara pastas de predicoes contra o mesmo GT com PSNR/SSIM/LPIPS, mesma metrica do metrics.py."""
import csv
import itertools
import json
import os
import sys
from argparse import ArgumentParser

import lpips
import torch
import torchvision.transforms.functional as tf
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.image_utils import psnr
from utils.loss_utils import ssim


def index_dir(path):
    """Mapeia o indice da vista (prefixo antes de '_' no nome) para o arquivo."""
    files = {}
    for name in sorted(os.listdir(path)):
        if name.lower().endswith((".png", ".jpg", ".jpeg")):
            key = os.path.splitext(name)[0].split("_")[0]
            if key in files:
                raise ValueError(f"{path}: indice duplicado '{key}'")
            files[key] = os.path.join(path, name)
    return files


def load(path, device):
    return tf.to_tensor(Image.open(path).convert("RGB")).unsqueeze(0).to(device)


def parse_method(spec):
    if "=" not in spec:
        raise ValueError(f"--method espera NOME=PASTA, recebeu '{spec}'")
    name, path = spec.split("=", 1)
    return name, path


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--gt", required=True, help="pasta com as imagens ground truth")
    parser.add_argument("--method", "-m", action="append", required=True,
                        help="NOME=PASTA com as predicoes; repetir para cada metodo")
    parser.add_argument("--out", default="comparison", help="pasta de saida")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lpips_fn = lpips.LPIPS(net="vgg").to(device)

    gt_files = index_dir(args.gt)
    methods = dict(parse_method(m) for m in args.method)
    method_files = {name: index_dir(path) for name, path in methods.items()}
    for name, files in method_files.items():
        if set(files) != set(gt_files):
            missing = sorted(set(gt_files) - set(files))[:5]
            extra = sorted(set(files) - set(gt_files))[:5]
            raise SystemExit(f"'{name}' nao casa com o GT (faltam {missing}, sobram {extra})")

    keys = sorted(gt_files)
    per_view = {name: {} for name in methods}
    with torch.no_grad():
        for key in keys:
            gt = load(gt_files[key], device)
            for name, files in method_files.items():
                pred = load(files[key], device)
                if pred.shape != gt.shape:
                    raise SystemExit(f"'{name}' vista {key}: {tuple(pred.shape)} != GT {tuple(gt.shape)}")
                per_view[name][key] = {
                    "PSNR": psnr(pred, gt).mean().item(),
                    "SSIM": ssim(pred, gt).item(),
                    "LPIPS": lpips_fn(pred, gt).item(),
                }

    metrics = ("PSNR", "SSIM", "LPIPS")
    means = {name: {m: sum(v[m] for v in views.values()) / len(views) for m in metrics}
             for name, views in per_view.items()}

    pairs = {}
    for a, b in itertools.combinations(methods, 2):
        delta = {m: [per_view[b][k][m] - per_view[a][k][m] for k in keys] for m in metrics}
        pairs[f"{b} - {a}"] = {
            m: {"media": sum(d) / len(d),
                "vistas_b_melhor": sum(x > 0 if m != "LPIPS" else x < 0 for x in d),
                "vistas": len(d)}
            for m, d in delta.items()
        }

    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "results.json"), "w") as f:
        json.dump({"medias": means, "pares": pairs, "n_vistas": len(keys)}, f, indent=2)
    with open(os.path.join(args.out, "per_view.json"), "w") as f:
        json.dump(per_view, f, indent=2)
    with open(os.path.join(args.out, "per_view.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["vista"] + [f"{n}_{m}" for n in methods for m in metrics])
        for k in keys:
            w.writerow([k] + [f"{per_view[n][k][m]:.6f}" for n in methods for m in metrics])

    print(f"{len(keys)} vistas")
    print(f"{'metodo':<24}{'PSNR':>10}{'SSIM':>10}{'LPIPS':>10}")
    for name, m in means.items():
        print(f"{name:<24}{m['PSNR']:>10.3f}{m['SSIM']:>10.4f}{m['LPIPS']:>10.4f}")
    for pair, res in pairs.items():
        print(f"\n{pair}: " + ", ".join(
            f"{m} {res[m]['media']:+.4f} ({res[m]['vistas_b_melhor']}/{res[m]['vistas']} vistas)" for m in metrics))


if __name__ == "__main__":
    main()
