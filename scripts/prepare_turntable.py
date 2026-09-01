"""Preparo da cena de testemunho em roletes: verificacao da captura e imagens de treino.

**A captura.** O testemunho gira sobre dois roletes enquanto a camera fotografa faixa a
faixa, do topo para a base. Dentro de uma faixa a camera fica parada e so o objeto gira;
entre faixas a camera desliza paralela ao eixo do cilindro, **sem mudar de orientacao**.
Essa ultima propriedade e o que sustenta `--light_frame view` (ver `03_FORMULACAO.md` §6):
a derivaçao exige que a rotacao sala->camera seja constante, e translacao nao a altera.

**`check`** confirma essa estrutura a partir dos pixels, nao do relato. Por correlacao de
fase, o passo entre quadros consecutivos e constante dentro de uma faixa (o giro) e
destoa exatamente na troca de faixa (o deslize). Se os passos atipicos nao cairem nas
fronteiras esperadas, a premissa da captura esta errada -- e e melhor descobrir isso em
noventa segundos do que depois de um treino de sete horas.

**`images`** gera as imagens de treino reduzidas a partir de `images/` (a saida nao
distorcida do COLMAP). As fotos originais tem 9504x6336; treinar exige reduzir, e fazer
isso uma vez so evita redecodificar 60 MP a cada run.

    python scripts/prepare_turntable.py check  -s ~/Documentos/rocha_fs16_16cm
    python scripts/prepare_turntable.py images -s ~/Documentos/rocha_fs16_16cm --downscale 8

*Nota sobre mascaras de fundo.* A primeira versao deste script mascarava o fundo por
variancia temporal, pelo raciocinio de que camera parada + objeto girando torna o fundo
estatico um degenerado de paralaxe zero para o SfM. A medicao derrubou a premissa: o
fundo aparece em **um** quadro dos 72 (o 001, com 33% do quadro), e o que parecia fundo
nos demais e a propria borda do cilindro girando para fora de vista. Sem fundo estatico
nao ha o que mascarar. O registro completo esta em ADR-009.
"""
import json
import os
from argparse import ArgumentParser

import cv2
import numpy as np

PREVIEW_WIDTH = 400


def list_frames(folder):
    exts = (".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG")
    return sorted(os.path.join(folder, f) for f in os.listdir(folder) if f.endswith(exts))


def measure_steps(paths, scale=32):
    """Deslocamento entre quadros consecutivos, em fracao do quadro, por correlacao de fase."""
    small = []
    for p in paths:
        im = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        h, w = im.shape
        s = cv2.resize(im, (w // scale, h // scale), interpolation=cv2.INTER_AREA)
        small.append(np.float64(s) - s.mean())
    return np.array([cv2.phaseCorrelate(a, b)[0] for a, b in zip(small, small[1:])]) / \
        np.array([small[0].shape[1], small[0].shape[0]])


def check_faixas(steps, faixa_size, n_frames):
    """Confere que os passos atipicos caem exatamente nas fronteiras de faixa."""
    boundaries = set(range(faixa_size - 1, n_frames - 1, faixa_size))
    inside = np.array([s for i, s in enumerate(steps) if i not in boundaries])
    median = np.median(inside, axis=0)
    spread = np.median(np.linalg.norm(inside - median, axis=1))
    tol = max(4.0 * spread, 0.05)
    outliers = set(int(i) for i in np.where(np.linalg.norm(steps - median, axis=1) > tol)[0])
    return {
        "passo_mediano_dentro_da_faixa": [round(float(v), 4) for v in median],
        "dispersao_dentro_da_faixa": round(float(spread), 4),
        "tolerancia": round(float(tol), 4),
        "fronteiras_esperadas": sorted(int(b) for b in boundaries),
        "passos_atipicos": sorted(outliers),
        "atipicos_fora_das_fronteiras": sorted(outliers - boundaries),
        "fronteiras_sem_passo_atipico": sorted(boundaries - outliers),
        "estrutura_confirmada": not (outliers ^ boundaries),
    }


def write_contact_sheet(path, paths, cols=3):
    tiles = []
    for p in paths:
        im = cv2.imread(p)
        h, w = im.shape[:2]
        t = cv2.resize(im, (PREVIEW_WIDTH, int(PREVIEW_WIDTH * h / w)), interpolation=cv2.INTER_AREA)
        cv2.putText(t, os.path.basename(p), (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        tiles.append(t)
    rows = [np.hstack(tiles[i:i + cols]) for i in range(0, len(tiles) - cols + 1, cols)]
    cv2.imwrite(path, np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 88])


def run_check(args):
    frames = list_frames(os.path.join(args.source_path, args.input_dir))
    if len(frames) % args.faixa_size:
        raise SystemExit(f"{len(frames)} quadros nao e multiplo de faixa_size={args.faixa_size}")

    report = {
        "quadros": len(frames),
        "faixa_size": args.faixa_size,
        "faixas": len(frames) // args.faixa_size,
        "resolucao": list(cv2.imread(frames[0]).shape[1::-1]),
        "verificacao_de_faixas": check_faixas(measure_steps(frames), args.faixa_size, len(frames)),
    }

    out = os.path.join(args.source_path, "preparo")
    os.makedirs(out, exist_ok=True)
    write_contact_sheet(os.path.join(out, "faixa01.jpg"), frames[:args.faixa_size])
    with open(os.path.join(out, "captura.json"), "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if not report["verificacao_de_faixas"]["estrutura_confirmada"]:
        raise SystemExit("estrutura de faixas NAO confirmada -- revise a premissa da captura")


def run_images(args):
    src = os.path.join(args.source_path, "images")
    frames = list_frames(src)
    if not frames:
        raise SystemExit(f"{src} vazio -- rode o COLMAP (image_undistorter) antes")

    out = os.path.join(args.source_path, f"images_{args.downscale}")
    os.makedirs(out, exist_ok=True)
    for p in frames:
        im = cv2.imread(p, cv2.IMREAD_COLOR)
        h, w = im.shape[:2]
        small = cv2.resize(im, (w // args.downscale, h // args.downscale),
                           interpolation=cv2.INTER_AREA)
        cv2.imwrite(os.path.join(out, os.path.basename(p)), small,
                    [cv2.IMWRITE_JPEG_QUALITY, 97])
    im = cv2.imread(frames[0])
    print(f"[prepare_turntable] {len(frames)} imagens em {out} "
          f"({im.shape[1] // args.downscale}x{im.shape[0] // args.downscale})")


if __name__ == "__main__":
    parser = ArgumentParser("Preparo da cena de testemunho em roletes")
    parser.add_argument("acao", choices=["check", "images"])
    parser.add_argument("--source_path", "-s", required=True)
    parser.add_argument("--input_dir", default="input", help="pasta das fotos para 'check'")
    parser.add_argument("--faixa_size", type=int, default=9,
                        help="fotos por faixa (a camera fica parada dentro de uma faixa)")
    parser.add_argument("--downscale", type=int, default=8, help="fator para 'images'")
    args = parser.parse_args()

    (run_check if args.acao == "check" else run_images)(args)
