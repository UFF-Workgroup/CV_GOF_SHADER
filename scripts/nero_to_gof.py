"""Adiciona camera_angle_x aos transforms_*.json de uma cena convertida por nero2blender.py, que o leitor Blender do GOF exige."""
import json
import math
import os
from argparse import ArgumentParser


def main():
    parser = ArgumentParser()
    parser.add_argument("--src", required=True, help="pasta <cena>_blender gerada por nero2blender.py")
    parser.add_argument("--dst", required=True)
    args = parser.parse_args()
    os.makedirs(args.dst, exist_ok=True)
    for split in ("train", "test"):
        with open(os.path.join(args.src, f"transforms_{split}.json")) as f:
            t = json.load(f)
        t["camera_angle_x"] = 2 * math.atan(t["w"] / (2 * t["fl_x"]))
        with open(os.path.join(args.dst, f"transforms_{split}.json"), "w") as f:
            json.dump(t, f)
    link = os.path.join(args.dst, "rgb")
    if not os.path.exists(link):
        os.symlink(os.path.abspath(os.path.join(args.src, "rgb")), link)


if __name__ == "__main__":
    main()
