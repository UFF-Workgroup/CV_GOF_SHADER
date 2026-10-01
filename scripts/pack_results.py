"""Empacota as predicoes e metricas de um run em release/<RUN_ID>/ para comparacao externa."""
import json
import os
import shutil
from argparse import ArgumentParser


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("-m", "--model_path", required=True)
    parser.add_argument("--iteration", type=int, default=30000)
    parser.add_argument("--scale", type=int, default=2, help="sufixo _<scale> das pastas de render")
    parser.add_argument("--out", default="release")
    args = parser.parse_args()

    run_id = os.path.basename(os.path.normpath(args.model_path))
    split = os.path.join(args.model_path, "test", f"ours_{args.iteration}")
    dest = os.path.join(args.out, run_id)
    os.makedirs(dest, exist_ok=True)

    shutil.copytree(os.path.join(split, f"test_preds_{args.scale}"), os.path.join(dest, "preds"), dirs_exist_ok=True)
    shutil.copytree(os.path.join(split, f"gt_{args.scale}"), os.path.join(dest, "gt"), dirs_exist_ok=True)

    for name in ("results.json", "per_view.json", "cfg_args", "cameras.json"):
        src = os.path.join(args.model_path, name)
        if os.path.exists(src):
            shutil.copy(src, dest)
    meta = os.path.join("docs", "experiments", run_id)
    for name in ("command.txt", "commit.txt", "env.txt"):
        src = os.path.join(meta, name)
        if os.path.exists(src):
            shutil.copy(src, dest)

    with open(os.path.join(args.model_path, "cameras.json")) as f:
        names = sorted(c["img_name"] for c in json.load(f))
    with open(os.path.join(dest, "test_views.csv"), "w") as f:
        f.write("indice,image_name\n")
        for i, name in enumerate(names[::8]):
            f.write(f"{i:05d},{name}\n")

    print(f"pacote em {dest}")


if __name__ == "__main__":
    main()
