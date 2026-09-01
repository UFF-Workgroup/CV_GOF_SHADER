"""Validacao geometrica das poses do COLMAP na cena de testemunho.

**Por que.** A escolha de `--light_frame view` repousa numa afirmacao sobre a captura: a
rotacao sala->camera, $C$, e constante nas 72 fotos (`03_FORMULACAO.md` §6). Ate aqui isso
era relato do operador. Este script transforma o relato em medida, usando as poses que o
COLMAP estimou -- dados independentes de quem descreveu a bancada.

**O movimento, medido nos pixels antes de mais nada** (`prepare_turntable.py check`): a
captura tem duas escalas. Dentro de uma faixa a camera **desliza** ao longo do eixo do
testemunho, com a amostra parada. Entre faixas o rolete **gira** a amostra, expondo uma
nova banda angular. Sao 8 faixas de 9 poses.

No referencial do objeto -- o que o COLMAP reconstroi -- isso prediz uma geometria bem
especifica, e cada verificacao abaixo testa uma parte dela:

1. **Registro.** Fracao das fotos que entrou na reconstrucao. A captura tem elos fracos
   entre faixas (o giro deixa pouca superficie comum); se o mapper os perder, faltam bandas
   angulares inteiras e nao se treina em cima disso.

2. **Deslize dentro da faixa.** Os centros de uma faixa devem ser **colineares**, e a
   orientacao deve ser **constante** ao longo dela -- e isso que "so desliza, mesma
   orientacao" significa em numeros.

3. **Eixo comum.** As 8 retas de deslize devem ser **paralelas entre si**: o eixo do
   testemunho e invariante sob a propria rotacao, entao ele nao muda de faixa para faixa.

4. **Giro entre faixas em torno desse mesmo eixo.** A rotacao relativa entre faixas
   consecutivas deve ter eixo alinhado com o eixo do deslize. **Este e o teste de $C$
   constante:** se a camera tivesse sido reapontada, a rotacao relativa entre faixas
   carregaria uma componente fora do eixo, e um unico envmap em espaco de vista deixaria de
   servir para todas as fotos.

    python scripts/inspect_poses.py -s ~/Documentos/rocha_fs16_16cm --faixa_size 9
"""
import json
import os
import sys
from argparse import ArgumentParser

import numpy as np

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from scene.colmap_loader import qvec2rotmat, read_extrinsics_binary, read_points3D_binary


def load_poses(sparse_dir):
    """Devolve (nomes, R_mundo2camera [N,3,3], centros [N,3]) ordenados por nome."""
    extr = read_extrinsics_binary(os.path.join(sparse_dir, "images.bin"))
    items = sorted(extr.values(), key=lambda im: im.name)
    names = [im.name for im in items]
    R = np.stack([qvec2rotmat(im.qvec) for im in items])
    t = np.stack([np.array(im.tvec) for im in items])
    centers = np.einsum("nji,nj->ni", R, -t)      # c = -R^T t
    return names, R, centers


def geodesic_deg(Ra, Rb):
    """Angulo da rotacao relativa entre duas orientacoes, em graus."""
    cos = (np.trace(Ra @ Rb.T) - 1.0) / 2.0
    return float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))))


def rotation_axis_angle(Q):
    """Eixo (unitario) e angulo (graus) de uma matriz de rotacao."""
    angle = np.arccos(np.clip((np.trace(Q) - 1.0) / 2.0, -1.0, 1.0))
    if angle < 1e-8:
        return np.array([0.0, 0.0, 1.0]), 0.0
    axis = np.array([Q[2, 1] - Q[1, 2], Q[0, 2] - Q[2, 0], Q[1, 0] - Q[0, 1]])
    n = np.linalg.norm(axis)
    if n < 1e-12:                      # rotacao de 180 graus: eixo pela parte simetrica
        w, v = np.linalg.eigh((Q + np.eye(3)) / 2.0)
        axis = v[:, -1]
    else:
        axis = axis / n
    return axis, float(np.degrees(angle))


def line_direction(points):
    """Direcao da reta que melhor ajusta os pontos, e o residuo RMS fora dela."""
    mean = points.mean(0)
    u, s, vt = np.linalg.svd(points - mean)
    d = vt[0]
    resid = points - mean - np.outer((points - mean) @ d, d)
    return d, float(np.sqrt((resid ** 2).sum(1).mean()))


def align(vectors):
    """Alinha sinais a um representante e devolve o vetor medio unitario."""
    ref = vectors[0]
    v = np.stack([w if w @ ref > 0 else -w for w in vectors])
    m = v.mean(0)
    return m / np.linalg.norm(m), v


def analyse(names, R, centers, faixa_size, n_input):
    n = len(names)
    report = {
        "fotos_de_entrada": n_input,
        "fotos_registradas": n,
        "fracao_registrada": round(n / n_input, 4),
        "faixas_completas": n // faixa_size,
    }
    faixas = [list(range(k * faixa_size, min((k + 1) * faixa_size, n)))
              for k in range(int(np.ceil(n / faixa_size)))]
    faixas = [f for f in faixas if len(f) >= 3]

    # --- 2. deslize dentro da faixa: colinearidade dos centros, orientacao constante
    dirs, retilinidade, giro_interno = [], [], []
    for idx in faixas:
        d, resid = line_direction(centers[idx])
        extensao = np.ptp(centers[idx] @ d)
        dirs.append(d)
        retilinidade.append(resid / max(extensao, 1e-12))
        giro_interno.extend(geodesic_deg(R[a], R[b]) for a, b in zip(idx, idx[1:]))
    report["deslize_na_faixa"] = {
        "residuo_fora_da_reta_relativo": {
            "mediana": round(float(np.median(retilinidade)), 5),
            "max": round(float(np.max(retilinidade)), 5),
        },
        "rotacao_entre_quadros_consecutivos_graus": {
            "mediana": round(float(np.median(giro_interno)), 4),
            "max": round(float(np.max(giro_interno)), 4),
        },
    }

    # --- 3. as retas de deslize sao paralelas? (o eixo do testemunho e invariante)
    axis, aligned = align(dirs)
    desalinho = np.degrees(np.arccos(np.clip(aligned @ axis, -1, 1)))
    report["eixo_do_testemunho"] = {
        "direcao": [round(float(v), 5) for v in axis],
        "desalinho_das_faixas_graus": {
            "mediana": round(float(np.median(desalinho)), 3),
            "max": round(float(np.max(desalinho)), 3),
        },
    }

    # --- raio: distancia dos centros ao eixo que passa pelo centroide
    origin = centers.mean(0)
    rel = centers - origin
    radial = rel - np.outer(rel @ axis, axis)
    raio = np.linalg.norm(radial, axis=1)
    report["cilindro"] = {
        "raio_medio": round(float(raio.mean()), 5),
        "raio_desvio_relativo": round(float(raio.std() / max(raio.mean(), 1e-9)), 4),
        "extensao_ao_longo_do_eixo": round(float(np.ptp(rel @ axis)), 5),
    }

    # --- 4. giro entre faixas: mesmo eixo? angulo regular?
    eixos, angulos = [], []
    for a, b in zip(faixas, faixas[1:]):
        # rotacao, em coordenadas do objeto, que leva o referencial de uma faixa ao da
        # seguinte; usa a orientacao mediana de cada faixa (constante dentro dela)
        Q = R[b[0]].T @ R[a[0]]
        e, ang = rotation_axis_angle(Q)
        eixos.append(e)
        angulos.append(ang)
    if eixos:
        eixo_giro, alinhados = align(eixos)
        desvio_do_eixo = np.degrees(np.arccos(np.clip(np.abs(alinhados @ axis), 0, 1)))
        report["giro_entre_faixas"] = {
            "angulo_graus": [round(a, 2) for a in angulos],
            "angulo_medio_graus": round(float(np.mean(angulos)), 2),
            "cobertura_angular_total_graus": round(float(np.sum(angulos)), 2),
            "desvio_entre_eixo_do_giro_e_eixo_do_deslize_graus": {
                "mediana": round(float(np.median(desvio_do_eixo)), 3),
                "max": round(float(np.max(desvio_do_eixo)), 3),
            },
        }
        # C constante exige: (a) orientacao fixa dentro da faixa e (b) o giro entre faixas
        # em torno do eixo do testemunho, sem componente de reapontamento.
        report["C_constante"] = {
            "sustentado": bool(report["deslize_na_faixa"]
                               ["rotacao_entre_quadros_consecutivos_graus"]["max"] < 2.0
                               and float(np.max(desvio_do_eixo)) < 10.0),
            "criterio": "giro dentro da faixa < 2 graus e eixo do giro a menos de 10 graus do eixo do deslize",
        }
    return report


def main():
    parser = ArgumentParser("Validacao das poses do COLMAP")
    parser.add_argument("--source_path", "-s", required=True)
    parser.add_argument("--faixa_size", type=int, default=9)
    parser.add_argument("--out", default=None,
                        help="json de saida (default: <fonte>/preparo/poses.json)")
    args = parser.parse_args()

    sparse = os.path.join(args.source_path, "sparse", "0")
    names, R, centers = load_poses(sparse)
    n_input = len([f for f in os.listdir(os.path.join(args.source_path, "input"))
                   if not f.startswith(".")])

    report = analyse(names, R, centers, args.faixa_size, n_input)
    report["pontos_sfm"] = len(read_points3D_binary(os.path.join(sparse, "points3D.bin"))[0])
    report["faltando"] = sorted(
        set(os.listdir(os.path.join(args.source_path, "input"))) - set(names))

    out = args.out or os.path.join(args.source_path, "preparo", "poses.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
