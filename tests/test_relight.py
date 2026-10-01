"""Testes dos envmaps sinteticos de relighting (scripts/relight.py)."""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))

from relight import build_envmaps, equirect_directions, match_mean
from scene.lighting import direction_to_equirect_uv


def test_directions_are_unit_length():
    dirs = equirect_directions(32)
    assert torch.allclose(dirs.norm(dim=0), torch.ones(32, 64), atol=1e-5)


def test_equirect_grid_is_exact_inverse_of_uv_mapping():
    """A propriedade que faz mapa sintetico e mapa aprendido serem comparaveis."""
    height, width = 32, 64
    dirs = equirect_directions(height).permute(1, 2, 0).reshape(-1, 3)
    uv = direction_to_equirect_uv(dirs)

    v_expected = ((torch.arange(height) + 0.5) / height * 2 - 1)[:, None].expand(height, width)
    u_expected = ((torch.arange(width) + 0.5) / width * 2 - 1)[None, :].expand(height, width)
    expected = torch.stack([u_expected, v_expected], dim=-1).reshape(-1, 2)

    err = (uv - expected).abs()
    # A costura em u e periodica: um erro de 2.0 e o mesmo ponto, nao um erro.
    err_u = torch.minimum(err[:, 0], (2.0 - err[:, 0]).abs())
    assert float(err[:, 1].max()) < 1e-5
    assert float(err_u.max()) < 1e-5


def test_poles_map_to_plus_and_minus_y():
    """v=0 e o polo +y. Ancora a orientacao vertical, que nenhum outro teste fixa."""
    dirs = equirect_directions(16)
    assert dirs[1, 0, :].mean() > 0.99    # primeira linha -> +y
    assert dirs[1, -1, :].mean() < -0.99  # ultima linha  -> -y


def test_match_mean_preserves_structure_and_sets_brightness():
    maps = build_envmaps(32)
    src = maps["estudio_lateral"]
    out = match_mean(src, 0.25)
    assert abs(float(out.mean()) - 0.25) < 1e-5
    # Multiplicacao por escalar: o contraste relativo nao pode mudar.
    assert abs(float(out.max() / out.mean()) - float(src.max() / src.mean())) < 1e-3


def test_envmaps_are_nonnegative_and_finite():
    for name, m in build_envmaps(32).items():
        assert torch.isfinite(m).all(), name
        assert float(m.min()) >= 0.0, name


def test_uniform_map_has_no_structure_and_others_do():
    """O 'uniforme' e o controle negativo: precisa mesmo ser degenerado."""
    maps = build_envmaps(32)
    assert float(maps["uniforme"].std()) == 0.0
    for name in ("estudio_lateral", "contraluz_dupla", "ceu_chao"):
        assert float(maps[name].std()) > 1e-3, name


def test_lateral_blob_lands_on_the_named_side():
    """O mapa 'estudio_lateral' tem a fonte em -x: o pico tem de cair nesse hemisferio."""
    maps = build_envmaps(64)
    m = maps["estudio_lateral"].mean(0)
    idx = int(torch.argmax(m))
    row, col = idx // m.shape[1], idx % m.shape[1]
    peak_dir = equirect_directions(64)[:, row, col]
    assert float(peak_dir[0]) < -0.5   # hemisferio -x
    assert float(peak_dir[1]) > 0.0    # acima do horizonte
