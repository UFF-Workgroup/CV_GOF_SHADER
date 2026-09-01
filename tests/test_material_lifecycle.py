"""Testes do ciclo de vida dos parametros de material (Fase 1 da auditoria).

Cobre T3, T4 e T5 do plano. Cada teste corresponde a um bug real encontrado na
auditoria de 2026-08-10, e todos os tres falhavam EM SILENCIO no codigo anterior:
formas corretas, nenhuma excecao, resultado errado.

Requer GPU (o GaussianModel fixa device="cuda").
"""
import os
import sys
from argparse import ArgumentParser

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from arguments import ModelParams, OptimizationParams
from scene.gaussian_model import GaussianModel
from utils.graphics_utils import BasicPointCloud

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="precisa de GPU")


def make_model(num_points=500, sh_degree=0, seed=0):
    rng = np.random.default_rng(seed)
    pcd = BasicPointCloud(
        points=rng.normal(size=(num_points, 3)).astype(np.float32),
        colors=rng.uniform(size=(num_points, 3)).astype(np.float32),
        normals=np.zeros((num_points, 3), dtype=np.float32),
    )
    model = GaussianModel(sh_degree)
    model.create_from_pcd(pcd, spatial_lr_scale=1.0)
    model.filter_3D = torch.zeros((num_points, 1), device="cuda")

    parser = ArgumentParser()
    opt = OptimizationParams(parser)
    training_args = opt.extract(parser.parse_args([]))
    model.training_setup(training_args)
    return model, training_args


def test_material_init_values():
    """B-3: material comeca quase difuso, nao com 50% de especular."""
    model, _ = make_model()
    assert torch.allclose(model.get_specular_tint, torch.full_like(model.get_specular_tint, 0.05), atol=1e-5)
    assert torch.allclose(model.get_roughness, torch.full_like(model.get_roughness, 0.70), atol=1e-5)
    assert torch.allclose(model.get_normal_residual, torch.zeros_like(model.get_normal_residual))


def test_t3_optimizer_binding_survives_prune():
    """T3/A-1: apos podar, o tensor do modelo tem de ser o MESMO objeto que o Adam atualiza.

    No codigo anterior prune_points recriava self._specular_tint a partir do tensor
    antigo, descartando o que _prune_optimizer havia podado. O Adam passava a atualizar
    um orfao e o parametro do modelo congelava a partir da iteracao 600.
    """
    model, _ = make_model()
    mask = torch.zeros(model.get_xyz.shape[0], dtype=torch.bool, device="cuda")
    mask[:100] = True  # poda 100 pontos

    model.prune_points(mask)  # levanta RuntimeError se a identidade quebrar

    attr_map = model._optimizer_attr_map()
    for group in model.optimizer.param_groups:
        attr = attr_map.get(group["name"])
        if attr is None:
            continue
        assert getattr(model, attr) is group["params"][0], f"'{group['name']}' desconectado"
        assert getattr(model, attr).shape[0] == 400


def test_t3_material_actually_learns_after_prune():
    """A-1, o efeito observavel: um passo do Adam apos a poda tem de MOVER o material.

    Este e o teste que teria pego o bug original. A checagem de identidade acima e
    estrutural; esta aqui e comportamental.
    """
    model, _ = make_model()
    mask = torch.zeros(model.get_xyz.shape[0], dtype=torch.bool, device="cuda")
    mask[:100] = True
    model.prune_points(mask)

    before = model._specular_tint.detach().clone()
    loss = model.get_specular_tint.sum() + model.get_roughness.sum()
    loss.backward()
    assert model._specular_tint.grad is not None, "material nao recebeu gradiente"
    model.optimizer.step()

    moved = (model._specular_tint.detach() - before).abs().max().item()
    assert moved > 0, "o Adam nao moveu _specular_tint: parametro orfao (bug A-1)"


def test_t3_binding_survives_densification():
    model, _ = make_model()
    n0 = model.get_xyz.shape[0]
    model.xyz_gradient_accum = torch.ones((n0, 1), device="cuda")
    model.xyz_gradient_accum_abs = torch.ones((n0, 1), device="cuda")
    model.denom = torch.ones((n0, 1), device="cuda")
    model.max_radii2D = torch.zeros(n0, device="cuda")

    model.densify_and_prune(max_grad=0.0002, min_opacity=0.005, extent=1.0, max_screen_size=None)

    attr_map = model._optimizer_attr_map()
    n = model.get_xyz.shape[0]
    for group in model.optimizer.param_groups:
        attr = attr_map.get(group["name"])
        if attr is None:
            continue
        assert getattr(model, attr) is group["params"][0], f"'{group['name']}' desconectado"
        assert getattr(model, attr).shape[0] == n, f"'{group['name']}' com contagem divergente"


def test_t4_ply_roundtrip_preserves_material(tmp_path):
    """T4/A-3: o material tem de sobreviver a save_ply -> load_ply.

    Sem isto, render.py/extract_mesh.py avaliariam com material default -- diferente do
    de treino -- e nada no log denunciaria.
    """
    model, _ = make_model()
    with torch.no_grad():
        model._specular_tint += torch.randn_like(model._specular_tint)
        model._roughness += torch.randn_like(model._roughness)
        model._normal_residual += torch.randn_like(model._normal_residual)
    expected = {a: getattr(model, a).detach().clone() for a in model.MATERIAL_PARAMS}

    path = str(tmp_path / "point_cloud.ply")
    model.save_ply(path)

    reloaded = GaussianModel(0)
    reloaded.load_ply(path)
    for attr, want in expected.items():
        got = getattr(reloaded, attr)
        assert got.shape == want.shape, f"{attr}: forma {got.shape} != {want.shape}"
        # PLY guarda float32; a tolerancia cobre so o arredondamento de ida e volta.
        assert torch.allclose(got, want, atol=1e-6), f"{attr} nao sobreviveu ao round-trip"


def test_t4_load_ply_without_material_falls_back(tmp_path):
    """Compatibilidade: PLYs antigos (sem material) ainda carregam, com os defaults."""
    model, _ = make_model()
    path = str(tmp_path / "legacy.ply")

    # Escreve um PLY no formato antigo, sem os atributos de material.
    from plyfile import PlyData, PlyElement
    xyz = model._xyz.detach().cpu().numpy()
    normals = np.zeros_like(xyz)
    f_dc = model._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
    f_rest = model._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
    opacities = model._opacity.detach().cpu().numpy()
    scale = model._scaling.detach().cpu().numpy()
    rotation = model._rotation.detach().cpu().numpy()
    filter_3D = model.filter_3D.detach().cpu().numpy()
    attrs = model.construct_list_of_attributes(include_material=False)
    elements = np.empty(xyz.shape[0], dtype=[(a, "f4") for a in attrs])
    elements[:] = list(map(tuple, np.concatenate(
        (xyz, normals, f_dc, f_rest, opacities, scale, rotation, filter_3D), axis=1)))
    PlyData([PlyElement.describe(elements, "vertex")]).write(path)

    reloaded = GaussianModel(0)
    reloaded.load_ply(path)  # nao pode estourar
    assert torch.allclose(reloaded.get_roughness, torch.full_like(reloaded.get_roughness, 0.70), atol=1e-5)


def test_t5_checkpoint_roundtrip(tmp_path):
    """T5/A-2: capture -> restore tem de trazer o material de volta."""
    model, training_args = make_model()
    with torch.no_grad():
        model._specular_tint += torch.randn_like(model._specular_tint)
    expected = {a: getattr(model, a).detach().clone() for a in model.MATERIAL_PARAMS}

    path = str(tmp_path / "chkpnt.pth")
    torch.save((model.capture(), 1000), path)

    (params, it) = torch.load(path)
    restored = GaussianModel(0)
    restored.restore(params, training_args)

    assert it == 1000
    for attr, want in expected.items():
        assert torch.allclose(getattr(restored, attr), want), f"{attr} perdido no checkpoint"
    restored._assert_optimizer_binding()


def make_model_with_lighting(num_points=500, sh_degree=0, seed=0, light_repr="sh"):
    """Como make_model, mas com --brdf ligado e a iluminacao configurada.

    setup_lighting precisa vir ANTES de training_setup, para que a iluminacao entre
    no grupo 'lighting' do Adam -- mesma ordem que scene/__init__.py + train.py usam.
    """
    model, training_args = make_model(num_points=num_points, sh_degree=sh_degree, seed=seed)
    parser = ArgumentParser()
    lp = ModelParams(parser)
    dataset_args = lp.extract(parser.parse_args(["--light_repr", light_repr]))
    dataset_args.brdf = True
    model.setup_lighting(dataset_args)
    model.training_setup(training_args)
    return model, training_args


def test_t5_checkpoint_roundtrip_preserves_lighting(tmp_path):
    """A-4: capture -> restore com --brdf tem que trazer o envmap/SH aprendido de volta.

    Achado ao auditar o checkpoint para retomada apos interrupcao (mesma classe do A-2:
    parametro desconectado do otimizador, agora na iluminacao). Antes desta correcao,
    capture()/restore() nao tocavam em self.lighting -- retomar de checkpoint reiniciava
    a iluminacao para o ruido inicial, silenciosamente, enquanto reaplicava o MOMENTO do
    Adam calculado para os valores antigos. Isso descartaria toda a iluminacao aprendida
    exatamente nos runs que mais precisam de checkpoint (E1/E2, --brdf, 30k iteracoes,
    maquina sem no-break).
    """
    model, training_args = make_model_with_lighting(light_repr="sh")
    with torch.no_grad():
        model.lighting.coeffs += torch.randn_like(model.lighting.coeffs)
    expected = model.lighting.coeffs.detach().clone()

    path = str(tmp_path / "chkpnt_brdf.pth")
    torch.save((model.capture(), 2000), path)

    (params, it) = torch.load(path)
    restored, _ = make_model_with_lighting(light_repr="sh")  # self.lighting != None ANTES do restore
    restored.restore(params, training_args)

    assert it == 2000
    assert torch.allclose(restored.lighting.coeffs, expected), "iluminacao perdida no checkpoint (A-4)"
    restored._assert_optimizer_binding()


def test_restore_lighting_mismatch_fails_loud():
    """Restaurar um checkpoint com iluminacao num modelo sem --brdf tem que falhar alto,
    nunca descartar a iluminacao aprendida em silencio (mesmo espirito de load_lighting)."""
    model, training_args = make_model_with_lighting(light_repr="sh")
    params = model.capture()

    restored, _ = make_model()  # sem setup_lighting -> self.lighting is None
    with pytest.raises(RuntimeError):
        restored.restore(params, training_args)


def test_restore_lighting_class_mismatch_fails_loud():
    """Checkpoint com envmap restaurado num modelo configurado para SH (ou vice-versa)
    tambem tem que falhar alto -- confundir as duas classes invalidaria os numeros."""
    model, training_args = make_model_with_lighting(light_repr="envmap")
    params = model.capture()

    restored, _ = make_model_with_lighting(light_repr="sh")
    with pytest.raises(RuntimeError):
        restored.restore(params, training_args)
