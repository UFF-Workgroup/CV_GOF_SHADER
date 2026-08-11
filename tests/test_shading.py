"""Testes do sombreamento BRDF (Fase 3). Cobre T1, T2, T6 e T7 do plano.

T1 e o teste mais importante do projeto. Ele prova que a nova via de cor
(colors_precomp calculado em PyTorch) e equivalente a antiga (SH avaliadas dentro do
CUDA). Sem essa garantia, qualquer diferenca de PSNR entre baseline e modelo com BRDF
poderia vir de uma mudanca acidental na cor difusa em vez do termo especular -- e toda
a matriz de ablacoes perderia o sentido.
"""
import os
import sys
from argparse import ArgumentParser, Namespace

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from arguments import ModelParams, OptimizationParams, PipelineParams
from gaussian_renderer import render
from scene import brdf
from scene.cameras import Camera
from scene.gaussian_model import GaussianModel
from scene.lighting import EnvironmentMap, SHLighting
from utils.graphics_utils import BasicPointCloud

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="precisa de GPU")


def make_camera(width=64, height=64, dist=4.0):
    R = np.eye(3)
    T = np.array([0.0, 0.0, dist])
    image = torch.zeros(3, height, width)
    return Camera(0, R, T, FoVx=0.9, FoVy=0.9, image=image, gt_alpha_mask=None,
                  image_name="synth", uid=0)


def make_scene(num_points=800, sh_degree=0, seed=1):
    rng = np.random.default_rng(seed)
    pcd = BasicPointCloud(
        points=(rng.normal(size=(num_points, 3)) * 0.3).astype(np.float32),
        colors=rng.uniform(0.2, 0.8, size=(num_points, 3)).astype(np.float32),
        normals=np.zeros((num_points, 3), dtype=np.float32),
    )
    model = GaussianModel(sh_degree)
    model.create_from_pcd(pcd, spatial_lr_scale=1.0)
    model.filter_3D = torch.full((num_points, 1), 1e-4, device="cuda")
    model.active_sh_degree = sh_degree
    return model


def default_args(**overrides):
    parser = ArgumentParser()
    mp = ModelParams(parser)
    args = mp.extract(parser.parse_args([]))
    for k, v in overrides.items():
        setattr(args, k, v)
    return args


def default_pipe(**overrides):
    parser = ArgumentParser()
    pp = PipelineParams(parser)
    pipe = pp.extract(parser.parse_args([]))
    for k, v in overrides.items():
        setattr(pipe, k, v)
    return pipe


# ---------------------------------------------------------------- T1


def test_t1_colors_precomp_matches_cuda_sh_path():
    """T1: renderizar por colors_precomp (Python) == renderizar por SH (CUDA).

    Compara a imagem inteira, nao so as cores por-Gaussiana: exercita o caminho de
    verdade que o treino usa.
    """
    model = make_scene(sh_degree=0)
    cam = make_camera()
    bg = torch.zeros(3, device="cuda")

    with torch.no_grad():
        # Caminho A: SH avaliadas dentro do CUDA (baseline do GOF).
        img_cuda = render(cam, model, default_pipe(), bg, kernel_size=0.0)["render"][:3]
        # Caminho B: cor por-Gaussiana calculada em PyTorch.
        img_python = render(cam, model, default_pipe(convert_SHs_python=True), bg,
                            kernel_size=0.0)["render"][:3]

    diff = (img_cuda - img_python).abs().max().item()
    assert diff < 1e-5, f"caminho colors_precomp divergiu do CUDA em {diff:.2e}"


def test_t1_brdf_reduces_to_baseline_when_specular_is_zero():
    """T1 (continuacao): com s=0 e sem Fresnel, o BRDF colapsa no baseline exato.

    Esta e a propriedade que torna a ablacao interpretavel: a diferenca medida entre
    baseline e BRDF e atribuivel ao termo especular, e nao a uma mudanca de parametrizacao
    da cor difusa.

    Nota: com o Fresnel de Schlick LIGADO essa identidade nao vale, e corretamente:
    F = F0 + (1-F0)(1-cos)^5 tende a 1 em angulo rasante mesmo com F0 = 0. Ver o teste
    seguinte, que verifica exatamente isso.
    """
    model = make_scene(sh_degree=0)
    cam = make_camera()
    bg = torch.zeros(3, device="cuda")

    args = default_args(brdf=True, no_fresnel=True, use_tonemap=False)
    model.setup_lighting(args)
    with torch.no_grad():
        model._specular_tint.fill_(-30.0)  # sigmoid(-30) ~ 0
        img_baseline = render(cam, model, default_pipe(), bg, kernel_size=0.0)["render"][:3]
        img_brdf = render(cam, model, default_pipe(), bg, kernel_size=0.0, brdf_args=args)["render"][:3]

    diff = (img_baseline - img_brdf).abs().max().item()
    assert diff < 1e-5, f"BRDF com s=0 nao reduziu ao baseline: diferenca {diff:.2e}"


def test_fresnel_breaks_exact_reduction_by_design():
    """Documenta que o Fresnel quebra a reducao exata -- de proposito, e fisicamente.

    Se este teste um dia passar a nao ver diferenca, o Fresnel parou de ser aplicado.
    """
    model = make_scene(sh_degree=0)
    cam = make_camera()
    bg = torch.zeros(3, device="cuda")

    args = default_args(brdf=True, no_fresnel=False)
    model.setup_lighting(args)
    with torch.no_grad():
        model._specular_tint.fill_(-30.0)
        img_baseline = render(cam, model, default_pipe(), bg, kernel_size=0.0)["render"][:3]
        img_fresnel = render(cam, model, default_pipe(), bg, kernel_size=0.0, brdf_args=args)["render"][:3]

    diff = (img_baseline - img_fresnel).abs().max().item()
    assert diff > 1e-4, "Fresnel de Schlick nao teve efeito algum com s=0"


# ---------------------------------------------------------------- T2


@pytest.mark.parametrize("light_repr", ["envmap", "sh"])
def test_t2_gradient_reaches_every_brdf_parameter(light_repr):
    """T2: um backward tem de levar gradiente nao-nulo a todo parametro novo.

    E a verificacao que substitui as derivadas manuais que o plano antigo escreveria em
    backward.cu: se o autograd conecta tudo, nao ha derivada para errar.
    """
    model = make_scene(sh_degree=0)
    cam = make_camera()
    bg = torch.zeros(3, device="cuda")

    args = default_args(brdf=True, light_repr=light_repr, use_normal_residual=True)
    model.setup_lighting(args)
    # Iluminacao com estrutura. Sob luz UNIFORME a rugosidade e nao-identificavel e o
    # gradiente dela e legitimamente zero -- ver o teste seguinte. Exigir gradiente
    # nao-nulo naquele caso seria exigir que o codigo violasse a matematica.
    with torch.no_grad():
        for p in model.lighting.parameters():
            p.add_(torch.randn_like(p) * 0.3)

    img = render(cam, model, default_pipe(), bg, kernel_size=0.0, brdf_args=args)["render"][:3]
    img.sum().backward()

    checks = {
        "_specular_tint": model._specular_tint,
        "_roughness": model._roughness,
        "_normal_residual": model._normal_residual,
    }
    for name, p in checks.items():
        assert p.grad is not None, f"{name} nao recebeu gradiente"
        assert torch.isfinite(p.grad).all(), f"{name} recebeu gradiente nao-finito"
        assert p.grad.abs().sum().item() > 0, f"{name} recebeu gradiente identicamente nulo"

    light_params = list(model.lighting.parameters())
    assert light_params, "iluminacao sem parametros"
    for p in light_params:
        assert p.grad is not None and p.grad.abs().sum().item() > 0, "iluminacao sem gradiente"


@pytest.mark.parametrize("light_repr", ["envmap", "sh"])
def test_roughness_is_unidentifiable_under_uniform_light(light_repr):
    """Limitacao documentada: com luz uniforme, dL/drho = 0.

    Nao e bug, e identificabilidade. Se todas as direcoes tem a mesma radiancia, borrar
    o lobulo especular nao muda nada, entao a rugosidade nao deixa assinatura na imagem e
    nao pode ser estimada. Consequencias praticas registradas em docs/07_LIMITACOES.md:

      1. As classes de iluminacao inicializam com ruido pequeno (init_std) para nao
         partir de um ponto exatamente degenerado.
      2. Numa cena de iluminacao quase uniforme (caixa de luz difusa, comum em bancada de
         rocha), rho fica mal condicionado e seu valor final deve ser lido com ceticismo.

    O teste fixa init_std=0 de proposito para exibir a degenerescencia pura.
    """
    dirs = torch.nn.functional.normalize(torch.randn(512, 3, device="cuda"), dim=-1)
    light = (EnvironmentMap(64, 6, init_std=0.0) if light_repr == "envmap"
             else SHLighting(3, init_std=0.0)).cuda()

    rough = torch.full((512, 1), 0.5, device="cuda", requires_grad=True)
    out = light.sample(dirs, rough)
    out.sum().backward()

    assert out.var().item() < 1e-12, "a luz deveria ser uniforme neste teste"
    # envmap deixa ruido de ponto flutuante (avg_pool + grid_sample em resolucoes
    # diferentes); SH da zero exato. Ambos sao "sem sinal util".
    assert rough.grad.abs().max().item() < 1e-6, (
        "gradiente de rugosidade nao deveria ter sinal util sob luz uniforme")


def test_lighting_init_breaks_the_degeneracy():
    """Com o init padrao (init_std>0), a rugosidade ja recebe gradiente na iteracao 0."""
    dirs = torch.nn.functional.normalize(torch.randn(512, 3, device="cuda"), dim=-1)
    for light in (EnvironmentMap(64, 6).cuda(), SHLighting(3).cuda()):
        rough = torch.full((512, 1), 0.5, device="cuda", requires_grad=True)
        light.sample(dirs, rough).sum().backward()
        assert rough.grad.abs().sum().item() > 1e-4, f"{type(light).__name__} partiu degenerado"


def test_t2_light_frame_view_actually_changes_result():
    """A escolha de referencial da luz tem de mudar o resultado.

    Se 'view' e 'world' produzissem a mesma imagem, a adaptacao para mesa giratoria
    (a contribuicao metodologica) seria inocua -- e um bug silencioso.
    """
    cam = make_camera()
    bg = torch.zeros(3, device="cuda")
    images = {}
    for frame in ("view", "world"):
        model = make_scene(sh_degree=0)
        args = default_args(brdf=True, light_frame=frame)
        model.setup_lighting(args)
        with torch.no_grad():
            model._specular_tint.fill_(2.0)              # bem especular
            model.lighting.base.normal_(0.5, 0.3)        # envmap nao-uniforme
            images[frame] = render(cam, model, default_pipe(), bg, kernel_size=0.0,
                                   brdf_args=args)["render"][:3].clone()

    # Camera na origem olhando -z com R=I: aqui view e world diferem por uma rotacao nao
    # trivial (T=[0,0,4]), entao as imagens tem de diferir.
    diff = (images["view"] - images["world"]).abs().max().item()
    assert diff > 1e-5, "light_frame nao teve efeito: a rotacao para espaco de vista nao foi aplicada"


# ---------------------------------------------------------------- unidades do BRDF


def test_shortest_axis_normal_picks_flat_direction():
    """A normal tem de ser o eixo de MENOR escala (a direcao achatada)."""
    n = 5
    R = torch.eye(3, device="cuda")[None].repeat(n, 1, 1)
    scales = torch.tensor([[1.0, 1.0, 0.01]], device="cuda").repeat(n, 1)
    normals = brdf.shortest_axis_normal(R, scales)
    expected = torch.tensor([0.0, 0.0, 1.0], device="cuda").expand(n, 3)
    assert torch.allclose(normals.abs(), expected.abs(), atol=1e-6)

    scales = torch.tensor([[0.01, 1.0, 1.0]], device="cuda").repeat(n, 1)
    normals = brdf.shortest_axis_normal(R, scales)
    expected = torch.tensor([1.0, 0.0, 0.0], device="cuda").expand(n, 3)
    assert torch.allclose(normals.abs(), expected.abs(), atol=1e-6)


def test_reflection_is_an_involution_on_the_normal():
    """Refletir wo em torno de n e depois refletir r deve devolver wo."""
    torch.manual_seed(0)
    n = torch.nn.functional.normalize(torch.randn(64, 3, device="cuda"), dim=-1)
    wo = torch.nn.functional.normalize(torch.randn(64, 3, device="cuda"), dim=-1)
    wo = brdf.orient_towards_camera(wo, n) * 0 + wo  # wo arbitrario
    r = brdf.reflect(wo, n)
    back = brdf.reflect(r, n)
    assert torch.allclose(back, wo, atol=1e-5)
    # reflexao preserva norma
    assert torch.allclose(r.norm(dim=-1), torch.ones(64, device="cuda"), atol=1e-5)


def test_orient_towards_camera_makes_dot_nonnegative():
    torch.manual_seed(0)
    n = torch.nn.functional.normalize(torch.randn(256, 3, device="cuda"), dim=-1)
    wo = torch.nn.functional.normalize(torch.randn(256, 3, device="cuda"), dim=-1)
    oriented = brdf.orient_towards_camera(n, wo)
    assert ((oriented * wo).sum(-1) >= -1e-6).all()


def test_schlick_fresnel_endpoints():
    f0 = torch.full((32, 3), 0.04, device="cuda")
    at_normal = brdf.schlick_fresnel(torch.ones(32, 1, device="cuda"), f0)
    assert torch.allclose(at_normal, f0, atol=1e-6), "incidencia normal deve dar F0"
    at_grazing = brdf.schlick_fresnel(torch.zeros(32, 1, device="cuda"), f0)
    assert torch.allclose(at_grazing, torch.ones_like(f0), atol=1e-6), "rasante deve dar 1"


# ---------------------------------------------------------------- iluminacao


def test_envmap_roughness_blurs_the_reflection():
    """Rugosidade alta tem de ler um mip mais borrado: menos variancia entre direcoes."""
    env = EnvironmentMap(resolution=64, num_levels=6).cuda()
    with torch.no_grad():
        torch.manual_seed(0)
        env.base.normal_(0.5, 0.25)

    dirs = torch.nn.functional.normalize(torch.randn(4096, 3, device="cuda"), dim=-1)
    sharp = env.sample(dirs, torch.zeros(4096, 1, device="cuda"))
    blurry = env.sample(dirs, torch.ones(4096, 1, device="cuda"))
    assert blurry.var().item() < sharp.var().item(), "rugosidade nao borrou a iluminacao"


def test_sh_lighting_attenuates_high_frequency_with_roughness():
    light = SHLighting(degree=3).cuda()
    with torch.no_grad():
        torch.manual_seed(0)
        light.coeffs.normal_(0.0, 0.5)
        light.coeffs[:, 0] = 2.0

    dirs = torch.nn.functional.normalize(torch.randn(4096, 3, device="cuda"), dim=-1)
    sharp = light.sample(dirs, torch.zeros(4096, 1, device="cuda"))
    blurry = light.sample(dirs, torch.ones(4096, 1, device="cuda"))
    assert blurry.var().item() < sharp.var().item(), "convolucao SH nao atenuou alta frequencia"


def test_t6_lighting_roundtrip(tmp_path):
    """T6: a iluminacao aprendida tem de sobreviver a save/load.

    Sem isto, avaliar um modelo treinado usaria um envmap cinza -- mesma classe de erro
    que o bug A-3 do PLY.
    """
    model = make_scene()
    args = default_args(brdf=True)
    model.setup_lighting(args)
    with torch.no_grad():
        model.lighting.base.normal_(0.3, 0.2)
    expected = model.lighting.base.detach().clone()

    path = str(tmp_path / "lighting.pth")
    model.save_lighting(path)

    other = make_scene()
    other.setup_lighting(args)
    other.load_lighting(path)
    assert torch.allclose(other.lighting.base, expected)


def test_lighting_repr_mismatch_is_loud(tmp_path):
    """Carregar SH sobre envmap tem de falhar alto, nao silenciosamente."""
    model = make_scene()
    model.setup_lighting(default_args(brdf=True, light_repr="envmap"))
    path = str(tmp_path / "lighting.pth")
    model.save_lighting(path)

    other = make_scene()
    other.setup_lighting(default_args(brdf=True, light_repr="sh"))
    with pytest.raises(RuntimeError, match="light_repr"):
        other.load_lighting(path)


# ---------------------------------------------------------------- T7


def test_t7_recovers_planted_specular_signal():
    """T7: o otimizador consegue recuperar um material especular plantado?

    Monta um alvo sintetico com s e rho conhecidos, parte de uma inicializacao errada e
    checa que a otimizacao anda na direcao certa. Nao e um teste de convergencia exata --
    e um teste de que os gradientes tem o sinal e a magnitude corretos ponta a ponta.
    """
    torch.manual_seed(0)
    model = make_scene(num_points=2000, sh_degree=0)
    cam = make_camera()
    bg = torch.zeros(3, device="cuda")
    args = default_args(brdf=True, no_fresnel=True)
    model.setup_lighting(args)

    with torch.no_grad():
        model.lighting.base.normal_(0.6, 0.2)
        model._specular_tint.fill_(2.0)     # alvo: s = sigmoid(2) ~ 0.88
        target = render(cam, model, default_pipe(), bg, kernel_size=0.0,
                        brdf_args=args)["render"][:3].clone()
        model._specular_tint.fill_(-2.0)    # inicio errado: s ~ 0.12

    opt = torch.optim.Adam([model._specular_tint], lr=0.1)
    first = last = None
    for step in range(60):
        opt.zero_grad()
        img = render(cam, model, default_pipe(), bg, kernel_size=0.0, brdf_args=args)["render"][:3]
        loss = (img - target).abs().mean()
        loss.backward()
        opt.step()
        if step == 0:
            first = loss.item()
        last = loss.item()

    assert last < first * 0.5, f"a perda mal caiu: {first:.5f} -> {last:.5f}"
    recovered = model.get_specular_tint.mean().item()
    assert recovered > 0.3, f"s nao subiu na direcao do alvo (~0.88): ficou em {recovered:.3f}"
