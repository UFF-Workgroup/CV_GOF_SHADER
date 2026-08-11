"""Blocos do BRDF, em PyTorch puro.

Funcoes deste modulo sao puras (tensores entram, tensores saem) e diferenciaveis,
para que o autograd cuide de todos os gradientes -- nenhuma derivada e escrita a mao.
Ver docs/03_FORMULACAO.md para a derivacao e docs/02_DECISOES.md (ADR-001) para o
porque de o sombreamento nao ir para CUDA.

Convencao de referencial: TODAS as funcoes daqui operam em ESPACO DE VISTA
(coordenadas de camera). Duas razoes:
  1. A normal renderizada pelo GOF ja sai em espaco de vista (forward.cu, renderCUDA),
     entao perdas de consistencia de normal nao precisam de transformacao.
  2. Para captura em mesa giratoria, o environment map correto vive no referencial da
     sala, que difere do de vista por uma rotacao global constante -- absorvida pelo
     proprio mapa aprendido. Ver docs/03_FORMULACAO.md secao "mesa giratoria".
"""
import torch


def world_to_view_rotation(viewpoint_camera):
    """Rotacao mundo->vista, 3x3.

    world_view_transform e guardada transposta (convencao glm do CUDA), entao a matriz
    mundo->vista de verdade e a transposta dela. Mesmo padrao usado em train.py para a
    perda de consistencia normal-profundidade.
    """
    return viewpoint_camera.world_view_transform.transpose(0, 1)[:3, :3]


def shortest_axis_normal(rotation_matrices, scales):
    """Normal geometrica de cada Gaussiana: o eixo mais curto do elipsoide.

    Uma Gaussiana 3D achatada aproxima um plano; a direcao de menor variancia e a
    normal desse plano. E a definicao do GaussianShader (Eq. 4), e difere da normal do
    GOF, que e por raio: o GOF calcula o gradiente do level-set no ponto de intersecao
    raio-Gaussiana. As duas coincidem quando a Gaussiana e bem achatada -- que e o que
    a distortion loss do GOF induz. Medir essa divergencia e uma contribuicao do
    trabalho (ver docs/07_LIMITACOES.md).

    Args:
        rotation_matrices: [N,3,3] de build_rotation(quaternions).
        scales: [N,3] escalas por eixo, POS-ativacao.
    Returns:
        [N,3] normais unitarias em coordenadas de mundo.
    """
    # argmin sobre as escalas cruas. O filtro 3D do GOF soma em quadratura
    # (sqrt(s^2+f^2)), que e monotona em s e portanto preserva o argmin -- mas altera a
    # razao de anisotropia, entao usamos as escalas sem filtro por clareza.
    min_axis = torch.argmin(scales, dim=1)  # [N]
    # Coluna k de R e a direcao do k-esimo eixo principal no mundo.
    return torch.gather(
        rotation_matrices, 2, min_axis[:, None, None].expand(-1, 3, -1)
    ).squeeze(-1)


def orient_towards_camera(normals, view_dirs):
    """Resolve a ambiguidade de sinal da normal, virando-a para a camera.

    O eixo mais curto define uma reta, nao um sentido: +n e -n sao igualmente validos.
    Para sombreamento so o hemisferio visivel importa, entao escolhemos o sinal com
    n . omega_o > 0.
    """
    sign = torch.where((normals * view_dirs).sum(-1, keepdim=True) < 0, -1.0, 1.0)
    return normals * sign


def apply_normal_residual(normals, residual):
    """n = normalize(v + delta_n), o residuo aprendivel do GaussianShader (Eq. 4).

    Deixa a normal de sombreamento se descolar um pouco da geometria, o que ajuda em
    superficies cuja microestrutura nao e capturada pelo elipsoide. L_reg = ||delta_n||^2
    impede que ela divirja (ver utils/loss_utils.py).
    """
    return torch.nn.functional.normalize(normals + residual, dim=-1, eps=1e-8)


def reflect(view_dirs, normals):
    """Direcao de reflexao especular r = 2 (n . wo) n - wo.

    wo aponta da superficie PARA a camera; r e a direcao de onde vem a luz que reflete
    especularmente para a camera. E com ela que amostramos o environment map.
    """
    return 2.0 * (view_dirs * normals).sum(-1, keepdim=True) * normals - view_dirs


def schlick_fresnel(cos_theta, f0):
    """Aproximacao de Schlick: F = F0 + (1-F0)(1-cos)^5.

    F0 e a refletancia na incidencia normal -- o papel do specular tint s. O termo
    (1-cos)^5 produz o brilho rasante, muito visivel em testemunho de rocha e em
    qualquer dieletrico. Ablacao: --no_fresnel reduz para F = s constante.
    """
    return f0 + (1.0 - f0) * torch.pow(torch.clamp(1.0 - cos_theta, min=0.0), 5.0)


def tonemap_srgb(linear):
    """Tone mapping linear -> sRGB (gamma do artigo).

    Necessario quando o envmap representa radiancia HDR. Desligado no Marco 1 para
    preservar a equivalencia exata com o baseline do GOF (ver ADR-004): com gamma =
    identidade, s=0 e c_r=0, a equacao colapsa em clamp(SH2RGB(f_dc),0,1), que e
    exatamente o que o caminho convert_SHs_python ja calcula.
    """
    linear = torch.clamp(linear, min=0.0)
    return torch.where(
        linear <= 0.0031308,
        linear * 12.92,
        1.055 * torch.pow(linear.clamp(min=1e-8), 1.0 / 2.4) - 0.055,
    )
