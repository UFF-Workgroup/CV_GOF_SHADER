"""Blocos do BRDF em PyTorch puro, diferenciaveis e em espaco de vista."""
import torch


def world_to_view_rotation(viewpoint_camera):
    """Rotacao mundo->vista 3x3 (world_view_transform vem transposta, convencao glm)."""
    return viewpoint_camera.world_view_transform.transpose(0, 1)[:3, :3]


def shortest_axis_normal(rotation_matrices, scales):
    """Normal de cada Gaussiana: o eixo mais curto do elipsoide, em coordenadas de mundo."""
    min_axis = torch.argmin(scales, dim=1)
    return torch.gather(
        rotation_matrices, 2, min_axis[:, None, None].expand(-1, 3, -1)
    ).squeeze(-1)


def orient_towards_camera(normals, view_dirs):
    """Vira a normal para a camera (n . omega_o > 0)."""
    sign = torch.where((normals * view_dirs).sum(-1, keepdim=True) < 0, -1.0, 1.0)
    return normals * sign


def apply_normal_residual(normals, residual):
    """n = normalize(v + delta_n), residuo aprendivel do GaussianShader (Eq. 4)."""
    return torch.nn.functional.normalize(normals + residual, dim=-1, eps=1e-8)


def reflect(view_dirs, normals):
    """Direcao de reflexao r = 2 (n . wo) n - wo."""
    return 2.0 * (view_dirs * normals).sum(-1, keepdim=True) * normals - view_dirs


def schlick_fresnel(cos_theta, f0):
    """Aproximacao de Schlick: F = F0 + (1-F0)(1-cos)^5."""
    return f0 + (1.0 - f0) * torch.pow(torch.clamp(1.0 - cos_theta, min=0.0), 5.0)


def tonemap_srgb(linear):
    """Tone mapping linear -> sRGB."""
    linear = torch.clamp(linear, min=0.0)
    return torch.where(
        linear <= 0.0031308,
        linear * 12.92,
        1.055 * torch.pow(linear.clamp(min=1e-8), 1.0 / 2.4) - 0.055,
    )
