"""Iluminacao incidente do termo especular: EnvironmentMap (alta frequencia) ou SHLighting (baixa)."""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.sh_utils import eval_sh


def direction_to_equirect_uv(dirs):
    """Direcao unitaria -> (u,v) equirretangular em [-1,1]; afasta do polo para evitar gradiente NaN."""
    eps = 1e-4
    x, y, z = dirs[..., 0], dirs[..., 1], dirs[..., 2]
    x_safe = torch.where((x.abs() < eps) & (z.abs() < eps), x + eps, x)
    u = torch.atan2(x_safe, -z) / (2.0 * math.pi) + 0.5      # [0,1]
    v = torch.acos(torch.clamp(y, -1.0 + eps, 1.0 - eps)) / math.pi  # [0,1]
    return torch.stack([u * 2.0 - 1.0, v * 2.0 - 1.0], dim=-1)


class EnvironmentMap(nn.Module):
    """Environment map equirretangular; a piramide de mips (pre-filtragem por rugosidade) e refeita por avg_pool a cada forward."""

    def __init__(self, resolution=128, num_levels=6, init_value=0.5, init_std=0.02):
        super().__init__()
        # O ruido evita um mapa uniforme, sob o qual a rugosidade nao e identificavel.
        base = torch.full((3, resolution, resolution * 2), float(init_value))
        if init_std > 0:
            base = base + torch.randn_like(base) * init_std
        self.base = nn.Parameter(base)
        self.num_levels = num_levels

    def _build_pyramid(self):
        levels = [self.base[None]]  # [1,3,H,W]
        current = self.base[None]
        for _ in range(self.num_levels - 1):
            if min(current.shape[-2:]) <= 2:
                break
            current = F.avg_pool2d(current, kernel_size=2)
            levels.append(current)
        return levels

    @staticmethod
    def _sample_level(level, uv):
        """grid_sample num nivel. uv: [N,2] em [-1,1]."""
        grid = uv[None, :, None, :]  # [1,N,1,2]
        out = F.grid_sample(level, grid, mode="bilinear",
                            padding_mode="border", align_corners=True)
        return out[0, :, :, 0].transpose(0, 1)  # [N,3]

    def sample(self, dirs, roughness):
        """Radiancia incidente pre-filtrada para direcoes dirs [N,3] e rugosidade [N,1]."""
        uv = direction_to_equirect_uv(dirs)
        pyramid = self._build_pyramid()
        max_level = len(pyramid) - 1

        # sqrt(rho) concentra os niveis onde a nitidez muda mais.
        level = torch.clamp(torch.sqrt(roughness.clamp(min=0.0)) * max_level, 0.0, float(max_level))
        lo = torch.floor(level).long().clamp(0, max_level)
        hi = torch.clamp(lo + 1, max=max_level)
        w = (level - lo.float())  # [N,1]

        stacked = torch.stack([self._sample_level(lv, uv) for lv in pyramid], dim=0)  # [L,N,3]
        idx_lo = lo.expand(-1, 3)[None]  # [1,N,3]
        idx_hi = hi.expand(-1, 3)[None]
        c_lo = torch.gather(stacked, 0, idx_lo)[0]
        c_hi = torch.gather(stacked, 0, idx_hi)[0]
        return torch.clamp((1.0 - w) * c_lo + w * c_hi, min=0.0)

    @torch.no_grad()
    def as_image(self):
        """Mapa base como imagem [3,H,W]."""
        return self.base.detach().clamp(min=0.0)


class SHLighting(nn.Module):
    """Iluminacao em SH, atenuada por banda a_l(rho) = exp(-l(l+1) rho^2 / 2); so baixa frequencia."""

    def __init__(self, degree=3, init_value=0.5, init_std=0.02):
        super().__init__()
        self.degree = degree
        num_coeffs = (degree + 1) ** 2
        coeffs = torch.zeros(3, num_coeffs)
        coeffs[:, 0] = init_value / 0.28209479177387814
        # Ruido nos graus >= 1: com c_lm = 0 o gradiente da rugosidade seria nulo.
        if init_std > 0 and num_coeffs > 1:
            coeffs[:, 1:] = torch.randn(3, num_coeffs - 1) * init_std
        self.coeffs = nn.Parameter(coeffs)

    def sample(self, dirs, roughness):
        n = dirs.shape[0]
        band = torch.arange(self.degree + 1, device=dirs.device, dtype=dirs.dtype)
        per_band = torch.exp(-band * (band + 1.0) * roughness.pow(2) / 2.0)  # [N,deg+1]
        repeats = torch.tensor([2 * int(l) + 1 for l in range(self.degree + 1)], device=dirs.device)
        atten = torch.repeat_interleave(per_band, repeats, dim=1)  # [N,(deg+1)^2]

        coeffs = self.coeffs[None].expand(n, -1, -1) * atten[:, None, :]
        return torch.clamp(eval_sh(self.degree, coeffs, dirs), min=0.0)

    @torch.no_grad()
    def as_image(self, resolution=128):
        """Renderiza a SH como equirretangular."""
        device = self.coeffs.device
        v = (torch.arange(resolution, device=device, dtype=torch.float32) + 0.5) / resolution * math.pi
        u = (torch.arange(2 * resolution, device=device, dtype=torch.float32) + 0.5) / (2 * resolution)
        u = (u - 0.5) * 2.0 * math.pi
        vv, uu = torch.meshgrid(v, u, indexing="ij")
        dirs = torch.stack([torch.sin(vv) * torch.sin(uu),
                            torch.cos(vv),
                            -torch.sin(vv) * torch.cos(uu)], dim=-1).reshape(-1, 3)
        rough = torch.zeros(dirs.shape[0], 1, device=device)
        return self.sample(dirs, rough).reshape(resolution, 2 * resolution, 3).permute(2, 0, 1)


def build_lighting(opt):
    """Fabrica a partir dos argumentos de linha de comando."""
    if getattr(opt, "light_repr", "envmap") == "sh":
        return SHLighting(degree=getattr(opt, "light_sh_degree", 3))
    return EnvironmentMap(resolution=getattr(opt, "envmap_res", 128),
                          num_levels=getattr(opt, "envmap_levels", 6))
