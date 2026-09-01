"""Representacoes de iluminacao incidente para o termo especular.

Duas implementacoes atras da mesma interface `sample(dirs, roughness) -> [N,3]`:

  EnvironmentMap  mapa equirretangular HDR com piramide de mips. Representa alta
                  frequencia (reflexos agudos). Default.
  SHLighting      iluminacao em harmonicas esfericas com convolucao dependente da
                  rugosidade. Barata (27-48 parametros), mas so baixa frequencia.

A comparacao entre as duas e a ablacao E4 do plano e responde uma pergunta concreta:
quanta alta frequencia a iluminacao de um testemunho de rocha realmente exige?

Nenhuma dependencia externa (sem nvdiffrast): tudo com F.grid_sample e avg_pool.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.sh_utils import eval_sh


def direction_to_equirect_uv(dirs):
    """Direcao unitaria -> coordenadas (u,v) do mapa equirretangular, em [-1,1].

    Convencao: y para cima, v=0 no polo superior. A escolha exata e arbitraria desde
    que consistente -- o mapa aprendido se acomoda a ela.

    NaN silencioso (achado 2026-09-01, run E1): esta parametrizacao tem DUAS
    singularidades de gradiente coincidentes no polo (x=z=0, y=+-1):
      1. d(acos)/dy = -1/sqrt(1-y^2) diverge quando y -> +-1.
      2. d(atan2)/d(x,z) = (z,-x)/(x^2+z^2) e uma forma 0/0 quando x=z=0 -- vira NaN,
         nao apenas grande, porque numerador E denominador zeram juntos.
    Com >1e5 Gaussianas por iteracao, alguma direcao de reflexao cai exatamente ou perto
    o bastante do polo para essas formas degenerarem em fp32. Uma unica Gaussiana com
    gradiente NaN contamina rotation/scaling dela via Adam, dai a cor de toda imagem que
    a inclui, e a partir dai a perda inteira -- foi exatamente o que aconteceu no run E1
    (Truck, --brdf --light_frame world): loss finito ate a iteracao 3010, NaN a partir da
    3020 (a primeira leva de iteracoes com o ramo especular ligado). Reproduzido
    isoladamente para as duas formas (ver docs/06_AUDITORIA.md, achado A-5).

    Corrigido afastando a direcao do polo por um epsilon antes de atan2/acos: o vies
    introduzido e da ordem de eps radianos, muito abaixo da resolucao de um texel do
    envmap, e o gradiente perto do polo fica grande mas finito -- o que o Adam absorve
    normalmente (e para o que ele foi desenhado), ao contrario de NaN/Inf.
    """
    eps = 1e-4
    x, y, z = dirs[..., 0], dirs[..., 1], dirs[..., 2]
    # So importa perto do polo (x=z=0): af asta o componente horizontal de zero sem
    # alterar visivelmente a direcao em nenhum outro ponto (|x|,|z| tipicamente O(1)).
    x_safe = torch.where((x.abs() < eps) & (z.abs() < eps), x + eps, x)
    u = torch.atan2(x_safe, -z) / (2.0 * math.pi) + 0.5      # [0,1]
    v = torch.acos(torch.clamp(y, -1.0 + eps, 1.0 - eps)) / math.pi  # [0,1]
    return torch.stack([u * 2.0 - 1.0, v * 2.0 - 1.0], dim=-1)


class EnvironmentMap(nn.Module):
    """Environment map equirretangular com piramide de mips para pre-filtragem.

    Aproximacao "split-sum": em vez de integrar o lobulo especular a cada amostra,
    pre-filtramos o mapa em niveis progressivamente mais borrados e escolhemos o nivel
    pela rugosidade. Uma Gaussiana rugosa le um mip borrado (lobulo largo); uma polida
    le o nivel 0 (reflexo agudo).

    A piramide e reconstruida a cada forward por avg_pool sucessivo, entao e
    diferenciavel e nao adiciona parametros: so os do mapa base sao otimizados.
    """

    def __init__(self, resolution=128, num_levels=6, init_value=0.5, init_std=0.02):
        super().__init__()
        # Cinza quase uniforme: sem hipotese sobre ONDE esta a luz.
        #
        # O ruido init_std nao e cosmetico. Com um mapa exatamente uniforme, todos os
        # niveis de mip sao identicos, entao a interpolacao por rugosidade devolve sempre
        # o mesmo valor e dL/drho = 0: a rugosidade e NAO-IDENTIFICAVEL sob luz uniforme.
        # Comecar de um ponto exatamente degenerado deixa rho parado ate a iluminacao
        # ganhar estrutura por conta propria. Ver test_roughness_is_unidentifiable_under_uniform_light.
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
        # padding_mode="border" no eixo v (polos); a costura em u tem erro de meio texel,
        # desprezivel na resolucao usada.
        out = F.grid_sample(level, grid, mode="bilinear",
                            padding_mode="border", align_corners=True)
        return out[0, :, :, 0].transpose(0, 1)  # [N,3]

    def sample(self, dirs, roughness):
        """Radiancia incidente pre-filtrada.

        Args:
            dirs: [N,3] direcoes de reflexao unitarias, no referencial da luz.
            roughness: [N,1] em [0,1].
        Returns:
            [N,3] radiancia (nao-negativa).
        """
        uv = direction_to_equirect_uv(dirs)
        pyramid = self._build_pyramid()
        max_level = len(pyramid) - 1

        # sqrt(rho) distribui os niveis de forma mais perceptualmente uniforme que rho
        # linear: a maior parte da mudanca visual de nitidez acontece em rho baixo.
        level = torch.clamp(torch.sqrt(roughness.clamp(min=0.0)) * max_level, 0.0, float(max_level))
        lo = torch.floor(level).long().clamp(0, max_level)
        hi = torch.clamp(lo + 1, max=max_level)
        w = (level - lo.float())  # [N,1]

        # Amostra todos os niveis e seleciona: mais simples e mais rapido que agrupar por
        # nivel, porque a piramide e minuscula perto de N (~1e6 Gaussianas).
        stacked = torch.stack([self._sample_level(lv, uv) for lv in pyramid], dim=0)  # [L,N,3]
        idx_lo = lo.expand(-1, 3)[None]  # [1,N,3]
        idx_hi = hi.expand(-1, 3)[None]
        c_lo = torch.gather(stacked, 0, idx_lo)[0]
        c_hi = torch.gather(stacked, 0, idx_hi)[0]
        return torch.clamp((1.0 - w) * c_lo + w * c_hi, min=0.0)

    @torch.no_grad()
    def as_image(self):
        """Mapa base como imagem [3,H,W] para logging/relighting."""
        return self.base.detach().clamp(min=0.0)


class SHLighting(nn.Module):
    """Iluminacao em harmonicas esfericas com convolucao dependente da rugosidade.

    L_s(r, rho) = sum_l a_l(rho) sum_m c_lm Y_lm(r),  a_l(rho) = exp(-l(l+1) rho^2 / 2)

    O fator a_l e a convolucao da SH com um lobulo de largura crescente em rho: e o
    analogo continuo dos mips do EnvironmentMap. Modos altos morrem primeiro, entao
    rugosidade alta produz iluminacao suave -- exatamente o comportamento fisico.

    Muito mais barato que o envmap (3*(deg+1)^2 parametros no total), mas incapaz de
    representar reflexo agudo. Se esta variante empatar com o envmap nas cenas de rocha,
    isso e um resultado, nao uma falha.
    """

    def __init__(self, degree=3, init_value=0.5, init_std=0.02):
        super().__init__()
        self.degree = degree
        num_coeffs = (degree + 1) ** 2
        coeffs = torch.zeros(3, num_coeffs)
        # Termo DC tal que a radiancia inicial seja init_value uniforme.
        coeffs[:, 0] = init_value / 0.28209479177387814
        # Ruido nos graus >= 1 pelo mesmo motivo do EnvironmentMap, e aqui e ainda mais
        # grave: a atenuacao a_l(rho) MULTIPLICA c_lm, entao com c_lm = 0 o gradiente da
        # rugosidade e exatamente zero (nao apenas ruido numerico) e rho jamais sairia
        # do lugar por esse caminho.
        if init_std > 0 and num_coeffs > 1:
            coeffs[:, 1:] = torch.randn(3, num_coeffs - 1) * init_std
        self.coeffs = nn.Parameter(coeffs)

    def sample(self, dirs, roughness):
        n = dirs.shape[0]
        # Atenuacao por banda. eval_sh espera [N,3,(deg+1)^2].
        band = torch.arange(self.degree + 1, device=dirs.device, dtype=dirs.dtype)
        per_band = torch.exp(-band * (band + 1.0) * roughness.pow(2) / 2.0)  # [N,deg+1]
        repeats = torch.tensor([2 * int(l) + 1 for l in range(self.degree + 1)], device=dirs.device)
        atten = torch.repeat_interleave(per_band, repeats, dim=1)  # [N,(deg+1)^2]

        coeffs = self.coeffs[None].expand(n, -1, -1) * atten[:, None, :]
        return torch.clamp(eval_sh(self.degree, coeffs, dirs), min=0.0)

    @torch.no_grad()
    def as_image(self, resolution=128):
        """Renderiza a SH como equirretangular, para logging comparavel ao envmap."""
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
