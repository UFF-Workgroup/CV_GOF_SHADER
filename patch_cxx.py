import re

def patch_file(filepath, replacements):
    with open(filepath, 'r') as f:
        content = f.read()
    for old, new in replacements:
        content = re.sub(old, new, content)
    with open(filepath, 'w') as f:
        f.write(content)

base = "submodules/diff-gaussian-rasterization/"

# 1. Headers C++ (rasterize_points.h e rasterizer.h)
patch_file(base + "rasterize_points.h", [
    (r'(const torch::Tensor&\s*cov3D_precomp,\s*)(const torch::Tensor&\s*view2gaussian_precomp)',
     r'\1const torch::Tensor& specular_tint,\n\tconst torch::Tensor& roughness,\n\tconst torch::Tensor& residual_color,\n\t\2')
])

patch_file(base + "cuda_rasterizer/rasterizer.h", [
    (r'(const float\*\s*cov3D_precomp,\s*)(const float\*\s*view2gaussian_precomp)',
     r'\1const float* specular_tint,\n\t\t\tconst float* roughness,\n\t\t\tconst float* residual_color,\n\t\t\t\2')
])

# 2. Implementações C++ (rasterize_points.cu e rasterizer_impl.cu)
patch_file(base + "rasterize_points.cu", [
    (r'(const torch::Tensor&\s*cov3D_precomp,\s*)(const torch::Tensor&\s*view2gaussian_precomp)',
     r'\1const torch::Tensor& specular_tint,\n\tconst torch::Tensor& roughness,\n\tconst torch::Tensor& residual_color,\n\t\2'),
    (r'(cov3D_precomp\.contiguous\(\)\.data<float>\(\),\s*)(view2gaussian_precomp\.contiguous\(\)\.data<float>\(\))',
     r'\1specular_tint.contiguous().data<float>(),\n\t\troughness.contiguous().data<float>(),\n\t\tresidual_color.contiguous().data<float>(),\n\t\t\2')
])

patch_file(base + "cuda_rasterizer/rasterizer_impl.cu", [
    (r'(const float\*\s*cov3D_precomp,\s*)(const float\*\s*view2gaussian_precomp)',
     r'\1const float* specular_tint,\n\tconst float* roughness,\n\tconst float* residual_color,\n\t\2')
])

# 3. Ponte Autograd do Python (__init__.py do rasterizador)
patch_file(base + "diff_gaussian_rasterization/__init__.py", [
    (r'(cov3Ds_precomp,\s*)(view2gaussian_precomp)',
     r'\1specular_tint,\n        roughness,\n        residual_color,\n        \2'),
    (r'(grad_cov3Ds_precomp,\s*)(grad_view2gaussian_precomp)',
     r'\1None, # grad_specular\n            None, # grad_roughness\n            None, # grad_residual\n            \2')
])

print("Patch cirúrgico aplicado com sucesso!")