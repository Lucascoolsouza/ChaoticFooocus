# Scientific Disco Diffusion Usage Guide

## 🧬 Como Funciona Cientificamente

### 1. Modelo de Difusão Base
O modelo trabalha com a equação de difusão reversa:
```
p_θ(x_{t-1} | x_t) = N(x_{t-1}; μ_θ(x_t, t), Σ_θ(x_t, t))
```

### 2. CLIP Guidance (Core Algorithm)
A cada step da difusão:
1. **Predição x₀**: Calcula a imagem limpa atual usando DDIM
2. **Decodificação**: Converte latent → RGB para análise CLIP
3. **Cutouts**: Cria N recortes aleatórios da imagem (fractal analysis)
4. **CLIP Embedding**: Codifica cutouts e texto no espaço CLIP
5. **Spherical Loss**: Calcula distância esférica entre embeddings
6. **Gradiente**: Aplica gradiente no latent para guiar geração

```python
# Spherical distance loss (core Disco Diffusion)
def spherical_dist_loss(x, y):
    x = F.normalize(x, dim=-1)
    y = F.normalize(y, dim=-1)
    return (x - y).norm(dim=-1).div(2).arcsin().pow(2).mul(2)
```

### 3. Transformações Geométricas
Aplicadas durante cada step:
- **Translate**: Movimento linear `T = [1,0,tx; 0,1,ty]`
- **Rotate**: Rotação `R = [cos(θ),-sin(θ),0; sin(θ),cos(θ),0]`
- **Zoom**: Escala `S = [sx,0,0; 0,sy,0]`

### 4. Losses Auxiliares
- **TV Loss**: `∇²I` para suavidade
- **Range Loss**: Mantém pixels em [-1,1]
- **Saturation Loss**: Controla saturação de cores

## 🎛️ Parâmetros Científicos

### CLIP Guidance Scale (disco_scale)
- **500-1000**: Guidance suave, mais artístico
- **1000-1500**: Guidance moderado, equilibrado
- **1500-2000+**: Guidance forte, mais literal ao prompt

### Cutouts (cutn)
- **8-16**: Análise básica, mais rápido
- **16-32**: Análise detalhada, qualidade média
- **32-64**: Análise intensiva, máxima qualidade

### Total Variation Scale (tv_scale)
- **0**: Sem suavização (pode gerar ruído)
- **50-150**: Suavização moderada
- **150-300**: Suavização forte

### Range Scale (range_scale)
- **50-150**: Controle suave de range
- **150-300**: Controle forte de range

## 🧪 Configurações Científicas Recomendadas

### Para Máxima Qualidade Científica
```python
disco_settings = {
    'disco_scale': 2000.0,      # CLIP guidance forte
    'cutn': 40,                 # Muitos cutouts
    'tv_scale': 150.0,          # Suavização moderada
    'range_scale': 300.0,       # Controle forte de range
    'disco_transforms': ['translate', 'rotate', 'zoom'],
    'disco_rotation_speed': 0.1,
    'disco_zoom_factor': 1.02,
    'disco_translation_x': 0.05,
    'disco_translation_y': 0.05
}
```

### Para Efeitos Fractais
```python
fractal_settings = {
    'disco_scale': 1500.0,
    'cutn': 32,
    'tv_scale': 100.0,
    'disco_transforms': ['zoom', 'rotate'],
    'disco_zoom_factor': 1.05,  # Zoom mais agressivo
    'disco_symmetry_mode': 'radial'
}
```

### Para Análise Detalhada (Slow but Scientific)
```python
detailed_settings = {
    'disco_scale': 2500.0,      # Guidance máximo
    'cutn': 64,                 # Cutouts máximos
    'tv_scale': 200.0,
    'range_scale': 400.0,
    'cut_pow': 0.8,             # Cutouts menores
    'disco_transforms': ['translate', 'rotate', 'zoom']
}
```

## 🔬 Prompts Científicos Recomendados

### Para Fractais Matemáticos
```
"mandelbrot set, fractal mathematics, infinite recursion, golden ratio, fibonacci spiral"
```

### Para Psicodelia Científica
```
"neural network visualization, synaptic connections, brain waves, consciousness patterns"
```

### Para Geometria Sagrada
```
"sacred geometry, platonic solids, geometric patterns, mathematical beauty"
```

## ⚡ Performance vs Qualidade

### Rápido (para testes)
- `cutn`: 8-16
- `disco_scale`: 500-1000
- `tv_scale`: 50

### Equilibrado (uso geral)
- `cutn`: 16-24
- `disco_scale`: 1000-1500
- `tv_scale`: 100-150

### Máxima Qualidade (render final)
- `cutn`: 32-64
- `disco_scale`: 1500-2500
- `tv_scale`: 150-300

## 🧮 Fórmulas Matemáticas Implementadas

### CLIP Loss
```
L_clip = 1 - cos(CLIP(image_cutouts), CLIP(text_prompt))
```

### Total Variation Loss
```
L_tv = Σ|∇_x I|² + |∇_y I|²
```

### Range Loss
```
L_range = Σ(I - clamp(I, -1, 1))²
```

### Loss Total
```
L_total = λ_clip * L_clip + λ_tv * L_tv + λ_range * L_range
```

## 🎯 Resultados Esperados

Com as configurações científicas corretas, você deve ver:
- **Detalhes fractais** emergindo naturalmente
- **Coerência semântica** com o prompt
- **Transformações suaves** entre frames
- **Qualidade artística** superior a filtros simples

O algoritmo verdadeiro produz resultados qualitativamente diferentes de simples filtros visuais!