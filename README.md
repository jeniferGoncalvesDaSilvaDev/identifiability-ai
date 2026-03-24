# RepresentationMetrics

> **"A capacidade de um modelo distinguir entradas está diretamente ligada à estrutura do seu operador — a não-unicidade é consequência de direções no núcleo ou de espectro degenerado."**

Ferramenta para análise de **identificabilidade e estrutura informacional** de representações internas em qualquer modelo PyTorch.

Funciona com classificadores, redes convolucionais, transformers, redes de política RL, autoencoders, redes quântico-clássicas híbridas — qualquer `nn.Module`.

---

## Índice

1. [Por que isso importa](#1-por-que-isso-importa)
2. [Fundamentos teóricos](#2-fundamentos-teóricos)
3. [Instalação](#3-instalação)
4. [Início rápido](#4-início-rápido)
5. [Arquitetura do código](#5-arquitetura-do-código)
6. [Métricas — fórmulas e interpretação](#6-métricas--fórmulas-e-interpretação)
7. [Parâmetros detalhados](#7-parâmetros-detalhados)
8. [O que são hooks e por que importam](#8-o-que-são-hooks-e-por-que-importam)
9. [Aplicações por domínio](#9-aplicações-por-domínio)
10. [Monitoramento ao longo do treino](#10-monitoramento-ao-longo-do-treino)
11. [Limitações e cuidados](#11-limitações-e-cuidados)
12. [Referências](#12-referências)

---

## 1. Por que isso importa

Quando você treina um modelo, normalmente monitora apenas a **saída final**: loss, acurácia, reward. Essas métricas dizem *o que* o modelo produz, mas não *como* ele representa internamente as informações.

Dois modelos com a mesma loss podem ter representações internas radicalmente diferentes:

- Um pode usar **64 dimensões efetivas** de um espaço de 128 — representação rica e distribuída
- O outro pode ter **colapsado para 3 dimensões efetivas** — modelo "esqueceu" como distinguir inputs diferentes

O segundo vai parar de aprender mais cedo, generalizar pior, e a loss não vai te contar isso.

Esta biblioteca mede exatamente essa diferença.

---

## 2. Fundamentos teóricos

### Identificabilidade

Um modelo é **identificável** se não existem dois inputs distintos que produzam a mesma saída:

```
T(x₁) = T(x₂)  ⟹  x₁ = x₂
```

Quando isso não vale, existe um **núcleo não-trivial**:

```
ker(T) = { x : T(x) = 0 }
```

Qualquer vetor no núcleo é "invisível" para o modelo — ele pode ser adicionado ao input sem mudar nada na saída.

### Conexão com álgebra linear

Para uma camada linear `A`:

| Conceito clássico | Interpretação aqui |
|---|---|
| `rank(A)` | Dimensões que o modelo realmente usa |
| `ker(A)` | Direções que o modelo ignora |
| Autovalores ≈ 0 | "Quase núcleo" — direções quase invisíveis |
| Espectro degenerado | Representação colapsada |

### Generalização para operadores não-lineares

Em redes profundas, a camada não é linear — mas podemos analisar seu **Jacobiano local**:

```
J(h) = ∂saída/∂h   avaliado em h
```

O núcleo do Jacobiano são as direções locais que não afetam a saída. A nulidade funcional desta biblioteca é uma estimativa Monte Carlo desse núcleo.

### Decomposição SVD

Para uma matriz de embeddings `X ∈ ℝ^(n×d)`:

```
X = U Σ Vᵀ

onde:
  U ∈ ℝ^(n×r)  — direções nos dados (amostras)
  Σ ∈ ℝ^(r×r)  — valores singulares (importância de cada direção)
  Vᵀ ∈ ℝ^(r×d) — direções no espaço de features
  r = min(n, d)
```

Os valores singulares `σ₁ ≥ σ₂ ≥ ... ≥ σᵣ ≥ 0` quantificam quanto cada direção do espaço de features é usada. Valores singulares próximos de zero = direções no "quase núcleo".

---

## 3. Instalação

Sem dependências além do stack padrão de ML:

```bash
pip install torch numpy matplotlib
```

Copie `representation_metrics.py` para o seu projeto:

```python
from representation_metrics import ModelAnalyzer, RepresentationMetrics, plot_spectral_decay
```

---

## 4. Início rápido

```python
import torch
from representation_metrics import ModelAnalyzer

model = MinhaRede()
x = torch.randn(32, 64)

analyzer = ModelAnalyzer(model)

# ver todas as camadas disponíveis
analyzer.print_layers()

# analisar camadas específicas
report = analyzer.analyze(x, layers=["encoder", "fc"])
report.print()
```

Saída:

```
==================================================
  Modelo: MinhaRede
==================================================

── encoder ──
  Rank efetivo        : 18.43
  Dimensões necessárias: 22 / 64
  Taxa de compressão  : 34.4%
  Nulidade funcional  : 12.5%
  Sensibilidade       : 0.0834
  Δ saída (compressão): 0.0021

── fc ──
  Rank efetivo        : 6.71
  Dimensões necessárias: 8 / 64
  Taxa de compressão  : 12.5%
  Nulidade funcional  : 43.8%
  Sensibilidade       : 0.2341
  Δ saída (compressão): 0.0103
==================================================
```

---

## 5. Arquitetura do código

```
representation_metrics.py
│
├── RepresentationMetrics       # métricas puras — tensores, sem modelo
│     ├── compute_svd()
│     ├── effective_rank()
│     ├── spectral_decay()
│     ├── compression_ratio()
│     ├── functional_nullity()
│     ├── sensitivity()
│     └── compression_delta()
│
├── HookManager                 # captura ativações sem modificar o modelo
│     ├── register()
│     ├── remove_all()
│     └── capture()             # context manager
│
├── ModelAnalyzer               # interface principal — orquestra tudo
│     ├── __init__()
│     ├── list_layers()
│     ├── print_layers()
│     ├── analyze()
│     ├── _analyze_layer()
│     ├── _infer_head()
│     └── _default_layers()
│
├── LayerReport                 # resultado por camada (dataclass)
├── AnalysisReport              # resultado completo (dataclass)
│
└── plot_spectral_decay()       # visualização
```

### Por que três camadas?

**`RepresentationMetrics`** é stateless — não guarda nada, só recebe tensores e devolve números. Isso significa que você pode usá-la em qualquer contexto: PyTorch, JAX com conversão, numpy puro, saída de circuito quântico. Sem acoplamento.

**`HookManager`** resolve o problema de observação sem intrusão. Você não precisa reescrever nenhum modelo — o manager se conecta e desconecta de forma segura.

**`ModelAnalyzer`** é a cola. Ela usa as duas acima e expõe uma interface simples de alto nível.

---

## 6. Métricas — fórmulas e interpretação

### 6.1 Rank Efetivo

**Fórmula:**

```
rank_eff(S) = (Σᵢ σᵢ)² / Σᵢ σᵢ²
```

Derivada da **desigualdade de Cauchy-Schwarz**. Se todos os valores singulares forem iguais (`σ₁ = σ₂ = ... = σᵣ`), o rank efetivo é exatamente `r` (máximo). Se um único valor singular dominar todos os outros, o rank efetivo se aproxima de 1.

**Por que não usar o rank matricial clássico?**

O rank clássico conta dimensões com `σ > 0`, mas em floats isso é sempre `r` — todo valor singular é ligeiramente não-zero por ruído numérico. O rank efetivo mede a *distribuição* da energia, não apenas a presença.

**Interpretação:**

| Valor | Situação |
|---|---|
| Próximo de `d` | Representação distribuída, rica em informação |
| 30–60% de `d` | Normal para redes treinadas — alguma compressão é esperada |
| < 10% de `d` | Colapso — modelo usa poucas direções |
| Próximo de 1 | Colapso severo — representação quase escalar |

**Código:**
```python
def effective_rank(S: torch.Tensor) -> float:
    num = torch.sum(S) ** 2          # (Σ σᵢ)²
    den = torch.sum(S ** 2) + 1e-8   # Σ σᵢ²  +  ε para evitar div/0
    return (num / den).item()
```

O `1e-8` é um epsilon de estabilidade numérica — sem ele, um vetor zero causaria divisão por zero. `.item()` converte tensor escalar para float Python nativo.

---

### 6.2 Decaimento Espectral

**Fórmula:**

```
decay[i] = σᵢ / Σⱼ σⱼ
```

Normaliza os valores singulares para que a soma seja 1, como uma distribuição de probabilidade discreta. Isso permite comparar o "formato" do espectro entre camadas e modelos de tamanhos diferentes.

**Interpretação visual:**

```
Decaimento saudável:     Colapso:
█                        ████████
██                       █
███                      █
████                     █
█████                    █
(gradual)                (um componente domina tudo)
```

**Código:**
```python
def spectral_decay(S: torch.Tensor) -> np.ndarray:
    s = S / (S.sum() + 1e-8)   # normaliza para somar 1
    return s.cpu().numpy()      # .cpu() obrigatório antes de .numpy()
                                # (numpy não fala com tensores na GPU)
```

---

### 6.3 Taxa de Compressão

**Fórmula:**

```
variância[i] = σᵢ²
variância_acumulada[k] = Σᵢ₌₁ᵏ σᵢ²

k* = min{ k : variância_acumulada[k] ≥ θ · Σᵢ σᵢ² }

taxa = k* / d
```

Onde `θ` é o threshold (padrão 0.90 = 90%).

**Por que `σ²` e não `σ`?**

Porque os valores singulares ao quadrado são proporcionais à variância explicada por cada componente — é a mesma lógica do PCA. `σ` seria a "amplitude", `σ²` é a "energia".

**`searchsorted` — como funciona:**

```python
cumulative = [10, 18, 24, 28, 30]   # soma acumulada
threshold  = 0.90 * 30 = 27

searchsorted([10,18,24,28,30], 27)  # → índice 3
k = 3 + 1 = 4                       # +1 porque índice é base-0
```

**Código:**
```python
def compression_ratio(S, variance_threshold=0.90):
    var = S ** 2                                    # variância por componente
    total = var.sum()                               # variância total
    cumulative = torch.cumsum(var, dim=0)           # acumulada

    k = int(torch.searchsorted(
        cumulative,
        variance_threshold * total                  # threshold absoluto
    ).item()) + 1

    k = min(k, len(S))                              # garante k ≤ d
    return k / len(S), k                            # (razão, absoluto)
```

---

### 6.4 Nulidade Funcional

**Definição:**

Estimativa Monte Carlo da fração do espaço de embedding que está no núcleo funcional:

```
nullity ≈ |{ v : ||T(h + ε·v̂) - T(h)|| < δ }| / N

onde:
  v̂  = vetor unitário aleatório
  ε  = magnitude da perturbação (padrão 1e-3)
  δ  = threshold de "invisibilidade" (1e-4)
  N  = número de amostras (padrão 32)
```

**Diferença entre núcleo algébrico e funcional:**

- **Núcleo algébrico:** `ker(A) = { x : Ax = 0 }` — definição linear exata
- **Núcleo funcional:** direções que não mudam a saída de forma mensurável — inclui não-linearidades, saturações, estrutura da tarefa

O núcleo funcional é mais relevante na prática porque captura invariâncias reais do modelo, não só propriedades lineares da matriz de pesos.

**Código:**
```python
def functional_nullity(forward_fn, h, epsilon=1e-3, n_directions=32):
    h = h.detach()                          # remove do grafo — não queremos gradientes
    base = forward_fn(h)                    # saída de referência

    null_count = 0
    for _ in range(n_directions):
        v = F.normalize(                    # vetor unitário aleatório
            torch.randn_like(h), dim=-1
        )
        h_perturbed = h + epsilon * v

        delta = torch.norm(
            forward_fn(h_perturbed) - base, # diferença na saída
            dim=-1
        ).mean().item()

        if delta < 1e-4:                    # "invisível" para o modelo
            null_count += 1

    return null_count / n_directions        # fração de direções invisíveis
```

**Por que normalizar `v`?**

Sem normalização, direções com maior magnitude teriam perturbações maiores — comparação injusta. Com `F.normalize`, toda direção tem norma 1, então a perturbação real é sempre exatamente `epsilon`.

---

### 6.5 Sensibilidade

**Fórmula:**

```
sensibilidade = E_batch[ ||∂saída/∂h|| ]

onde a norma é a norma L2 do gradiente
```

Mede o quanto a saída muda para perturbações infinitesimais no embedding — é a norma do Jacobiano local, calculada via backpropagation.

**Diferença para nulidade funcional:**

| Métrica | O que mede | Como calcula |
|---|---|---|
| Nulidade funcional | Direções discretas, perturbações finitas | Amostragem Monte Carlo |
| Sensibilidade | Magnitude total do gradiente | Backprop (diferencial) |

As duas se complementam: nulidade funcional encontra direções ignoradas, sensibilidade quantifica a intensidade da resposta global.

**Código:**
```python
def sensitivity(forward_fn, h):
    h_grad = h.clone().detach().float().requires_grad_(True)
    #         ↑ copia   ↑ sai do grafo  ↑ fp32  ↑ habilita gradiente

    out = forward_fn(h_grad)
    scalar = out.max(dim=-1)[0].mean()  # proxy escalar: logit máximo médio
    scalar.backward()                   # calcula ∂scalar/∂h_grad

    return torch.norm(h_grad.grad, dim=-1).mean().item()
```

**Por que `out.max()` como proxy?**

Precisamos de um escalar para chamar `.backward()`. O logit máximo representa a "decisão mais confiante" do modelo — perturbações que mudam esse valor são semanticamente significativas.

---

### 6.6 Delta de Compressão

**Fórmula:**

```
h_k = PCA(h, k)  =  h̄ + (h - h̄) Vₖ Vₖᵀ

delta = E_batch[ ||T(h) - T(h_k)|| ]

onde:
  h̄  = média das amostras
  Vₖ = k primeiros vetores singulares direitos (componentes principais)
```

Projeta o embedding no subespaço das `k` componentes principais e mede o impacto na saída. Conecta estrutura espectral com semântica funcional.

**Por que isso é importante?**

Um rank efetivo baixo com delta alto indica algo crítico: as dimensões importantes para a saída **não** são as dimensões com maior variância espectral. O modelo aprendeu a codificar informação semântica em dimensões "secundárias".

```
delta pequeno: compressão PCA é segura, dimensões descartadas eram redundantes
delta grande:  informação importante estava nas dimensões "menos importantes"
               → modelo tem representação mal-organizada
```

**Código:**
```python
def compression_delta(forward_fn, h, k):
    h = h.float()
    h_centered = h - h.mean(dim=0, keepdim=True)         # centraliza

    _, _, Vt = torch.linalg.svd(h_centered, full_matrices=False)
    V_k = Vt[:k].T                                        # (d, k) — k componentes principais

    # projeção: h_centered → subespaço k-dimensional → volta para d-dimensional
    h_proj = h_centered @ V_k @ V_k.T + h.mean(dim=0, keepdim=True)

    with torch.no_grad():
        out_orig = forward_fn(h)
        out_proj = forward_fn(h_proj)

    return torch.norm(out_orig - out_proj, dim=-1).mean().item()
```

---

## 7. Parâmetros detalhados

### `ModelAnalyzer.__init__()`

| Parâmetro | Tipo | Padrão | Descrição |
|---|---|---|---|
| `model` | `nn.Module` | — | Qualquer modelo PyTorch |
| `output_head` | `Callable` ou `None` | `None` | Função que mapeia embedding → logits. Se `None`, tenta auto-detectar nos atributos `fc`, `head`, `classifier`, `output` |
| `device` | `str` ou `torch.device` | `"cpu"` | Dispositivo de execução |
| `variance_thr` | `float` | `0.90` | Threshold para `compression_ratio`. `0.90` = 90% da variância |

**Sobre `output_head`:**

É a função que transforma o embedding capturado em uma saída interpretável (logits, probabilidades, valor escalar). A biblioteca envolve ela em `softmax` internamente para as métricas funcionais.

Se seu modelo não tem uma cabeça padrão, passe explicitamente:

```python
# rede ator-crítico: dois heads diferentes
analyzer_ator  = ModelAnalyzer(model, output_head=model.policy_head)
analyzer_valor = ModelAnalyzer(model, output_head=model.value_head)

# autoencoder: reconstrução como "saída"
analyzer = ModelAnalyzer(autoencoder, output_head=autoencoder.decoder)

# rede quântico-clássica: camada de medição como saída
analyzer = ModelAnalyzer(modelo_hibrido, output_head=modelo_hibrido.measurement_layer)
```

---

### `ModelAnalyzer.analyze()`

| Parâmetro | Tipo | Padrão | Descrição |
|---|---|---|---|
| `x` | `torch.Tensor` | — | Batch de entrada `(B, ...)` |
| `layers` | `list[str]` ou `None` | `None` | Nomes das camadas a analisar. `None` = auto-detecta camadas folha com parâmetros |
| `n_directions` | `int` | `32` | Amostras para nulidade funcional. Mais = mais preciso, mais lento |
| `epsilon` | `float` | `1e-3` | Magnitude da perturbação na nulidade funcional |

**Sobre `n_directions`:**

A nulidade funcional é uma estimativa. O erro padrão da estimativa de proporção é:

```
σ_erro ≈ sqrt(p(1-p) / N)

Para p=0.5 (pior caso) e N=32:  σ ≈ 0.088  (±8.8%)
Para p=0.5 e N=128:             σ ≈ 0.044  (±4.4%)
Para p=0.5 e N=512:             σ ≈ 0.022  (±2.2%)
```

Use `n_directions=32` para exploração rápida, `n_directions=128` ou mais para resultados publicáveis.

**Sobre `epsilon`:**

Controla o tamanho da perturbação. Muito pequeno (`< 1e-4`) → pode estar abaixo do ruído numérico. Muito grande (`> 0.1`) → perturbações tão grandes que quase sempre afetam a saída, subestimando a nulidade.

`1e-3` é um bom padrão para embeddings normalizados. Se seu embedding tem escala muito diferente, ajuste:

```python
# estima escala do embedding e ajusta epsilon
scale = h.std().item()
report = analyzer.analyze(x, epsilon=scale * 0.01)
```

---

## 8. O que são hooks e por que importam

### O problema de observação

Quando você executa `model(x)`, os dados passam por dezenas de camadas em sequência. Você vê apenas a saída final. As ativações intermediárias existem momentaneamente na memória e são descartadas.

```
input → [conv1] → [relu] → [conv2] → [pool] → [fc] → output
           ↑ você não vê nada aqui ↑
```

### O que um hook faz

Um hook é uma função registrada para ser chamada automaticamente quando a execução passa por um módulo específico:

```python
def minha_funcao(modulo, entrada, saida):
    # chamada automaticamente toda vez que 'modulo' executa
    guardar(saida)

modulo.register_forward_hook(minha_funcao)
```

```
input → [conv1] → [relu] → [conv2] → [pool] → [fc] → output
                     ↑
               hook registrado aqui:
               minha_funcao é chamada com
               (relu_module, entrada, saída_do_relu)
```

### Tipos de hooks no PyTorch

| Hook | Quando é chamado | Para que serve |
|---|---|---|
| `register_forward_hook` | Após o forward de um módulo | Capturar ativações (usado aqui) |
| `register_forward_pre_hook` | Antes do forward | Modificar entradas |
| `register_backward_hook` | Durante o backward | Capturar/modificar gradientes |
| `register_full_backward_hook` | Backward completo | Análise de gradientes mais precisa |

### Por que hooks são a solução certa aqui

Alternativas para capturar ativações intermediárias:

| Abordagem | Problema |
|---|---|
| Reescrever o modelo | Não funciona com modelos pré-treinados/externos |
| Partir o modelo em pedaços | Frágil, precisa conhecer a arquitetura |
| Torchvision `IntermediateLayerGetter` | Funciona só com alguns modelos |
| **Hooks** (nossa solução) | Funciona com qualquer `nn.Module`, sem modificações |

### Ciclo de vida seguro dos hooks

```python
# 1. REGISTRA — hook fica ativo a partir daqui
hook_mgr.register(named["encoder"], "encoder")

# 2. FORWARD PASS — hook é chamado automaticamente aqui
with torch.no_grad():
    model(x)

# 3. REMOVE — OBRIGATÓRIO
# sem isso, o hook fica ativo para sempre:
# - memória vaza (cada forward acumula mais ativações)
# - forward passes futuros ficam mais lentos
hook_mgr.remove_all()
```

O `HookManager` garante limpeza automática. O método `capture()` usa context manager:

```python
with hook_mgr.capture():
    model(x)
# hooks são removidos automaticamente aqui, mesmo se ocorrer exceção
```

### Monitoramento contínuo com hooks

```python
ranks_por_step = []

def monitor_hook(module, input, output):
    h = output.detach().flatten(1)
    S = RepresentationMetrics.compute_svd(h)
    ranks_por_step.append(RepresentationMetrics.effective_rank(S))

hook = model.encoder.register_forward_hook(monitor_hook)

# treino normal — hook coleta dados automaticamente
for batch in dataloader:
    loss = model(batch).loss
    loss.backward()
    optimizer.step()

hook.remove()
```

---

## 9. Aplicações por domínio

### 9.1 Machine Learning clássico com camadas neurais

```python
# regressão logística como rede de 1 camada
class LogisticReg(nn.Module):
    def __init__(self, d_in, n_classes):
        super().__init__()
        self.linear = nn.Linear(d_in, n_classes)
        self.classifier = self.linear

    def forward(self, x):
        return self.linear(x)

model = LogisticReg(20, 3)
analyzer = ModelAnalyzer(model)
report = analyzer.analyze(X_tensor, layers=["linear"])

# rank efetivo alto = features são realmente diversas
# rank efetivo baixo = features são redundantes → considere PCA antes
```

**Caso de uso prático:** antes de treinar um modelo em dados tabulares, analisar o rank efetivo dos embeddings de features ajuda a decidir se pré-processamento (PCA, seleção de features) é necessário.

---

### 9.2 Redes Convolucionais (CNN)

```python
import torchvision.models as models

resnet = models.resnet50(pretrained=True)
analyzer = ModelAnalyzer(resnet, device="cuda")

analyzer.print_layers()

x = torch.randn(32, 3, 224, 224).cuda()
report = analyzer.analyze(
    x,
    layers=["layer1", "layer2", "layer3", "layer4"],
    n_directions=64
)
report.print()
plot_spectral_decay(report)
```

**O que esperar:**
- Camadas iniciais (layer1, layer2): rank efetivo alto — detectores de bordas/texturas são diversos
- Camadas finais (layer4): rank efetivo menor — compressão semântica esperada
- Nulidade funcional alta nas camadas finais = muitas features de layer4 são redundantes para a tarefa

---

### 9.3 Transformers

```python
from transformers import AutoModel

bert = AutoModel.from_pretrained("bert-base-uncased")

analyzer = ModelAnalyzer(
    bert,
    output_head=lambda h: h
)

report = analyzer.analyze(
    input_ids,
    layers=[
        "encoder.layer.0.attention.self",
        "encoder.layer.5.attention.self",
        "encoder.layer.11.attention.self",
    ]
)
```

**Aplicação:** medir colapso de atenção. Quando muitas cabeças de atenção convergem para os mesmos padrões, o rank efetivo das ativações da camada de atenção cai — evidência de redundância entre cabeças.

---

### 9.4 Autoencoders e modelos generativos

```python
class VAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(784, 256), nn.ReLU(),
            nn.Linear(256, 64)
        )
        self.decoder = nn.Sequential(
            nn.Linear(32, 256), nn.ReLU(),
            nn.Linear(256, 784), nn.Sigmoid()
        )
        self.fc_mu  = nn.Linear(64, 32)
        self.fc_var = nn.Linear(64, 32)

    def forward(self, x):
        h = self.encoder(x)
        mu, log_var = self.fc_mu(h), self.fc_var(h)
        z = self.reparametrize(mu, log_var)
        return self.decoder(z), mu, log_var

vae = VAE()
analyzer = ModelAnalyzer(vae, output_head=vae.decoder)
report = analyzer.analyze(x_batch, layers=["encoder.0", "encoder.2", "fc_mu"])
```

**O que monitorar:**
- Rank efetivo de `fc_mu` = dimensionalidade real do espaço latente
- Se for muito menor que `latent_dim` → o VAE está colapsando dimensões latentes (posterior collapse)
- Nulidade funcional de `fc_mu` = quantas dimensões latentes não afetam a reconstrução

---

### 9.5 Reinforcement Learning

#### Rede Ator-Crítico

```python
class AtorCritico(nn.Module):
    def __init__(self, obs_dim, n_acoes):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(obs_dim, 256), nn.ReLU(),
            nn.Linear(256, 128),    nn.ReLU()
        )
        self.policy = nn.Linear(128, n_acoes)
        self.value  = nn.Linear(128, 1)

    def forward(self, obs):
        h = self.encoder(obs)
        return self.policy(h), self.value(h)

model = AtorCritico(obs_dim=48, n_acoes=6)

# dois analyzers com heads diferentes
analyzer_pol = ModelAnalyzer(model, output_head=model.policy)
analyzer_val = ModelAnalyzer(model, output_head=model.value)
```

#### Monitoramento de colapso durante treino RL

```python
historico = {"rank": [], "nullity": [], "reward": [], "steps": []}

for episodio in range(n_episodios):
    reward_ep = treinar_episodio(agente, env)

    if episodio % 50 == 0:
        obs_sample = coletar_observacoes(env, n=64)
        report = analyzer_pol.analyze(obs_sample, layers=["encoder"])

        lr = report.layers["encoder"]
        historico["rank"].append(lr.effective_rank)
        historico["nullity"].append(lr.functional_nullity)
        historico["reward"].append(reward_ep)

        print(f"Ep {episodio:4d} | Reward: {reward_ep:6.1f} | "
              f"Rank: {lr.effective_rank:5.2f} | "
              f"Nulidade: {lr.functional_nullity:.1%}")
```

**Sinal de alerta:** reward estagnando enquanto rank efetivo cai → colapso de representação. O agente perdeu capacidade de distinguir estados.

#### Integração com `AgenteQuantico`

```python
class AgenteQuantico:
    def __init__(self, ...):
        self.rede = MinhaRedeQuantica()
        self.analyzer = ModelAnalyzer(
            self.rede,
            output_head=self.rede.measurement_layer
        )
        self.metricas_treino = []

    def treinar_passo(self, estados, acoes, recompensas):
        # ... lógica UCB e atualização Q ...

        if self.passo % self.intervalo_analise == 0:
            h = torch.tensor(estados, dtype=torch.float32)
            report = self.analyzer.analyze(h, layers=["camada_quantica"])
            self.metricas_treino.append({
                "passo": self.passo,
                "rank": report.layers["camada_quantica"].effective_rank,
                "nullity": report.layers["camada_quantica"].functional_nullity,
            })
```

---

### 9.6 Computação Quântica — integração com PennyLane

Circuitos quânticos parametrizados (PQC) produzem embeddings clássicos após a camada de medição. Esses embeddings podem ser analisados diretamente:

```python
import pennylane as qml

n_qubits = 4
dev = qml.device("default.qubit", wires=n_qubits)

@qml.qnode(dev, interface="torch")
def circuito(inputs, weights):
    qml.AngleEmbedding(inputs, wires=range(n_qubits))
    qml.BasicEntanglerLayers(weights, wires=range(n_qubits))
    return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

class RedeHibrida(nn.Module):
    def __init__(self):
        super().__init__()
        weight_shapes = {"weights": (3, n_qubits)}
        self.quantum_layer = qml.qnn.TorchLayer(circuito, weight_shapes)
        self.classical = nn.Linear(n_qubits, 2)
        self.classifier = self.classical

    def forward(self, x):
        q_out = self.quantum_layer(x)
        return self.classical(q_out)

modelo_hibrido = RedeHibrida()
analyzer = ModelAnalyzer(modelo_hibrido, output_head=modelo_hibrido.classical)

x = torch.randn(32, n_qubits)
report = analyzer.analyze(x, layers=["quantum_layer", "classical"])
report.print()
```

**Por que analisar circuitos quânticos com isso?**

O poder expressivo de um PQC está diretamente ligado ao rank efetivo dos embeddings que ele produz. Um circuito que produz embeddings de baixo rank efetivo está colapsando o espaço de Hilbert — não explorando o entrelaçamento quântico disponível.

```
rank_eff(saída quântica) ≈ n_qubits      → circuito usando bem o entrelaçamento
rank_eff(saída quântica) << n_qubits     → circuito colapsado — equivale a menos qubits
```

**Análise da barreira quântico-clássica:**

```python
report = analyzer.analyze(x, layers=["quantum_layer", "classical"])

rank_quantico = report.layers["quantum_layer"].effective_rank
rank_classico = report.layers["classical"].effective_rank

print(f"Informação retida na transição quântico-clássica: {rank_classico/rank_quantico:.1%}")
```

---

### 9.7 Séries Temporais (LSTM, GRU)

```python
class RedeSerial(nn.Module):
    def __init__(self, input_dim, hidden_dim, n_classes):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.classifier = nn.Linear(hidden_dim, n_classes)

    def forward(self, x):
        out, (h_n, _) = self.lstm(x)
        return self.classifier(h_n.squeeze(0))

model = RedeSerial(10, 64, 5)
analyzer = ModelAnalyzer(model)

# sequências: (batch, seq_len, features)
x = torch.randn(32, 20, 10)
report = analyzer.analyze(x, layers=["lstm", "classifier"])
```

---

### 9.8 Graph Neural Networks (GNN)

```python
from torch_geometric.nn import GCNConv

class GCN(nn.Module):
    def __init__(self, input_dim, hidden_dim, n_classes):
        super().__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, n_classes)
        self.classifier = self.conv2

    def forward(self, x, edge_index):
        h = F.relu(self.conv1(x, edge_index))
        return self.conv2(h, edge_index)

gcn = GCN(16, 64, 7)
analyzer = ModelAnalyzer(gcn)
node_features = torch.randn(100, 16)
report = analyzer.analyze(node_features, layers=["conv1"])
```

---

## 10. Monitoramento ao longo do treino

### Loop completo com histórico

```python
def criar_monitor(model, output_head, layer_name, device="cpu"):
    analyzer = ModelAnalyzer(model, output_head=output_head, device=device)
    historico = {"rank": [], "nullity": [], "sensitivity": [], "epoch": []}

    def registrar(x_sample, epoch):
        report = analyzer.analyze(x_sample.to(device), layers=[layer_name])
        lr = report.layers[layer_name]
        historico["rank"].append(lr.effective_rank)
        historico["nullity"].append(lr.functional_nullity)
        historico["sensitivity"].append(lr.sensitivity or 0.0)
        historico["epoch"].append(epoch)

    return registrar, historico


model = MinhaRede()
optimizer = torch.optim.Adam(model.parameters())
registrar, hist = criar_monitor(model, model.classifier, "encoder")

for epoch in range(n_epochs):
    for x_batch, y_batch in dataloader:
        loss = criterio(model(x_batch), y_batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    x_val, _ = next(iter(val_loader))
    registrar(x_val, epoch)
```

### Sinais de alerta

| Padrão observado | Diagnóstico | Ação sugerida |
|---|---|---|
| Rank efetivo caindo ao longo do treino | Colapso de representação | Weight decay, spectral normalization, reinicialização de camadas |
| Nulidade > 60% em camadas intermediárias | Muitas features redundantes | Reduzir largura, adicionar bottleneck |
| Sensibilidade muito alta + rank baixo | Representação instável e colapsada | Learning rate menor, gradient clipping |
| Delta de compressão alto + rank baixo | Informação semântica em dimensões secundárias | Reorganização do espaço latente |
| Rank estável mas reward estagnando em RL | Colapso não é o problema | Exploração, reward shaping, curriculum |

---

## 11. Limitações e cuidados

**`_flatten` assume batch na dimensão 0.** CNNs com saída `(B, C, H, W)` são achatadas para `(B, C*H*W)`. Para analisar o espaço de canais separadamente, passe um `output_head` que faz `adaptive_avg_pool` antes.

**Hooks capturam a saída do módulo, não a entrada.** Para analisar a entrada de uma camada específica, registre o hook na camada anterior ou use `register_forward_pre_hook`.

**Nulidade funcional é uma estimativa Monte Carlo.** Com `n_directions=32` e embeddings de alta dimensão (`d > 256`), a cobertura é parcial. O erro padrão é `~sqrt(p(1-p)/N)`.

**Métricas funcionais requerem o output head correto.** Se `sensitivity` ou `compression_delta` retornam valores suspeitos, verifique se o `output_head` aceita o shape do embedding capturado.

**Em modelos LSTM/GRU**, a saída do hook é uma tupla `(output, (h_n, c_n))`. O `_flatten` pega o primeiro elemento. Para analisar o hidden state, passe um `output_head` que extrai `h_n`.

**Em circuitos quânticos**, os embeddings pós-medição são inerentemente limitados a `[-1, 1]` (valores esperados de Pauli). Ajuste `epsilon` para a escala real do embedding.

---

## 12. Referências

- **Roy, O. & Vetterli, M. (2007).** *The effective rank: A measure of effective dimensionality.* EUSIPCO 2007. — Fórmula do rank efetivo.

- **Kumar, A. et al. (2021).** *Implicit Under-Parameterization Inhibits Data-Efficient Deep Reinforcement Learning.* ICLR 2021. — Colapso de representação em RL.

- **Lyle, C. et al. (2022).** *Understanding and Preventing Capacity Loss in Reinforcement Learning.* ICLR 2022. — Análise espectral de redes de valor.

- **Daneshmand, H. et al. (2021).** *Batch Normalization Provably Avoids Ranks Collapse for Randomly Initialised Deep Networks.* NeurIPS 2021. — Rank e normalização.

- **Schulman, J. et al. (2017).** *Proximal Policy Optimization Algorithms.* arXiv:1707.06347. — PPO, contexto de redes ator-crítico.

- **Benedetti, M. et al. (2019).** *Parameterized quantum circuits as machine learning models.* Quantum Science and Technology. — PQC como modelos de ML.

- **Golub, G. & Van Loan, C. (2013).** *Matrix Computations, 4th ed.* Johns Hopkins. — SVD e álgebra linear numérica.
