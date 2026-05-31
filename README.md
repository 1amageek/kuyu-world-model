# kuyu-world-model

Physics-residual learned world model for the Kuyu fused environment.

## Overview

kuyu-world-model implements a physics-informed world model that learns to correct and extend analytical physics predictions. Built on MLX Swift for Apple Silicon acceleration. Analytical physics remains the reference trajectory; learned outputs are residual and extension signals that must pass validation gates before use.

### Architecture

The world model receives physics predictions as the reference input and learns two outputs:

- **Residual** — Additive corrections to physics predictions (same dimension space). Captures phenomena outside the ODE: ground effect, motor degradation, wind turbulence, etc.
- **Extension** — New latent dimensions not present in physics: environment context, adaptation vectors, and adaptation features.
- **Uncertainty** — Per-dimension risk score where `0` is low uncertainty and `1` is high uncertainty.

### Components

- **`StateWorldModel`** — Main MLX Module combining all sub-components.
- **`PhysicsEncoder`** — Embeds physics predictions into a latent representation (physics-informed prior).
- **`StateTokenizer`** — 1D causal convolution encoder/decoder for time-series compression.
- **`TransitionModel`** — Physics-informed GRU with categorical stochastic latent (DreamerV3-style RSSM).
- **`ResidualDecoder`** / **`ExtensionDecoder`** / **`UncertaintyDecoder`** — Decode hidden state into outputs.
- **`WorldPredictor`** — Prior-only multi-step rollout. Callers can inject a physics advance callback so imagination uses the analytical integrator plus learned residuals without making this package depend on kuyu-physics.
- **`DomainAdapter`** — Sim-to-real transfer mapping.
- **`MLXWorldModelController`** — Bridges `StateWorldModel` to `WorldModelProtocol` and `PhysicsAwareWorldModelProtocol` (thread-safe via Mutex).

### Key Design Properties

- **Physics is never modified**: The world model only provides corrections on top. Environment adapters build corrected outputs from a reference physics result; they do not mutate the physics simulator state.
- **Untrained = identity**: Residual and extension output layers start at zero, and the domain adapter starts as an exact identity projection. An untrained neural model therefore emits zero correction and zero extension.
- **Prior learns from posterior**: Training includes a categorical KL term that moves prior logits toward stop-gradient observation-conditioned posterior logits, so prior-only rollout has a supervised path without degrading the posterior correction path.
- **Physics-aware imagination is composed outside this package**: kuyu-world-model does not depend on kuyu-physics. Fused callers that own both systems precompute analytical physics predictions and pass them through `PhysicsAwareWorldModelProtocol`; lower-level callers can still pass a physics advance callback into `WorldPredictor`.
- **Uncertainty polarity is explicit**: `0` means low uncertainty and `1` means high uncertainty. Residual uncertainty is trained with a Gaussian negative log-likelihood term against residual error. Environment acceptance gates use residual uncertainty only; extension uncertainty remains latent metadata.

## Package Structure

| Module | Dependencies | Description |
|--------|-------------|-------------|
| **KuyuWorldModel** | KuyuCore, mlx-swift | World model implementation |

## Requirements

- Swift 6.2+
- macOS 26+
- Apple Silicon (MLX requires Metal)

## Dependency Graph

```
KuyuCore
  |
  +-- KuyuWorldModel (this package) + mlx-swift
  |     |
  |     +-- kuyu/KuyuMLX (assembles FusedEnvironment)
  |
  +-- KuyuPhysics (independent, no mutual dependency)
```

kuyu-world-model and kuyu-physics are independent of each other. They are composed together in kuyu via `FusedEnvironment<QuadrotorAnalyticalModel, MLXWorldModelController, SensorField>`.

## License

See repository for license information.
