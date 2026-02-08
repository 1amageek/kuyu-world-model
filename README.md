# kuyu-world-model

DreamerV3-based learned world model for the Kuyu fused environment.

## Overview

kuyu-world-model implements a physics-informed world model that learns to correct and extend analytical physics predictions. Built on MLX Swift for Apple Silicon acceleration.

### Architecture

The world model receives physics predictions as a prior and learns two outputs:

- **Residual** — Additive corrections to physics predictions (same dimension space). Captures phenomena outside the ODE: ground effect, motor degradation, wind turbulence, etc.
- **Extension** — New latent dimensions not present in physics: environment context, adaptation vectors, uncertainty estimates.

### Components

- **`StateWorldModel`** — Main MLX Module combining all sub-components.
- **`PhysicsEncoder`** — Embeds physics predictions into a latent representation (physics-informed prior).
- **`StateTokenizer`** — 1D causal convolution encoder/decoder for time-series compression.
- **`TransitionModel`** — Physics-informed GRU with categorical stochastic latent (DreamerV3-style RSSM).
- **`ResidualDecoder`** / **`ExtensionDecoder`** / **`UncertaintyDecoder`** — Decode hidden state into outputs.
- **`WorldPredictor`** — Multi-step rollout for imagination-based RL.
- **`DomainAdapter`** — Sim-to-real transfer mapping.
- **`MLXWorldModelController`** — Bridges `StateWorldModel` to the `WorldModelProtocol` (thread-safe via Mutex).

### Key Design Properties

- **Physics is never modified**: The world model only provides corrections on top.
- **KL divergence as physics accuracy metric**: When physics is accurate, prior ~ posterior (KL ~ 0). When inaccurate, posterior diverges (KL > 0).
- **Untrained = identity**: Random initialization produces near-zero residuals, so `FusedState ~ physics_prediction`.

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
