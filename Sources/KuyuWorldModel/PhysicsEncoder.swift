import MLX
import MLXNN

/// Encodes physics state into a dense embedding.
/// Input: [batch, physicsDim] -> Output: [batch, physicsEmbedDim]
/// 2-layer MLP with ReLU activation.
public final class PhysicsEncoder: Module {

    @ModuleInfo public var layer1: Linear
    @ModuleInfo public var layer2: Linear

    public init(config: WorldModelConfig) {
        self._layer1.wrappedValue = Linear(config.physicsDimensions, config.physicsEmbedDimensions)
        self._layer2.wrappedValue = Linear(config.physicsEmbedDimensions, config.physicsEmbedDimensions)
    }

    public func callAsFunction(_ physicsState: MLXArray) -> MLXArray {
        var h = relu(layer1(physicsState))
        h = relu(layer2(h))
        return h
    }
}
