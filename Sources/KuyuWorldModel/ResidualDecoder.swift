import MLX
import MLXNN

/// Decodes additive residual corrections from combined deterministic + stochastic state.
///
/// Input: [batch, hiddenDim + stochasticLatentSize] -> Output: [batch, physicsDim]
/// The output is an additive correction to the physics prediction.
/// 2-layer MLP with tanh output to bound the correction magnitude.
public final class ResidualDecoder: Module {

    @ModuleInfo public var layer1: Linear
    @ModuleInfo public var layer2: Linear
    @ModuleInfo public var outputLayer: Linear

    public init(config: WorldModelConfig) {
        let inputDim = config.decoderInputDimensions
        let hiddenDim = config.hiddenDimensions
        self._layer1.wrappedValue = Linear(inputDim, hiddenDim)
        self._layer2.wrappedValue = Linear(hiddenDim, hiddenDim / 2)
        self._outputLayer.wrappedValue = Linear(hiddenDim / 2, config.residualDimensions)
    }

    /// Decode residual corrections.
    /// - Parameter state: combined state [batch, decoderInputDim]
    /// - Returns: residual correction [batch, residualDim], bounded by tanh
    public func callAsFunction(_ state: MLXArray) -> MLXArray {
        var h = relu(layer1(state))
        h = relu(layer2(h))
        // tanh bounds the output to [-1, 1], preventing large corrections
        return tanh(outputLayer(h))
    }
}
