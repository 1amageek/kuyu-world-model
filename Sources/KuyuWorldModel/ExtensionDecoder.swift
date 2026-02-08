import MLX
import MLXNN

/// Decodes latent extension dimensions from combined deterministic + stochastic state.
///
/// Input: [batch, hiddenDim + stochasticLatentSize] -> Output: [batch, extensionDim]
/// Extensions represent dimensions not present in the physics model
/// (environment context, adaptation vectors, unmodeled dynamics).
/// 2-layer MLP with no bounded output (extensions are free-form latent dimensions).
public final class ExtensionDecoder: Module {

    @ModuleInfo public var layer1: Linear
    @ModuleInfo public var layer2: Linear
    @ModuleInfo public var outputLayer: Linear

    public init(config: WorldModelConfig) {
        let inputDim = config.decoderInputDimensions
        let hiddenDim = config.hiddenDimensions
        self._layer1.wrappedValue = Linear(inputDim, hiddenDim)
        self._layer2.wrappedValue = Linear(hiddenDim, hiddenDim / 2)
        self._outputLayer.wrappedValue = Linear(hiddenDim / 2, config.extensionDimensions)
    }

    /// Decode extension dimensions.
    /// - Parameter state: combined state [batch, decoderInputDim]
    /// - Returns: extension features [batch, extensionDim]
    public func callAsFunction(_ state: MLXArray) -> MLXArray {
        var h = relu(layer1(state))
        h = relu(layer2(h))
        return outputLayer(h)
    }
}
