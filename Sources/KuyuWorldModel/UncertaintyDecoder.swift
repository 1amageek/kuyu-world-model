import MLX
import MLXNN

/// Estimates per-dimension uncertainty from combined deterministic + stochastic state.
///
/// Input: [batch, hiddenDim + stochasticLatentSize]
/// Output: [batch, residualDim + extensionDim]
/// Sigmoid output: 0 = no confidence, 1 = full confidence.
public final class UncertaintyDecoder: Module {

    @ModuleInfo public var layer1: Linear
    @ModuleInfo public var layer2: Linear
    @ModuleInfo public var outputLayer: Linear

    private let outputDimensions: Int

    public init(config: WorldModelConfig) {
        self.outputDimensions = config.residualDimensions + config.extensionDimensions
        let inputDim = config.decoderInputDimensions
        let hiddenDim = config.hiddenDimensions
        self._layer1.wrappedValue = Linear(inputDim, hiddenDim)
        self._layer2.wrappedValue = Linear(hiddenDim, hiddenDim / 2)
        self._outputLayer.wrappedValue = Linear(hiddenDim / 2, outputDimensions)
    }

    /// Decode per-dimension uncertainty estimates.
    /// - Parameter state: combined state [batch, decoderInputDim]
    /// - Returns: confidence scores [batch, residualDim + extensionDim] in [0, 1]
    public func callAsFunction(_ state: MLXArray) -> MLXArray {
        var h = relu(layer1(state))
        h = relu(layer2(h))
        return sigmoid(outputLayer(h))
    }
}
