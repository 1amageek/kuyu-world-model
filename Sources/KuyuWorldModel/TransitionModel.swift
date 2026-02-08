import MLX
import MLXNN

/// Physics-informed GRU transition model.
///
/// Combines action and physics embedding as input to a GRU cell,
/// producing deterministic hidden state and stochastic latent logits.
///
/// For training: uses posterior (conditioned on observations).
/// For imagination: uses prior only (no observations).
public final class TransitionModel: Module {

    /// Projects concatenated action + physics embedding into GRU input space.
    @ModuleInfo public var inputProjection: Linear

    /// GRU cell for single-step transitions.
    @ModuleInfo public var gru: GRUCell

    /// Prior head: h -> categorical logits for p(z_t | h_t).
    @ModuleInfo public var priorHead: Linear

    /// Posterior head: concat(h, obsEmbed) -> categorical logits for q(z_t | h_t, x_t).
    @ModuleInfo public var posteriorHead: Linear

    private let hiddenDimensions: Int
    private let stochasticLatentSize: Int

    public init(config: WorldModelConfig) {
        self.hiddenDimensions = config.hiddenDimensions
        self.stochasticLatentSize = config.stochasticLatentSize

        // Input: action (actionDim) + physics embedding (physicsEmbedDim)
        let inputSize = config.actionDimensions + config.physicsEmbedDimensions
        self._inputProjection.wrappedValue = Linear(inputSize, config.hiddenDimensions)

        self._gru.wrappedValue = GRUCell(
            inputSize: config.hiddenDimensions,
            hiddenSize: config.hiddenDimensions
        )

        // Prior: h -> stochastic logits
        self._priorHead.wrappedValue = Linear(
            config.hiddenDimensions, config.stochasticLatentSize
        )

        // Posterior: h + obsEmbed -> stochastic logits
        self._posteriorHead.wrappedValue = Linear(
            config.hiddenDimensions + config.physicsEmbedDimensions,
            config.stochasticLatentSize
        )
    }

    /// Prior-only forward (imagination / inference without observations).
    /// - Parameters:
    ///   - h: previous hidden state [batch, hiddenDim]
    ///   - action: action taken [batch, actionDim]
    ///   - physicsEmbed: physics state embedding [batch, physicsEmbedDim]
    /// - Returns: (newH [batch, hiddenDim], priorLogits [batch, stochasticLatentSize])
    public func forward(
        h: MLXArray,
        action: MLXArray,
        physicsEmbed: MLXArray
    ) -> (MLXArray, MLXArray) {
        let input = concatenated([action, physicsEmbed], axis: -1)
        let projected = relu(inputProjection(input))
        let newH = gru(x: projected, h: h)
        let priorLogits = priorHead(newH)
        return (newH, priorLogits)
    }

    /// Posterior forward (training with observations).
    /// - Parameters:
    ///   - h: previous hidden state [batch, hiddenDim]
    ///   - action: action taken [batch, actionDim]
    ///   - physicsEmbed: physics state embedding [batch, physicsEmbedDim]
    ///   - obsEmbed: observation embedding [batch, physicsEmbedDim]
    /// - Returns: (newH, priorLogits, posteriorLogits) all with batch dimension
    public func forward(
        h: MLXArray,
        action: MLXArray,
        physicsEmbed: MLXArray,
        obsEmbed: MLXArray
    ) -> (MLXArray, MLXArray, MLXArray) {
        let input = concatenated([action, physicsEmbed], axis: -1)
        let projected = relu(inputProjection(input))
        let newH = gru(x: projected, h: h)
        let priorLogits = priorHead(newH)
        let posteriorInput = concatenated([newH, obsEmbed], axis: -1)
        let posteriorLogits = posteriorHead(posteriorInput)
        return (newH, priorLogits, posteriorLogits)
    }
}
