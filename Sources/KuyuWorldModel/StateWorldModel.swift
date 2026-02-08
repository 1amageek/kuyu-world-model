import MLX
import MLXRandom
import MLXNN

/// Full world model combining physics encoder, state tokenizer, transition model,
/// and decoder heads for residual, extension, and uncertainty.
///
/// Implements the forward pass for both training (sequence) and inference (single step).
/// Physics predictions are never modified; corrections are additive only.
public final class StateWorldModel: Module {

    public let config: WorldModelConfig

    @ModuleInfo public var physicsEncoder: PhysicsEncoder
    @ModuleInfo public var tokenizer: StateTokenizer
    @ModuleInfo public var transition: TransitionModel
    @ModuleInfo public var residualDecoder: ResidualDecoder
    @ModuleInfo public var extensionDecoder: ExtensionDecoder
    @ModuleInfo public var uncertaintyDecoder: UncertaintyDecoder

    public init(config: WorldModelConfig) {
        self.config = config
        self._physicsEncoder.wrappedValue = PhysicsEncoder(config: config)
        self._tokenizer.wrappedValue = StateTokenizer(config: config)
        self._transition.wrappedValue = TransitionModel(config: config)
        self._residualDecoder.wrappedValue = ResidualDecoder(config: config)
        self._extensionDecoder.wrappedValue = ExtensionDecoder(config: config)
        self._uncertaintyDecoder.wrappedValue = UncertaintyDecoder(config: config)
    }

    /// Training forward pass over a sequence.
    ///
    /// Processes each time step sequentially through the transition model
    /// to compute posterior/prior logits for KL loss.
    ///
    /// - Parameters:
    ///   - physicsStates: [batch, seq, physicsDim]
    ///   - sensorObs: [batch, seq, sensorDim]
    ///   - actions: [batch, seq, actionDim]
    ///   - initialH: optional initial hidden state [batch, hiddenDim]
    /// - Returns: StateWorldModelOutput with all decoder outputs and logits
    public func forward(
        physicsStates: MLXArray,
        sensorObs: MLXArray,
        actions: MLXArray,
        initialH: MLXArray? = nil
    ) -> StateWorldModelOutput {
        let batch = physicsStates.dim(0)
        let seq = physicsStates.dim(1)

        // Encode sensor observations: [batch, seq, sensorDim] -> [batch, seq, physicsEmbedDim]
        let obsEmbeddings = tokenizer.encode(sensorObs)

        // Process each time step through transition model
        var h = initialH ?? MLXArray.zeros([batch, config.hiddenDimensions])

        var allPriorLogits: [MLXArray] = []
        var allPosteriorLogits: [MLXArray] = []
        var allDecoderStates: [MLXArray] = []

        for t in 0..<seq {
            // Extract time step t
            let physT = physicsStates[.ellipsis, t, 0...]   // [batch, physicsDim]
            let actT = actions[.ellipsis, t, 0...]           // [batch, actionDim]
            let obsEmbT = obsEmbeddings[.ellipsis, t, 0...]  // [batch, physicsEmbedDim]

            // Encode physics state
            let physEmb = physicsEncoder(physT)  // [batch, physicsEmbedDim]

            // Transition with posterior
            let (newH, priorLogits, posteriorLogits) = transition.forward(
                h: h, action: actT, physicsEmbed: physEmb, obsEmbed: obsEmbT
            )
            h = newH

            // Sample stochastic latent from posterior (training)
            let z = sampleCategorical(logits: posteriorLogits)

            // Combine deterministic + stochastic for decoder
            let decoderState = concatenated([h, z], axis: -1)

            allPriorLogits.append(priorLogits)
            allPosteriorLogits.append(posteriorLogits)
            allDecoderStates.append(decoderState)
        }

        // Stack time steps: [batch, seq, dim]
        let priorLogitsSeq = stacked(allPriorLogits, axis: 1)
        let posteriorLogitsSeq = stacked(allPosteriorLogits, axis: 1)
        let decoderStatesSeq = stacked(allDecoderStates, axis: 1)

        // Decode all time steps at once (Linear operates on last axis)
        let residual = decodeResidualSequence(decoderStatesSeq)
        let extensions = decodeExtensionSequence(decoderStatesSeq)
        let uncertainty = decodeUncertaintySequence(decoderStatesSeq)

        return StateWorldModelOutput(
            residual: residual,
            extensions: extensions,
            uncertainty: uncertainty,
            priorLogits: priorLogitsSeq,
            posteriorLogits: posteriorLogitsSeq,
            finalH: h
        )
    }

    /// Single-step inference forward.
    ///
    /// - Parameters:
    ///   - physicsState: current physics state [batch, physicsDim]
    ///   - sensorObs: current sensor observations [batch, sensorDim]
    ///   - action: current action [batch, actionDim]
    ///   - h: current hidden state [batch, hiddenDim]
    /// - Returns: tuple of (residual, extension, uncertainty, newH)
    public func step(
        physicsState: MLXArray,
        sensorObs: MLXArray,
        action: MLXArray,
        h: MLXArray
    ) -> (residual: MLXArray, extensions: MLXArray, uncertainty: MLXArray, h: MLXArray) {
        // Encode physics
        let physEmb = physicsEncoder(physicsState)

        // Encode sensor observation (single step -> wrap in sequence)
        let sensorSeq = expandedDimensions(sensorObs, axis: 1)  // [batch, 1, sensorDim]
        let obsEmb = tokenizer.encode(sensorSeq)
        let obsEmbFlat = obsEmb.squeezed(axis: 1)  // [batch, physicsEmbedDim]

        // Transition with posterior (use observations during inference for best accuracy)
        let (newH, _, posteriorLogits) = transition.forward(
            h: h, action: action, physicsEmbed: physEmb, obsEmbed: obsEmbFlat
        )

        // Sample from posterior
        let z = sampleCategorical(logits: posteriorLogits)

        // Decode
        let decoderState = concatenated([newH, z], axis: -1)
        let residual = residualDecoder(decoderState)
        let extensions = extensionDecoder(decoderState)
        let uncertainty = uncertaintyDecoder(decoderState)

        return (residual: residual, extensions: extensions, uncertainty: uncertainty, h: newH)
    }

    /// Prior-only step for imagination (no observations).
    ///
    /// - Parameters:
    ///   - physicsState: current physics state [batch, physicsDim]
    ///   - action: action to take [batch, actionDim]
    ///   - h: current hidden state [batch, hiddenDim]
    /// - Returns: tuple of (residual, extension, uncertainty, newH)
    public func imaginationStep(
        physicsState: MLXArray,
        action: MLXArray,
        h: MLXArray
    ) -> (residual: MLXArray, extensions: MLXArray, uncertainty: MLXArray, h: MLXArray) {
        let physEmb = physicsEncoder(physicsState)
        let (newH, priorLogits) = transition.forward(
            h: h, action: action, physicsEmbed: physEmb
        )

        let z = sampleCategorical(logits: priorLogits)

        let decoderState = concatenated([newH, z], axis: -1)
        let residual = residualDecoder(decoderState)
        let extensions = extensionDecoder(decoderState)
        let uncertainty = uncertaintyDecoder(decoderState)

        return (residual: residual, extensions: extensions, uncertainty: uncertainty, h: newH)
    }

    /// Default callAsFunction for Module conformance.
    /// Runs single-step inference.
    public func callAsFunction(
        _ physicsState: MLXArray,
        sensorObs: MLXArray,
        action: MLXArray,
        h: MLXArray
    ) -> MLXArray {
        let result = step(
            physicsState: physicsState,
            sensorObs: sensorObs,
            action: action,
            h: h
        )
        return result.residual
    }

    // MARK: - Internal

    /// DreamerV3-style categorical sampling with unimix and straight-through gradient.
    func sampleCategorical(logits: MLXArray) -> MLXArray {
        let cats = config.stochasticCategories
        let cls = config.stochasticClasses
        let shape = logits.shape
        let leadingDims = Array(shape.dropLast())
        let reshaped = logits.reshaped(leadingDims + [cats, cls])

        var probs = softmax(reshaped, axis: -1)

        let unimix = config.stochasticUnimixRatio
        if unimix > 0 {
            let uniform = MLXArray.ones(like: probs) * (1.0 / Float(cls))
            probs = (1.0 - unimix) * probs + unimix * uniform
        }

        if self.training {
            // Gumbel-softmax with straight-through gradient
            let u = MLXRandom.uniform(0 ..< 1, probs.shape)
            let gumbel = -log(-log(u + 1e-8) + 1e-8)
            let noisyLogits = log(probs + 1e-8) + gumbel
            let sharpProbs = softmax(noisyLogits * 10.0, axis: -1)
            let stProbs = probs + stopGradient(sharpProbs - probs)
            return stProbs.reshaped(leadingDims + [cats * cls])
        } else {
            let hardProbs = softmax(reshaped * 100.0, axis: -1)
            return hardProbs.reshaped(leadingDims + [cats * cls])
        }
    }

    /// Decode residual for a sequence of decoder states.
    /// Input: [batch, seq, decoderInputDim] -> Output: [batch, seq, residualDim]
    private func decodeResidualSequence(_ states: MLXArray) -> MLXArray {
        // Linear operates on last axis, so [batch, seq, dim] works directly
        let batch = states.dim(0)
        let seq = states.dim(1)
        let flat = states.reshaped([batch * seq, states.dim(2)])
        let decoded = residualDecoder(flat)
        return decoded.reshaped([batch, seq, config.residualDimensions])
    }

    /// Decode extensions for a sequence of decoder states.
    private func decodeExtensionSequence(_ states: MLXArray) -> MLXArray {
        let batch = states.dim(0)
        let seq = states.dim(1)
        let flat = states.reshaped([batch * seq, states.dim(2)])
        let decoded = extensionDecoder(flat)
        return decoded.reshaped([batch, seq, config.extensionDimensions])
    }

    /// Decode uncertainty for a sequence of decoder states.
    private func decodeUncertaintySequence(_ states: MLXArray) -> MLXArray {
        let batch = states.dim(0)
        let seq = states.dim(1)
        let totalDim = config.residualDimensions + config.extensionDimensions
        let flat = states.reshaped([batch * seq, states.dim(2)])
        let decoded = uncertaintyDecoder(flat)
        return decoded.reshaped([batch, seq, totalDim])
    }
}
