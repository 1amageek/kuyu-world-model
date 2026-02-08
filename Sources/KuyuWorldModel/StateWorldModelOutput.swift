import MLX

/// Output from the StateWorldModel training forward pass.
/// Contains all MLXArray outputs needed for loss computation.
public struct StateWorldModelOutput {
    /// Residual corrections to physics: [batch, seq, residualDim]
    public let residual: MLXArray

    /// Extension dimensions: [batch, seq, extensionDim]
    public let extensions: MLXArray

    /// Per-dimension uncertainty estimates: [batch, seq, residualDim + extensionDim]
    public let uncertainty: MLXArray

    /// Prior logits from transition model: [batch, seq, stochasticLatentSize]
    public let priorLogits: MLXArray

    /// Posterior logits from transition model: [batch, seq, stochasticLatentSize]
    public let posteriorLogits: MLXArray

    /// Final hidden state after processing the sequence: [batch, hiddenDim]
    public let finalH: MLXArray

    public init(
        residual: MLXArray,
        extensions: MLXArray,
        uncertainty: MLXArray,
        priorLogits: MLXArray,
        posteriorLogits: MLXArray,
        finalH: MLXArray
    ) {
        self.residual = residual
        self.extensions = extensions
        self.uncertainty = uncertainty
        self.priorLogits = priorLogits
        self.posteriorLogits = posteriorLogits
        self.finalH = finalH
    }
}
