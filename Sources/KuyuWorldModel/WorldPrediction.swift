import MLX

/// A single prediction step from the world predictor rollout.
public struct WorldPrediction {
    /// Residual correction to physics state: [batch, residualDim]
    public let residual: MLXArray

    /// Extension dimensions: [batch, extensionDim]
    public let extensions: MLXArray

    /// Per-dimension uncertainty: [batch, residualDim + extensionDim]
    public let uncertainty: MLXArray

    /// Hidden state after this step: [batch, hiddenDim]
    public let h: MLXArray

    public init(
        residual: MLXArray,
        extensions: MLXArray,
        uncertainty: MLXArray,
        h: MLXArray
    ) {
        self.residual = residual
        self.extensions = extensions
        self.uncertainty = uncertainty
        self.h = h
    }
}
