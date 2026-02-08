import Foundation

/// Inference-time state for the world model.
/// Stored as plain Float arrays for Sendable conformance and easy serialization.
public struct WorldModelState: Sendable, Equatable {
    /// Deterministic hidden state from the GRU transition model.
    public var h: [Float]

    /// Stochastic latent variable z (optional, only present when RSSM is enabled).
    public var z: [Float]?

    /// Last known physics state from the most recent infer() call.
    /// Used by predictFuture() as the initial physics context instead of zeros.
    public var lastPhysics: [Float]?

    public init(h: [Float], z: [Float]? = nil, lastPhysics: [Float]? = nil) {
        self.h = h
        self.z = z
        self.lastPhysics = lastPhysics
    }

    /// Create a zero-initialized state for the given config.
    public static func initial(config: WorldModelConfig) -> WorldModelState {
        WorldModelState(
            h: Array(repeating: 0, count: config.hiddenDimensions),
            z: config.rssmEnabled
                ? Array(repeating: 0, count: config.stochasticLatentSize)
                : nil,
            lastPhysics: nil
        )
    }
}
