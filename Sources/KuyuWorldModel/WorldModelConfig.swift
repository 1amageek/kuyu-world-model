import Foundation

/// Configuration for the StateWorldModel.
/// All hyperparameters controlling architecture dimensions and behavior.
public struct WorldModelConfig: Codable, Sendable, Equatable {

    /// Number of physics state dimensions (e.g. 13 for quadrotor: pos3 + quat4 + vel3 + omega3).
    public var physicsDimensions: Int

    /// Number of sensor observation dimensions (e.g. 6 for IMU: accel3 + gyro3).
    public var sensorDimensions: Int

    /// Number of action dimensions (e.g. 4 for quadrotor motor commands).
    public var actionDimensions: Int

    /// Hidden dimension for the GRU transition model.
    public var hiddenDimensions: Int

    /// Dimension of the stochastic latent variable z (= stochasticCategories * stochasticClasses).
    public var latentDimensions: Int

    /// Number of categorical distributions in the stochastic latent.
    public var stochasticCategories: Int

    /// Number of classes per categorical distribution.
    public var stochasticClasses: Int

    /// Dimension of the residual output (same as physics dimensions).
    public var residualDimensions: Int

    /// Dimension of the extension output (additional latent dimensions).
    public var extensionDimensions: Int

    /// Number of Conv1d layers in the state tokenizer encoder.
    public var tokenizerLayers: Int

    /// Kernel size for tokenizer Conv1d layers.
    public var tokenizerKernelSize: Int

    /// Embedding dimension for the physics encoder output.
    public var physicsEmbedDimensions: Int

    /// Unimix ratio for categorical sampling (0 = no mixing, >0 = mix with uniform).
    public var stochasticUnimixRatio: Float

    /// Computed stochastic latent size.
    public var stochasticLatentSize: Int {
        stochasticCategories * stochasticClasses
    }

    /// Whether the RSSM stochastic latent is enabled.
    public var rssmEnabled: Bool {
        stochasticLatentSize > 0
    }

    /// Size of combined deterministic + stochastic state for decoder input.
    public var decoderInputDimensions: Int {
        hiddenDimensions + stochasticLatentSize
    }

    public init(
        physicsDimensions: Int = 13,
        sensorDimensions: Int = 6,
        actionDimensions: Int = 4,
        hiddenDimensions: Int = 128,
        latentDimensions: Int = 32,
        stochasticCategories: Int = 8,
        stochasticClasses: Int = 8,
        residualDimensions: Int = 13,
        extensionDimensions: Int = 16,
        tokenizerLayers: Int = 3,
        tokenizerKernelSize: Int = 3,
        physicsEmbedDimensions: Int = 64,
        stochasticUnimixRatio: Float = 0.01
    ) {
        self.physicsDimensions = physicsDimensions
        self.sensorDimensions = sensorDimensions
        self.actionDimensions = actionDimensions
        self.hiddenDimensions = hiddenDimensions
        self.latentDimensions = latentDimensions
        self.stochasticCategories = stochasticCategories
        self.stochasticClasses = stochasticClasses
        self.residualDimensions = residualDimensions
        self.extensionDimensions = extensionDimensions
        self.tokenizerLayers = tokenizerLayers
        self.tokenizerKernelSize = tokenizerKernelSize
        self.physicsEmbedDimensions = physicsEmbedDimensions
        self.stochasticUnimixRatio = stochasticUnimixRatio
    }

    private enum CodingKeys: String, CodingKey {
        case physicsDimensions, sensorDimensions, actionDimensions
        case hiddenDimensions, latentDimensions
        case stochasticCategories, stochasticClasses
        case residualDimensions, extensionDimensions
        case tokenizerLayers, tokenizerKernelSize
        case physicsEmbedDimensions, stochasticUnimixRatio
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        physicsDimensions = try container.decodeIfPresent(Int.self, forKey: .physicsDimensions) ?? 13
        sensorDimensions = try container.decodeIfPresent(Int.self, forKey: .sensorDimensions) ?? 6
        actionDimensions = try container.decodeIfPresent(Int.self, forKey: .actionDimensions) ?? 4
        hiddenDimensions = try container.decodeIfPresent(Int.self, forKey: .hiddenDimensions) ?? 128
        latentDimensions = try container.decodeIfPresent(Int.self, forKey: .latentDimensions) ?? 32
        stochasticCategories = try container.decodeIfPresent(Int.self, forKey: .stochasticCategories) ?? 8
        stochasticClasses = try container.decodeIfPresent(Int.self, forKey: .stochasticClasses) ?? 8
        residualDimensions = try container.decodeIfPresent(Int.self, forKey: .residualDimensions) ?? 13
        extensionDimensions = try container.decodeIfPresent(Int.self, forKey: .extensionDimensions) ?? 16
        tokenizerLayers = try container.decodeIfPresent(Int.self, forKey: .tokenizerLayers) ?? 3
        tokenizerKernelSize = try container.decodeIfPresent(Int.self, forKey: .tokenizerKernelSize) ?? 3
        physicsEmbedDimensions = try container.decodeIfPresent(Int.self, forKey: .physicsEmbedDimensions) ?? 64
        stochasticUnimixRatio = try container.decodeIfPresent(Float.self, forKey: .stochasticUnimixRatio) ?? 0.01
    }
}
