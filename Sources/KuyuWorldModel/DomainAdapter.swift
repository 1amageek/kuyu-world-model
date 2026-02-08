import MLX
import MLXNN

/// Sim-to-real domain adaptation module.
///
/// Applies a learned projection to residual and extension outputs
/// to bridge the sim-to-real gap. Initially acts as near-identity.
/// Will be trained with real data via LoRA or full fine-tuning.
public final class DomainAdapter: Module {

    @ModuleInfo public var residualProjection: Linear
    @ModuleInfo public var extensionProjection: Linear

    public init(config: WorldModelConfig) {
        self._residualProjection.wrappedValue = Linear(
            config.residualDimensions, config.residualDimensions
        )
        self._extensionProjection.wrappedValue = Linear(
            config.extensionDimensions, config.extensionDimensions
        )
    }

    /// Adapt sim-domain residual and extension to real domain.
    /// - Parameters:
    ///   - simResidual: residual from sim world model [batch, residualDim]
    ///   - simExtension: extension from sim world model [batch, extensionDim]
    /// - Returns: adapted (residual, extension)
    public func callAsFunction(
        simResidual: MLXArray,
        simExtension: MLXArray
    ) -> (MLXArray, MLXArray) {
        let adaptedResidual = residualProjection(simResidual)
        let adaptedExtension = extensionProjection(simExtension)
        return (adaptedResidual, adaptedExtension)
    }
}
