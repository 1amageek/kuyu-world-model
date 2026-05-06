import MLX

/// Tensor batch for training StateWorldModel from analytical physics rollouts.
public struct StateWorldModelTrainingBatch {
    public let physicsStates: MLXArray
    public let sensorObservations: MLXArray
    public let actions: MLXArray
    public let residualTargets: MLXArray

    public init(
        physicsStates: MLXArray,
        sensorObservations: MLXArray,
        actions: MLXArray,
        residualTargets: MLXArray
    ) {
        self.physicsStates = physicsStates
        self.sensorObservations = sensorObservations
        self.actions = actions
        self.residualTargets = residualTargets
    }
}
