import Foundation
import MLX
import MLXNN
import MLXOptimizers

/// Minimal trainer for StateWorldModel residual learning.
///
/// This trains the learned residual head against physics-vs-actual residuals.
/// Physics remains the source of truth for validation; the trained model is
/// consumed through LearnedWorldModelEnvironmentAdapter gates.
public enum StateWorldModelTrainer {
    public struct Config: Sendable, Codable, Equatable {
        public let residualWeight: Float
        public let uncertaintyWeight: Float

        public init(
            residualWeight: Float = 1.0,
            uncertaintyWeight: Float = 0.001
        ) {
            self.residualWeight = residualWeight
            self.uncertaintyWeight = uncertaintyWeight
        }
    }

    public static func train(
        model: StateWorldModel,
        batches: [StateWorldModelTrainingBatch],
        config: Config = Config(),
        learningRate: Float = 0.001,
        maxGradNorm: Float? = 1.0,
        epochs: Int
    ) -> [Float] {
        let optimizer = Adam(learningRate: learningRate)
        let lossAndGrad = valueAndGrad(model: model) { model, inputs, targets in
            let physicsEnd = model.config.physicsDimensions
            let sensorEnd = physicsEnd + model.config.sensorDimensions
            let actionEnd = sensorEnd + model.config.actionDimensions
            let physicsStates = inputs[.ellipsis, 0..<physicsEnd]
            let sensorObservations = inputs[.ellipsis, physicsEnd..<sensorEnd]
            let actions = inputs[.ellipsis, sensorEnd..<actionEnd]
            let output = model.forward(
                physicsStates: physicsStates,
                sensorObs: sensorObservations,
                actions: actions
            )
            let residualLoss = mseLoss(
                predictions: output.residual,
                targets: targets,
                reduction: .mean
            )
            let uncertaintyPenalty = output.uncertainty.mean()
            return config.residualWeight * residualLoss + config.uncertaintyWeight * uncertaintyPenalty
        }
        var epochLosses: [Float] = []
        model.train(true)

        for _ in 0..<epochs {
            var totalLoss: Float = 0
            var totalStepCount = 0
            for batch in batches {
                let (loss, stepCount): (Float, Int) = autoreleasepool {
                    let packedInputs = concatenated(
                        [batch.physicsStates, batch.sensorObservations, batch.actions],
                        axis: -1
                    )
                    let (lossValue, gradients) = lossAndGrad(model, packedInputs, batch.residualTargets)
                    let clipped = clipIfNeeded(gradients, maxNorm: maxGradNorm)
                    optimizer.update(model: model, gradients: clipped)
                    eval(model, optimizer)
                    return (lossValue.item(Float.self), batchStepCount(batch.residualTargets))
                }
                totalLoss += loss * Float(stepCount)
                totalStepCount += stepCount
            }
            epochLosses.append(totalLoss / Float(max(totalStepCount, 1)))
        }

        model.train(false)
        return epochLosses
    }

    private static func clipIfNeeded(_ gradients: ModuleParameters, maxNorm: Float?) -> ModuleParameters {
        guard let maxNorm else { return gradients }
        return clipGradNorm(gradients: gradients, maxNorm: maxNorm).0
    }

    private static func batchStepCount(_ targets: MLXArray) -> Int {
        let shape = targets.shape
        guard shape.count >= 3 else {
            return shape.first ?? 1
        }
        return max(1, shape[0] * shape[1])
    }
}
