import Foundation
import MLX
import MLXNN
import MLXOptimizers

/// Minimal trainer for StateWorldModel residual and prior learning.
///
/// This trains the learned residual head against physics-vs-actual residuals
/// and regularizes the prior toward the observation-conditioned posterior.
/// Physics remains the source of truth for validation; the trained model is
/// consumed through LearnedWorldModelEnvironmentAdapter gates.
public enum StateWorldModelTrainer {
    public struct Config: Sendable, Codable, Equatable {
        public let residualWeight: Float
        public let uncertaintyWeight: Float
        public let klWeight: Float
        public let uncertaintyVarianceFloor: Float

        public init(
            residualWeight: Float = 1.0,
            uncertaintyWeight: Float = 0.001,
            klWeight: Float = 0.001,
            uncertaintyVarianceFloor: Float = 1e-4
        ) {
            self.residualWeight = residualWeight
            self.uncertaintyWeight = uncertaintyWeight
            self.klWeight = klWeight
            self.uncertaintyVarianceFloor = max(uncertaintyVarianceFloor, Float.leastNonzeroMagnitude)
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
            let residualUncertainty = output.uncertainty[.ellipsis, 0..<model.config.residualDimensions]
            let uncertaintyLoss = residualUncertaintyNLL(
                predictions: output.residual,
                targets: targets,
                uncertainty: residualUncertainty,
                varianceFloor: config.uncertaintyVarianceFloor
            )
            let klLoss = categoricalKLLoss(
                priorLogits: output.priorLogits,
                posteriorLogits: output.posteriorLogits,
                modelConfig: model.config
            )
            return config.residualWeight * residualLoss
                + config.uncertaintyWeight * uncertaintyLoss
                + config.klWeight * klLoss
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

    static func residualUncertaintyNLL(
        predictions: MLXArray,
        targets: MLXArray,
        uncertainty: MLXArray,
        varianceFloor: Float
    ) -> MLXArray {
        let variance = square(uncertainty) + max(varianceFloor, Float.leastNonzeroMagnitude)
        let error = predictions - targets
        return (0.5 * (square(error) / variance + log(variance))).mean()
    }

    private static func categoricalKLLoss(
        priorLogits: MLXArray,
        posteriorLogits: MLXArray,
        modelConfig: WorldModelConfig
    ) -> MLXArray {
        guard modelConfig.stochasticCategories > 0, modelConfig.stochasticClasses > 0 else {
            return MLXArray(Float(0))
        }

        let leadingDimensions = Array(priorLogits.shape.dropLast())
        let categoricalShape = leadingDimensions + [
            modelConfig.stochasticCategories,
            modelConfig.stochasticClasses,
        ]
        let priorLogProbs = logSoftmax(priorLogits.reshaped(categoricalShape), axis: -1)
        let posteriorLogProbs = logSoftmax(posteriorLogits.reshaped(categoricalShape), axis: -1)
        return klDivLoss(
            inputs: priorLogProbs,
            targets: posteriorLogProbs,
            axis: -1,
            reduction: .mean
        )
    }
}
