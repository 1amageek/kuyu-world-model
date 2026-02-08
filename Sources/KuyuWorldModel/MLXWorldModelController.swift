import Foundation
import Synchronization
import MLX
import MLXNN
import KuyuCore

/// Bridges the MLX-based StateWorldModel to the KuyuCore WorldModelProtocol.
///
/// Converts between Float arrays (protocol boundary) and MLXArray (internal computation).
/// Maintains inference state (hidden + stochastic latent) across calls.
///
/// Thread safety: StateWorldModel is a class (MLX Module subclass) that is not Sendable.
/// Access is serialized through the mutating protocol methods, and the model reference
/// is protected by the Storage class with Mutex.
public struct MLXWorldModelController: WorldModelProtocol {

    public enum ControllerError: Error, Equatable {
        case dimensionMismatch(expected: Int, got: Int)
        case actionCountMismatch(expected: Int, got: Int)
        case stepCountMismatch(expected: Int, got: Int)
    }

    /// Thread-safe storage for non-Sendable model and mutable state.
    private final class Storage: Sendable {
        private let mutex: Mutex<MutableState>

        struct MutableState {
            nonisolated(unsafe) var model: StateWorldModel
            var worldState: WorldModelState
        }

        init(model: StateWorldModel, worldState: WorldModelState) {
            self.mutex = Mutex(MutableState(model: model, worldState: worldState))
        }

        func withLock<R>(_ body: (inout MutableState) -> R) -> R {
            mutex.withLock { state in
                body(&state)
            }
        }

        func withLockThrowing<R>(_ body: (inout MutableState) throws -> R) throws -> R {
            try mutex.withLock { state in
                try body(&state)
            }
        }
    }

    private let storage: Storage
    private let config: WorldModelConfig

    public init(model: StateWorldModel, config: WorldModelConfig) {
        self.config = config
        self.storage = Storage(
            model: model,
            worldState: WorldModelState.initial(config: config)
        )
    }

    public mutating func infer(
        physicsPrediction: some AnalyticalState,
        sensorObservations: [ChannelSample],
        action: [ActuatorValue],
        dt: TimeInterval
    ) throws -> WorldModelOutput {
        // Convert physics prediction to MLXArray [1, physicsDim]
        let physicsArray = physicsPrediction.toArray()
        guard physicsArray.count == config.physicsDimensions else {
            throw ControllerError.dimensionMismatch(expected: config.physicsDimensions, got: physicsArray.count)
        }
        let physicsMLX = MLXArray(physicsArray).reshaped([1, config.physicsDimensions])

        // Convert sensor observations to MLXArray [1, sensorDim]
        let sensorValues = sensorObservationsToArray(sensorObservations)
        let sensorMLX = MLXArray(sensorValues).reshaped([1, config.sensorDimensions])

        // Convert actions to MLXArray [1, actionDim]
        let actionValues = action.map { Float($0.value) }
        guard actionValues.count == config.actionDimensions else {
            throw ControllerError.dimensionMismatch(expected: config.actionDimensions, got: actionValues.count)
        }
        let actionMLX = MLXArray(actionValues).reshaped([1, config.actionDimensions])

        return try storage.withLockThrowing { state in
            // Convert hidden state to MLXArray [1, hiddenDim]
            let hMLX = MLXArray(state.worldState.h).reshaped([1, config.hiddenDimensions])

            // Run single step inference
            let result = state.model.step(
                physicsState: physicsMLX,
                sensorObs: sensorMLX,
                action: actionMLX,
                h: hMLX
            )

            // Extract results and update state
            eval(result.residual, result.extensions, result.uncertainty, result.h)

            let residualValues = result.residual.squeezed(axis: 0).asArray(Float.self)
            let extensionValues = result.extensions.squeezed(axis: 0).asArray(Float.self)
            let uncertaintyValues = result.uncertainty.squeezed(axis: 0).asArray(Float.self)

            // Update internal state
            state.worldState.h = result.h.squeezed(axis: 0).asArray(Float.self)
            state.worldState.lastPhysics = physicsArray

            return try WorldModelOutput(
                residual: residualValues,
                extensions: extensionValues,
                uncertainty: uncertaintyValues
            )
        }
    }

    public mutating func predictFuture(
        steps: Int,
        actions: [[ActuatorValue]]
    ) throws -> [WorldModelOutput] {
        guard actions.count == steps else {
            throw ControllerError.stepCountMismatch(expected: steps, got: actions.count)
        }

        // Build actions tensor [1, steps, actionDim]
        var actionData: [Float] = []
        for stepActions in actions {
            let values = stepActions.map { Float($0.value) }
            actionData.append(contentsOf: values)
        }
        let actionsMLX = MLXArray(actionData).reshaped([1, steps, config.actionDimensions])

        return try storage.withLockThrowing { state in
            let hMLX = MLXArray(state.worldState.h).reshaped([1, config.hiddenDimensions])
            let physicsArray = state.worldState.lastPhysics
                ?? Array(repeating: Float(0), count: config.physicsDimensions)
            let physicsMLX = MLXArray(physicsArray).reshaped([1, config.physicsDimensions])

            let predictor = WorldPredictor(model: state.model)
            let predictions = predictor.rollout(
                initialH: hMLX,
                initialPhysics: physicsMLX,
                actions: actionsMLX
            )

            var outputs: [WorldModelOutput] = []
            for prediction in predictions {
                eval(prediction.residual, prediction.extensions, prediction.uncertainty)

                let residualValues = prediction.residual.squeezed(axis: 0).asArray(Float.self)
                let extensionValues = prediction.extensions.squeezed(axis: 0).asArray(Float.self)
                let uncertaintyValues = prediction.uncertainty.squeezed(axis: 0).asArray(Float.self)

                outputs.append(try WorldModelOutput(
                    residual: residualValues,
                    extensions: extensionValues,
                    uncertainty: uncertaintyValues
                ))
            }

            return outputs
        }
    }

    public mutating func reset() throws {
        storage.withLock { state in
            state.worldState = WorldModelState.initial(config: config)
        }
    }

    // MARK: - Private

    /// Convert sensor observations to a fixed-size float array.
    /// Fills in order of channel index, zero-padding missing channels.
    private func sensorObservationsToArray(_ observations: [ChannelSample]) -> [Float] {
        var result = Array<Float>(repeating: 0, count: config.sensorDimensions)
        for sample in observations {
            let index = Int(sample.channelIndex)
            if index < config.sensorDimensions {
                result[index] = Float(sample.value)
            }
        }
        return result
    }
}
