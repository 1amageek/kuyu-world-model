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
public struct MLXWorldModelController: PhysicsAwareWorldModelProtocol {

    public enum ControllerError: Error, Equatable {
        case invalidTimeStep(Double)
        case invalidStepCount(Int)
        case dimensionMismatch(expected: Int, got: Int)
        case actionCountMismatch(expected: Int, got: Int)
        case stepCountMismatch(expected: Int, got: Int)
        case physicsPredictionCountMismatch(expected: Int, got: Int)
        case sensorChannelOutOfRange(channelIndex: UInt32, limit: Int)
        case nonFinitePhysicsState(index: Int)
        case nonFiniteAction(index: Int)
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
        guard dt.isFinite, dt > 0 else {
            throw ControllerError.invalidTimeStep(dt)
        }

        // Convert physics prediction to MLXArray [1, physicsDim]
        let physicsArray = physicsPrediction.toArray()
        guard physicsArray.count == config.physicsDimensions else {
            throw ControllerError.dimensionMismatch(expected: config.physicsDimensions, got: physicsArray.count)
        }
        try validateFinitePhysicsArray(physicsArray)
        let physicsMLX = MLXArray(physicsArray).reshaped([1, config.physicsDimensions])

        // Convert sensor observations to MLXArray [1, sensorDim]
        let sensorValues = try sensorObservationsToArray(sensorObservations)
        let sensorMLX = MLXArray(sensorValues).reshaped([1, config.sensorDimensions])

        // Convert actions to MLXArray [1, actionDim]
        let actionValues = try actionValues(from: action)
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
        guard steps >= 0 else {
            throw ControllerError.invalidStepCount(steps)
        }
        guard actions.count == steps else {
            throw ControllerError.stepCountMismatch(expected: steps, got: actions.count)
        }
        guard steps > 0 else {
            return []
        }

        // Build actions tensor [1, steps, actionDim]
        var actionData: [Float] = []
        for stepActions in actions {
            let values = try actionValues(from: stepActions)
            guard values.count == config.actionDimensions else {
                throw ControllerError.actionCountMismatch(expected: config.actionDimensions, got: values.count)
            }
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

    public func predictFuture(
        physicsPredictions: [[Float]],
        actions: [[ActuatorValue]],
        dt: TimeInterval
    ) throws -> [WorldModelOutput] {
        guard dt.isFinite, dt > 0 else {
            throw ControllerError.invalidTimeStep(dt)
        }
        guard physicsPredictions.count == actions.count else {
            throw ControllerError.physicsPredictionCountMismatch(
                expected: actions.count,
                got: physicsPredictions.count
            )
        }
        guard !physicsPredictions.isEmpty else {
            return []
        }

        var preparedSteps: [(physics: [Float], action: [Float])] = []
        preparedSteps.reserveCapacity(physicsPredictions.count)
        for (physicsPrediction, stepActions) in zip(physicsPredictions, actions) {
            guard physicsPrediction.count == config.physicsDimensions else {
                throw ControllerError.dimensionMismatch(
                    expected: config.physicsDimensions,
                    got: physicsPrediction.count
                )
            }
            try validateFinitePhysicsArray(physicsPrediction)

            let actionValues = try actionValues(from: stepActions)
            guard actionValues.count == config.actionDimensions else {
                throw ControllerError.actionCountMismatch(
                    expected: config.actionDimensions,
                    got: actionValues.count
                )
            }
            preparedSteps.append((physicsPrediction, actionValues))
        }

        return try storage.withLockThrowing { state in
            var h = MLXArray(state.worldState.h).reshaped([1, config.hiddenDimensions])
            var outputs: [WorldModelOutput] = []
            outputs.reserveCapacity(preparedSteps.count)

            for step in preparedSteps {
                let physicsMLX = MLXArray(step.physics).reshaped([1, config.physicsDimensions])
                let actionMLX = MLXArray(step.action).reshaped([1, config.actionDimensions])
                let result = state.model.imaginationStep(
                    physicsState: physicsMLX,
                    action: actionMLX,
                    h: h
                )
                eval(result.residual, result.extensions, result.uncertainty, result.h)

                h = result.h
                outputs.append(try WorldModelOutput(
                    residual: result.residual.squeezed(axis: 0).asArray(Float.self),
                    extensions: result.extensions.squeezed(axis: 0).asArray(Float.self),
                    uncertainty: result.uncertainty.squeezed(axis: 0).asArray(Float.self)
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
    private func sensorObservationsToArray(_ observations: [ChannelSample]) throws -> [Float] {
        var result = Array<Float>(repeating: 0, count: config.sensorDimensions)
        for sample in observations {
            let index = Int(sample.channelIndex)
            guard index < config.sensorDimensions else {
                throw ControllerError.sensorChannelOutOfRange(
                    channelIndex: sample.channelIndex,
                    limit: config.sensorDimensions
                )
            }
            result[index] = Float(sample.value)
        }
        return result
    }

    private func validateFinitePhysicsArray(_ values: [Float]) throws {
        for (index, value) in values.enumerated() where !value.isFinite {
            throw ControllerError.nonFinitePhysicsState(index: index)
        }
    }

    private func actionValues(from actions: [ActuatorValue]) throws -> [Float] {
        try actions.enumerated().map { index, action in
            let value = Float(action.value)
            guard value.isFinite else {
                throw ControllerError.nonFiniteAction(index: index)
            }
            return value
        }
    }
}
