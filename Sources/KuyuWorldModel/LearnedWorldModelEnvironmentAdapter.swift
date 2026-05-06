import Foundation
import Synchronization
import KuyuCore

/// Environment adapter that validates learned world-model residuals against
/// the analytical physics step without mutating the environment's physics state.
public struct LearnedWorldModelEnvironmentAdapter<Model: WorldModelProtocol>: WorldModelEnvironmentAdapter {
    public enum AdapterError: Error, Equatable {
        case invalidTimeStep(Double)
        case residualDimensionMismatch(expected: Int, actual: Int)
    }

    private final class Storage: Sendable {
        private let mutex: Mutex<Model>

        init(model: Model) {
            self.mutex = Mutex(model)
        }

        func infer(
            physicsPrediction: RootBodyAnalyticalState,
            sensorObservations: [ChannelSample],
            action: [ActuatorValue],
            dt: TimeInterval
        ) throws -> WorldModelOutput {
            try mutex.withLock { model in
                try model.infer(
                    physicsPrediction: physicsPrediction,
                    sensorObservations: sensorObservations,
                    action: action,
                    dt: dt
                )
            }
        }

        func reset() throws {
            try mutex.withLock { model in
                try model.reset()
            }
        }
    }

    private let storage: Storage
    private let timeStep: TimeInterval
    private let validator: PhysicsOnlyWorldModelAdapter

    public init(
        model: Model,
        timeStep: TimeInterval,
        validator: PhysicsOnlyWorldModelAdapter = PhysicsOnlyWorldModelAdapter()
    ) throws {
        guard timeStep.isFinite, timeStep > 0 else {
            throw AdapterError.invalidTimeStep(timeStep)
        }
        self.storage = Storage(model: model)
        self.timeStep = timeStep
        self.validator = validator
    }

    public func predict(reference: EnvironmentStep) throws -> WorldModelPrediction {
        let physicsState = RootBodyAnalyticalState(observation: reference.observation)
        let output = try storage.infer(
            physicsPrediction: physicsState,
            sensorObservations: reference.log.sensorSamples,
            action: reference.log.actuatorValues,
            dt: timeStep
        )
        let predictedStep = try apply(output: output, to: reference)
        return try WorldModelPrediction(
            step: predictedStep,
            uncertainty: maxUncertainty(output)
        )
    }

    public func validate(predicted: EnvironmentStep, reference: EnvironmentStep) throws -> WorldModelAdapterValidation {
        try validator.validate(predicted: predicted, reference: reference)
    }

    public func validate(
        prediction: WorldModelPrediction,
        reference: EnvironmentStep,
        configuration: WorldModelAdapterConfiguration
    ) throws -> WorldModelAdapterValidation {
        try validator.validate(prediction: prediction, reference: reference, configuration: configuration)
    }

    public func accept(
        prediction: WorldModelPrediction,
        reference: EnvironmentStep,
        configuration: WorldModelAdapterConfiguration
    ) throws -> EnvironmentStep {
        try validator.accept(prediction: prediction, reference: reference, configuration: configuration)
    }

    public func reset() throws {
        try storage.reset()
    }

    private func apply(output: WorldModelOutput, to reference: EnvironmentStep) throws -> EnvironmentStep {
        let residual = output.residual.map(Double.init)
        guard residual.count == RootBodyAnalyticalState.dimensionCount else {
            throw AdapterError.residualDimensionMismatch(
                expected: RootBodyAnalyticalState.dimensionCount,
                actual: residual.count
            )
        }

        let plantState = try correctedPlantState(
            reference.observation.plantState,
            residual: residual
        )
        let observationSensorSamples = try correctedSensorSamples(
            reference.observation.sensorSamples,
            plantState: plantState
        )
        let logSensorSamples = try correctedSensorSamples(
            reference.log.sensorSamples,
            plantState: plantState
        )
        let observation = EnvironmentObservation(
            time: reference.observation.time,
            sensorSamples: observationSensorSamples,
            plantState: plantState,
            safetyTrace: reference.observation.safetyTrace,
            actuatorTelemetry: reference.observation.actuatorTelemetry,
            disturbances: reference.observation.disturbances
        )
        let log = WorldStepLog(
            time: reference.log.time,
            events: reference.log.events,
            sensorSamples: logSensorSamples,
            driveIntents: reference.log.driveIntents,
            reflexCorrections: reference.log.reflexCorrections,
            actuatorValues: reference.log.actuatorValues,
            actuatorTelemetry: reference.log.actuatorTelemetry,
            motorNerveTrace: reference.log.motorNerveTrace,
            safetyTrace: reference.log.safetyTrace,
            plantState: plantState,
            disturbances: reference.log.disturbances
        )
        return try EnvironmentStep(
            observation: observation,
            reward: reference.reward,
            done: reference.done,
            truncated: reference.truncated,
            info: reference.info,
            log: log
        )
    }

    private func correctedPlantState(
        _ plantState: PlantStateSnapshot,
        residual: [Double]
    ) throws -> PlantStateSnapshot {
        let root = plantState.root
        let correctedRoot = RigidBodySnapshot(
            id: root.id,
            position: Axis3(
                x: root.position.x + residual[0],
                y: root.position.y + residual[1],
                z: root.position.z + residual[2]
            ),
            velocity: Axis3(
                x: root.velocity.x + residual[3],
                y: root.velocity.y + residual[4],
                z: root.velocity.z + residual[5]
            ),
            orientation: QuaternionSnapshot(
                w: root.orientation.w + residual[6],
                x: root.orientation.x + residual[7],
                y: root.orientation.y + residual[8],
                z: root.orientation.z + residual[9]
            ),
            angularVelocity: Axis3(
                x: root.angularVelocity.x + residual[10],
                y: root.angularVelocity.y + residual[11],
                z: root.angularVelocity.z + residual[12]
            )
        )
        return PlantStateSnapshot(
            root: correctedRoot,
            bodies: plantState.bodies,
            scalars: plantState.scalars
        )
    }

    private func correctedSensorSamples(
        _ samples: [ChannelSample],
        plantState: PlantStateSnapshot
    ) throws -> [ChannelSample] {
        try samples.map { sample in
            let value: Double
            switch sample.channelIndex {
            case 6:
                value = plantState.root.position.z
            case 7:
                value = plantState.root.velocity.z
            default:
                value = sample.value
            }
            return try ChannelSample(
                channelIndex: sample.channelIndex,
                value: value,
                timestamp: sample.timestamp
            )
        }
    }

    private func maxUncertainty(_ output: WorldModelOutput) -> Double {
        output.uncertainty
            .map(Double.init)
            .max() ?? 0.0
    }
}
