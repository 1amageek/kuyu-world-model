import Foundation
import Testing
import KuyuCore
@testable import KuyuWorldModel

@Test func rootBodyAnalyticalStateExportsThirteenDimensions() throws {
    let step = try makeEnvironmentStep()
    let state = RootBodyAnalyticalState(observation: step.observation)

    #expect(state.dimensions == 13)
    #expect(state.toArray().count == 13)
    #expect(state.toArray()[2] == 1.0)
}

@Test func learnedWorldModelEnvironmentAdapterAppliesResidualAndUncertainty() throws {
    let step = try makeEnvironmentStep()
    let model = FixedWorldModel(
        residual: residual(z: 0.25),
        uncertainty: Array(repeating: Float(0.05), count: 13)
    )
    let adapter = try LearnedWorldModelEnvironmentAdapter(
        model: model,
        timeStep: 0.001
    )

    let prediction = try adapter.predict(reference: step)
    #expect(prediction.step.observation.plantState.root.position.z == 1.25)
    #expect(prediction.step.observation.sensorSamples.first { $0.channelIndex == 6 }?.value == 1.25)
    #expect(prediction.step.log.sensorSamples.first { $0.channelIndex == 6 }?.value == 1.25)
    #expect(abs(prediction.uncertainty - 0.05) < 1e-6)

    let validation = try adapter.validate(
        prediction: prediction,
        reference: step,
        configuration: WorldModelAdapterConfiguration(
            residualThreshold: 0.3,
            uncertaintyThreshold: 0.1
        )
    )
    #expect(validation.accepted)
}

@Test func learnedWorldModelEnvironmentAdapterRejectsResidualBeyondGate() throws {
    let step = try makeEnvironmentStep()
    let model = FixedWorldModel(
        residual: residual(z: 1.0),
        uncertainty: Array(repeating: Float(0.0), count: 13)
    )
    let adapter = try LearnedWorldModelEnvironmentAdapter(
        model: model,
        timeStep: 0.001
    )

    let prediction = try adapter.predict(reference: step)
    #expect(throws: WorldModelAdapterRejection.residualExceeded(actual: 1.0, limit: 0.5)) {
        try adapter.accept(
            prediction: prediction,
            reference: step,
            configuration: WorldModelAdapterConfiguration(
                residualThreshold: 0.5,
                uncertaintyThreshold: 0.1
            )
        )
    }
}

@Test func learnedWorldModelEnvironmentAdapterRejectsInvalidTimeStep() {
    #expect(throws: LearnedWorldModelEnvironmentAdapter<FixedWorldModel>.AdapterError.invalidTimeStep(0)) {
        try LearnedWorldModelEnvironmentAdapter(
            model: FixedWorldModel(
                residual: residual(),
                uncertainty: []
            ),
            timeStep: 0
        )
    }
}

private struct FixedWorldModel: WorldModelProtocol {
    var residual: [Float]
    var uncertainty: [Float]

    mutating func infer(
        physicsPrediction: some AnalyticalState,
        sensorObservations: [ChannelSample],
        action: [ActuatorValue],
        dt: TimeInterval
    ) throws -> WorldModelOutput {
        try WorldModelOutput(
            residual: residual,
            extensions: [],
            uncertainty: uncertainty
        )
    }

    mutating func predictFuture(
        steps: Int,
        actions: [[ActuatorValue]]
    ) throws -> [WorldModelOutput] {
        let output = try WorldModelOutput(
            residual: residual,
            extensions: [],
            uncertainty: uncertainty
        )
        return Array(repeating: output, count: steps)
    }

    mutating func reset() throws {}
}

private func residual(
    x: Float = 0,
    y: Float = 0,
    z: Float = 0
) -> [Float] {
    [
        x, y, z,
        0, 0, 0,
        0, 0, 0, 0,
        0, 0, 0,
    ]
}

private func makeEnvironmentStep() throws -> EnvironmentStep {
    let log = try makeStepLog()
    return try EnvironmentStep(
        observation: EnvironmentObservation(log: log),
        reward: 1.0,
        done: false,
        truncated: false,
        info: EpisodeInfo(
            scenarioId: try ScenarioID("learned-adapter"),
            seed: ScenarioSeed(1),
            configHash: "hash",
            stepCount: 1,
            rewardSum: 1.0
        ),
        log: log
    )
}

private func makeStepLog() throws -> WorldStepLog {
    let time = try WorldTime(stepIndex: 1, time: 0.001)
    let root = RigidBodySnapshot(
        id: "root",
        position: Axis3(x: 0, y: 0, z: 1),
        velocity: Axis3(x: 0, y: 0, z: 0),
        orientation: QuaternionSnapshot(w: 1, x: 0, y: 0, z: 0),
        angularVelocity: Axis3(x: 0, y: 0, z: 0)
    )
    let sensorSamples = try (0..<8).map { index in
        try ChannelSample(
            channelIndex: UInt32(index),
            value: index == 6 ? 1.0 : 0.0,
            timestamp: time.time
        )
    }
    return WorldStepLog(
        time: time,
        events: [],
        sensorSamples: sensorSamples,
        driveIntents: [],
        reflexCorrections: [],
        actuatorValues: [],
        actuatorTelemetry: ActuatorTelemetrySnapshot(channels: []),
        safetyTrace: try SafetyTrace(omegaMagnitude: 0, tiltRadians: 0),
        plantState: PlantStateSnapshot(root: root),
        disturbances: DisturbanceSnapshot(
            forceWorld: Axis3(x: 0, y: 0, z: 0),
            torqueBody: Axis3(x: 0, y: 0, z: 0)
        )
    )
}
