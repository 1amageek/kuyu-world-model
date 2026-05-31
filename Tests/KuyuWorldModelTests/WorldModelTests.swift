import Foundation
import Testing
import MLX
import MLXNN
import MLXRandom
import KuyuCore
@testable import KuyuWorldModel

@Suite(.serialized) struct SerializedWorldModelMLXTests {

@Suite struct WorldModelConfigTests {

    @Test func configCreatesWithDefaults() {
        let config = WorldModelConfig()
        #expect(config.physicsDimensions == 13)
        #expect(config.sensorDimensions == 6)
        #expect(config.actionDimensions == 4)
        #expect(config.hiddenDimensions == 128)
        #expect(config.stochasticCategories == 8)
        #expect(config.stochasticClasses == 8)
        #expect(config.stochasticLatentSize == 64)
        #expect(config.rssmEnabled == true)
        #expect(config.residualDimensions == 13)
        #expect(config.extensionDimensions == 16)
        #expect(config.decoderInputDimensions == 128 + 64)
    }

    @Test func configDecodesFromJSON() throws {
        let json = """
        {"physicsDimensions": 7, "sensorDimensions": 3}
        """
        let data = json.data(using: .utf8)!
        let config = try JSONDecoder().decode(WorldModelConfig.self, from: data)
        #expect(config.physicsDimensions == 7)
        #expect(config.sensorDimensions == 3)
        // Defaults for missing keys
        #expect(config.hiddenDimensions == 128)
        #expect(config.stochasticCategories == 8)
    }
}

@Suite struct PhysicsEncoderTests {

    @Test func forwardProducesCorrectShape() {
        let config = WorldModelConfig()
        let encoder = PhysicsEncoder(config: config)
        let input = MLXArray.zeros([2, config.physicsDimensions])
        let output = encoder(input)
        eval(output)
        #expect(output.shape == [2, config.physicsEmbedDimensions])
    }

    @Test func singleBatchForward() {
        let config = WorldModelConfig()
        let encoder = PhysicsEncoder(config: config)
        let input = MLXArray.zeros([1, config.physicsDimensions])
        let output = encoder(input)
        eval(output)
        #expect(output.shape == [1, config.physicsEmbedDimensions])
    }
}

@Suite struct StateTokenizerTests {

    @Test func encodeProducesCorrectShape() {
        let config = WorldModelConfig()
        let tokenizer = StateTokenizer(config: config)
        let input = MLXArray.zeros([2, 10, config.sensorDimensions])
        let encoded = tokenizer.encode(input)
        eval(encoded)
        #expect(encoded.shape[0] == 2)
        #expect(encoded.shape[1] == 10)  // Causal padding preserves sequence length
        #expect(encoded.shape[2] == config.physicsEmbedDimensions)
    }

    @Test func decodeProducesCorrectShape() {
        let config = WorldModelConfig()
        let tokenizer = StateTokenizer(config: config)
        let latent = MLXArray.zeros([2, 10, config.physicsEmbedDimensions])
        let decoded = tokenizer.decode(latent)
        eval(decoded)
        #expect(decoded.shape == [2, 10, config.sensorDimensions])
    }
}

@Suite struct GRUCellTests {

    @Test func singleStepForward() {
        let cell = GRUCell(inputSize: 32, hiddenSize: 64)
        let x = MLXArray.zeros([2, 32])
        let h = MLXArray.zeros([2, 64])
        let newH = cell(x: x, h: h)
        eval(newH)
        #expect(newH.shape == [2, 64])
    }
}

@Suite struct TransitionModelTests {

    @Test func priorOnlyForward() {
        let config = WorldModelConfig()
        let model = TransitionModel(config: config)
        let h = MLXArray.zeros([2, config.hiddenDimensions])
        let action = MLXArray.zeros([2, config.actionDimensions])
        let physEmb = MLXArray.zeros([2, config.physicsEmbedDimensions])
        let (newH, priorLogits) = model.forward(h: h, action: action, physicsEmbed: physEmb)
        eval(newH, priorLogits)
        #expect(newH.shape == [2, config.hiddenDimensions])
        #expect(priorLogits.shape == [2, config.stochasticLatentSize])
    }

    @Test func posteriorForward() {
        let config = WorldModelConfig()
        let model = TransitionModel(config: config)
        let h = MLXArray.zeros([2, config.hiddenDimensions])
        let action = MLXArray.zeros([2, config.actionDimensions])
        let physEmb = MLXArray.zeros([2, config.physicsEmbedDimensions])
        let obsEmb = MLXArray.zeros([2, config.physicsEmbedDimensions])
        let (newH, priorLogits, posteriorLogits) = model.forward(
            h: h, action: action, physicsEmbed: physEmb, obsEmbed: obsEmb
        )
        eval(newH, priorLogits, posteriorLogits)
        #expect(newH.shape == [2, config.hiddenDimensions])
        #expect(priorLogits.shape == [2, config.stochasticLatentSize])
        #expect(posteriorLogits.shape == [2, config.stochasticLatentSize])
    }
}

@Suite struct DecoderTests {

    @Test func residualDecoderShape() {
        let config = WorldModelConfig()
        let decoder = ResidualDecoder(config: config)
        let input = MLXArray.zeros([2, config.decoderInputDimensions])
        let output = decoder(input)
        eval(output)
        #expect(output.shape == [2, config.residualDimensions])
    }

    @Test func residualDecoderStartsAtZero() {
        let config = WorldModelConfig()
        let decoder = ResidualDecoder(config: config)
        let input = MLXRandom.uniform(0 ..< 1, [2, config.decoderInputDimensions])
        let output = decoder(input)
        eval(output)
        let values = output.asArray(Float.self)
        #expect(values.allSatisfy { abs($0) <= 1e-6 })
    }

    @Test func extensionDecoderShape() {
        let config = WorldModelConfig()
        let decoder = ExtensionDecoder(config: config)
        let input = MLXArray.zeros([2, config.decoderInputDimensions])
        let output = decoder(input)
        eval(output)
        #expect(output.shape == [2, config.extensionDimensions])
    }

    @Test func extensionDecoderStartsAtZero() {
        let config = WorldModelConfig()
        let decoder = ExtensionDecoder(config: config)
        let input = MLXRandom.uniform(0 ..< 1, [2, config.decoderInputDimensions])
        let output = decoder(input)
        eval(output)
        let values = output.asArray(Float.self)
        #expect(values.allSatisfy { abs($0) <= 1e-6 })
    }

    @Test func uncertaintyDecoderShape() {
        let config = WorldModelConfig()
        let decoder = UncertaintyDecoder(config: config)
        let input = MLXArray.zeros([2, config.decoderInputDimensions])
        let output = decoder(input)
        eval(output)
        #expect(output.shape == [2, config.residualDimensions + config.extensionDimensions])
    }

    @Test func uncertaintyOutputBounded() {
        let config = WorldModelConfig()
        let decoder = UncertaintyDecoder(config: config)
        let input = MLXRandom.uniform(0 ..< 1, [4, config.decoderInputDimensions])
        let output = decoder(input)
        eval(output)
        let minVal = min(output).item(Float.self)
        let maxVal = max(output).item(Float.self)
        // Sigmoid output must be in [0, 1]
        #expect(minVal >= 0.0)
        #expect(maxVal <= 1.0)
    }

    @Test func uncertaintyDecoderStartsNeutral() {
        let config = WorldModelConfig()
        let decoder = UncertaintyDecoder(config: config)
        let input = MLXRandom.uniform(0 ..< 1, [2, config.decoderInputDimensions])
        let output = decoder(input)
        eval(output)
        let values = output.asArray(Float.self)
        #expect(values.allSatisfy { abs($0 - 0.5) <= 1e-6 })
    }
}

@Suite struct StateWorldModelTests {

    @Test func stepProducesCorrectDimensions() {
        let config = WorldModelConfig()
        let model = StateWorldModel(config: config)
        let physics = MLXArray.zeros([1, config.physicsDimensions])
        let sensor = MLXArray.zeros([1, config.sensorDimensions])
        let action = MLXArray.zeros([1, config.actionDimensions])
        let h = MLXArray.zeros([1, config.hiddenDimensions])
        let result = model.step(physicsState: physics, sensorObs: sensor, action: action, h: h)
        eval(result.residual, result.extensions, result.uncertainty, result.h)
        #expect(result.residual.shape == [1, config.residualDimensions])
        #expect(result.extensions.shape == [1, config.extensionDimensions])
        #expect(result.uncertainty.shape == [1, config.residualDimensions + config.extensionDimensions])
        #expect(result.h.shape == [1, config.hiddenDimensions])
    }

    @Test func sequenceForwardProducesCorrectDimensions() {
        let config = WorldModelConfig()
        let model = StateWorldModel(config: config)
        let seq = 5
        let batch = 2
        let physics = MLXArray.zeros([batch, seq, config.physicsDimensions])
        let sensor = MLXArray.zeros([batch, seq, config.sensorDimensions])
        let actions = MLXArray.zeros([batch, seq, config.actionDimensions])
        let output = model.forward(physicsStates: physics, sensorObs: sensor, actions: actions)
        eval(output.residual, output.extensions, output.uncertainty, output.priorLogits, output.posteriorLogits, output.finalH)
        #expect(output.residual.shape == [batch, seq, config.residualDimensions])
        #expect(output.extensions.shape == [batch, seq, config.extensionDimensions])
        #expect(output.uncertainty.shape == [batch, seq, config.residualDimensions + config.extensionDimensions])
        #expect(output.priorLogits.shape == [batch, seq, config.stochasticLatentSize])
        #expect(output.posteriorLogits.shape == [batch, seq, config.stochasticLatentSize])
        #expect(output.finalH.shape == [batch, config.hiddenDimensions])
    }

    @Test func imaginationStepProducesCorrectDimensions() {
        let config = WorldModelConfig()
        let model = StateWorldModel(config: config)
        let physics = MLXArray.zeros([1, config.physicsDimensions])
        let action = MLXArray.zeros([1, config.actionDimensions])
        let h = MLXArray.zeros([1, config.hiddenDimensions])
        let result = model.imaginationStep(physicsState: physics, action: action, h: h)
        eval(result.residual, result.extensions, result.uncertainty, result.h)
        #expect(result.residual.shape == [1, config.residualDimensions])
        #expect(result.extensions.shape == [1, config.extensionDimensions])
        #expect(result.h.shape == [1, config.hiddenDimensions])
    }
}

@Suite struct WorldPredictorTests {

    @Test func rolloutProducesCorrectStepCount() {
        let config = WorldModelConfig()
        let model = StateWorldModel(config: config)
        let predictor = WorldPredictor(model: model)
        let steps = 3
        let h = MLXArray.zeros([1, config.hiddenDimensions])
        let physics = MLXArray.zeros([1, config.physicsDimensions])
        let actions = MLXArray.zeros([1, steps, config.actionDimensions])
        let predictions = predictor.rollout(initialH: h, initialPhysics: physics, actions: actions)
        #expect(predictions.count == steps)
        for pred in predictions {
            eval(pred.residual, pred.extensions, pred.uncertainty, pred.h)
            #expect(pred.residual.shape == [1, config.residualDimensions])
            #expect(pred.h.shape == [1, config.hiddenDimensions])
        }
    }

    @Test func rolloutUsesPhysicsAdvanceCallback() {
        let config = WorldModelConfig()
        let model = StateWorldModel(config: config)
        let predictor = WorldPredictor(model: model)
        let steps = 3
        let h = MLXArray.zeros([1, config.hiddenDimensions])
        let physics = MLXArray.zeros([1, config.physicsDimensions])
        let actions = MLXArray.zeros([1, steps, config.actionDimensions])
        var visitedSteps: [Int] = []

        let predictions = predictor.rollout(
            initialH: h,
            initialPhysics: physics,
            actions: actions,
            advancePhysics: { currentPhysics, _, residual, stepIndex in
                visitedSteps.append(stepIndex)
                return currentPhysics + residual
            }
        )

        #expect(predictions.count == steps)
        #expect(visitedSteps == [0, 1, 2])
    }
}

@Suite struct MLXWorldModelControllerTests {

    @Test func inferRejectsInvalidTimeStep() throws {
        let config = smallControllerConfig()
        var controller = MLXWorldModelController(model: StateWorldModel(config: config), config: config)

        #expect(throws: MLXWorldModelController.ControllerError.invalidTimeStep(0)) {
            _ = try controller.infer(
                physicsPrediction: FixedAnalyticalState(values: [0, 0, 0]),
                sensorObservations: [],
                action: try actuatorValues(count: 1),
                dt: 0
            )
        }
    }

    @Test func inferRejectsNonFinitePhysicsState() throws {
        let config = smallControllerConfig()
        var controller = MLXWorldModelController(model: StateWorldModel(config: config), config: config)

        #expect(throws: MLXWorldModelController.ControllerError.nonFinitePhysicsState(index: 1)) {
            _ = try controller.infer(
                physicsPrediction: FixedAnalyticalState(values: [0, .nan, 0]),
                sensorObservations: [],
                action: try actuatorValues(count: 1),
                dt: 0.01
            )
        }
    }

    @Test func inferRejectsOutOfRangeSensorChannel() throws {
        let config = smallControllerConfig()
        var controller = MLXWorldModelController(model: StateWorldModel(config: config), config: config)
        let samples = [
            try ChannelSample(channelIndex: 2, value: 1.0, timestamp: 0.0),
        ]

        #expect(throws: MLXWorldModelController.ControllerError.sensorChannelOutOfRange(channelIndex: 2, limit: 2)) {
            _ = try controller.infer(
                physicsPrediction: FixedAnalyticalState(values: [0, 0, 0]),
                sensorObservations: samples,
                action: try actuatorValues(count: 1),
                dt: 0.01
            )
        }
    }

    @Test func predictFutureRejectsPerStepActionDimensionMismatch() throws {
        let config = smallControllerConfig()
        var controller = MLXWorldModelController(model: StateWorldModel(config: config), config: config)

        #expect(throws: MLXWorldModelController.ControllerError.actionCountMismatch(expected: 1, got: 2)) {
            _ = try controller.predictFuture(
                steps: 1,
                actions: [try actuatorValues(count: 2)]
            )
        }
    }

    @Test func predictFutureAllowsZeroSteps() throws {
        let config = smallControllerConfig()
        var controller = MLXWorldModelController(model: StateWorldModel(config: config), config: config)

        let outputs = try controller.predictFuture(steps: 0, actions: [])
        #expect(outputs.isEmpty)
    }
}

@Suite struct DomainAdapterTests {

    @Test func adapterPreservesShape() {
        let config = WorldModelConfig()
        let adapter = DomainAdapter(config: config)
        let residual = MLXArray.zeros([2, config.residualDimensions])
        let ext = MLXArray.zeros([2, config.extensionDimensions])
        let (adaptedRes, adaptedExt) = adapter(simResidual: residual, simExtension: ext)
        eval(adaptedRes, adaptedExt)
        #expect(adaptedRes.shape == [2, config.residualDimensions])
        #expect(adaptedExt.shape == [2, config.extensionDimensions])
    }

    @Test func adapterStartsAsIdentity() {
        let config = WorldModelConfig()
        let adapter = DomainAdapter(config: config)
        let residualValues = (0..<config.residualDimensions).map { Float($0) / 10 }
        let extensionValues = (0..<config.extensionDimensions).map { Float($0) / 20 }
        let residual = MLXArray(residualValues).reshaped([1, config.residualDimensions])
        let ext = MLXArray(extensionValues).reshaped([1, config.extensionDimensions])

        let (adaptedRes, adaptedExt) = adapter(simResidual: residual, simExtension: ext)
        eval(adaptedRes, adaptedExt)

        expectAllClose(adaptedRes.asArray(Float.self), residualValues)
        expectAllClose(adaptedExt.asArray(Float.self), extensionValues)
    }
}

@Suite struct WorldModelStateTests {

    @Test func initialStateHasCorrectDimensions() {
        let config = WorldModelConfig()
        let state = WorldModelState.initial(config: config)
        #expect(state.h.count == config.hiddenDimensions)
        #expect(state.z?.count == config.stochasticLatentSize)
    }

    @Test func initialStateWithDisabledRSSM() {
        let config = WorldModelConfig(stochasticCategories: 0, stochasticClasses: 0)
        let state = WorldModelState.initial(config: config)
        #expect(state.h.count == config.hiddenDimensions)
        #expect(state.z == nil)
    }
}

@Suite struct StateWorldModelTrainerTests {

    @Test func trainerConfigStoresKLWeight() {
        let config = StateWorldModelTrainer.Config(klWeight: 0.25, uncertaintyVarianceFloor: 1e-3)
        #expect(config.klWeight == 0.25)
        #expect(config.uncertaintyVarianceFloor == 1e-3)
    }

    @Test func uncertaintyNLLRewardsHigherUncertaintyForLargeResidualError() {
        let predictions = MLXArray([Float(0), 0]).reshaped([1, 1, 2])
        let targets = MLXArray([Float(0), 1]).reshaped([1, 1, 2])
        let lowUncertainty = MLXArray([Float(0.1), 0.1]).reshaped([1, 1, 2])
        let calibratedUncertainty = MLXArray([Float(0.1), 0.9]).reshaped([1, 1, 2])

        let lowLoss = StateWorldModelTrainer.residualUncertaintyNLL(
            predictions: predictions,
            targets: targets,
            uncertainty: lowUncertainty,
            varianceFloor: 1e-4
        )
        let calibratedLoss = StateWorldModelTrainer.residualUncertaintyNLL(
            predictions: predictions,
            targets: targets,
            uncertainty: calibratedUncertainty,
            varianceFloor: 1e-4
        )
        eval(lowLoss, calibratedLoss)

        #expect(calibratedLoss.item(Float.self) < lowLoss.item(Float.self))
    }

    @Test func trainerProducesFiniteLossForSmallResidualBatch() {
        let config = WorldModelConfig(
            physicsDimensions: 3,
            sensorDimensions: 2,
            actionDimensions: 1,
            hiddenDimensions: 8,
            stochasticCategories: 2,
            stochasticClasses: 2,
            residualDimensions: 3,
            extensionDimensions: 2,
            tokenizerLayers: 1,
            physicsEmbedDimensions: 8
        )
        let model = StateWorldModel(config: config)
        let batch = StateWorldModelTrainingBatch(
            physicsStates: MLXArray.zeros([1, 2, config.physicsDimensions]),
            sensorObservations: MLXArray.zeros([1, 2, config.sensorDimensions]),
            actions: MLXArray.zeros([1, 2, config.actionDimensions]),
            residualTargets: MLXArray.zeros([1, 2, config.residualDimensions])
        )

        let losses = StateWorldModelTrainer.train(
            model: model,
            batches: [batch],
            learningRate: 0.001,
            maxGradNorm: nil,
            epochs: 1
        )

        #expect(losses.count == 1)
        #expect(losses[0].isFinite)
    }
}

}

private func smallControllerConfig() -> WorldModelConfig {
    WorldModelConfig(
        physicsDimensions: 3,
        sensorDimensions: 2,
        actionDimensions: 1,
        hiddenDimensions: 8,
        stochasticCategories: 2,
        stochasticClasses: 2,
        residualDimensions: 3,
        extensionDimensions: 2,
        tokenizerLayers: 1,
        physicsEmbedDimensions: 8
    )
}

private func actuatorValues(count: Int) throws -> [ActuatorValue] {
    try (0..<count).map { index in
        try ActuatorValue(index: ActuatorIndex(UInt32(index)), value: 0)
    }
}

private func expectAllClose(_ actual: [Float], _ expected: [Float], tolerance: Float = 1e-6) {
    #expect(actual.count == expected.count)
    for (actualValue, expectedValue) in zip(actual, expected) {
        #expect(abs(actualValue - expectedValue) <= tolerance)
    }
}

private struct FixedAnalyticalState: AnalyticalState {
    let values: [Float]

    var dimensions: Int { values.count }

    func toArray() -> [Float] {
        values
    }

    func toPlantStateSnapshot() -> PlantStateSnapshot {
        let root = RigidBodySnapshot(
            id: "root",
            position: Axis3(x: 0, y: 0, z: 0),
            velocity: Axis3(x: 0, y: 0, z: 0),
            orientation: QuaternionSnapshot(w: 1, x: 0, y: 0, z: 0),
            angularVelocity: Axis3(x: 0, y: 0, z: 0)
        )
        return PlantStateSnapshot(root: root)
    }
}
