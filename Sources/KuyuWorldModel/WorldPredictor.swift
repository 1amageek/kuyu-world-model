import MLX
import MLXNN

/// Rolls out the world model transition for N steps using prior only.
///
/// The model stays independent from analytical physics. Callers that own a physics
/// integrator can provide `PhysicsAdvance` so imagination uses the real physics step
/// plus learned residuals; the default keeps the previous residual-only behavior.
public final class WorldPredictor: Module {

    public typealias PhysicsAdvance = (
        _ currentPhysics: MLXArray,
        _ action: MLXArray,
        _ residual: MLXArray,
        _ stepIndex: Int
    ) -> MLXArray

    @ModuleInfo public var model: StateWorldModel

    public init(model: StateWorldModel) {
        self._model.wrappedValue = model
    }

    /// Rollout the world model for multiple steps.
    ///
    /// Uses prior-only transitions (imagination mode) since no real
    /// observations are available for future steps.
    ///
    /// - Parameters:
    ///   - initialH: initial hidden state [batch, hiddenDim]
    ///   - initialPhysics: initial physics state [batch, physicsDim]
    ///   - actions: sequence of actions [batch, steps, actionDim]
    ///   - advancePhysics: optional physics integrator callback for the next state
    /// - Returns: array of WorldPrediction for each step
    public func rollout(
        initialH: MLXArray,
        initialPhysics: MLXArray,
        actions: MLXArray,
        advancePhysics: PhysicsAdvance? = nil
    ) -> [WorldPrediction] {
        let steps = actions.dim(1)
        var h = initialH
        var currentPhysics = initialPhysics
        var predictions: [WorldPrediction] = []
        let advancePhysics = advancePhysics ?? Self.residualOnlyAdvance

        for t in 0..<steps {
            let action = actions[.ellipsis, t, 0...]  // [batch, actionDim]

            let result = model.imaginationStep(
                physicsState: currentPhysics,
                action: action,
                h: h
            )

            let prediction = WorldPrediction(
                residual: result.residual,
                extensions: result.extensions,
                uncertainty: result.uncertainty,
                h: result.h
            )
            predictions.append(prediction)

            h = result.h

            currentPhysics = advancePhysics(currentPhysics, action, result.residual, t)
        }

        return predictions
    }

    public func callAsFunction(
        initialH: MLXArray,
        initialPhysics: MLXArray,
        actions: MLXArray
    ) -> [WorldPrediction] {
        rollout(initialH: initialH, initialPhysics: initialPhysics, actions: actions)
    }

    public static func residualOnlyAdvance(
        currentPhysics: MLXArray,
        action: MLXArray,
        residual: MLXArray,
        stepIndex: Int
    ) -> MLXArray {
        currentPhysics + residual
    }
}
