import MLX
import MLXNN

/// Rolls out the world model transition for N steps using prior only (no observations).
/// Used for imagination-based RL training (DreamerV3 pattern).
public final class WorldPredictor: Module {

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
    /// - Returns: array of WorldPrediction for each step
    public func rollout(
        initialH: MLXArray,
        initialPhysics: MLXArray,
        actions: MLXArray
    ) -> [WorldPrediction] {
        let steps = actions.dim(1)
        var h = initialH
        var currentPhysics = initialPhysics
        var predictions: [WorldPrediction] = []

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

            // Update physics state with residual for next step's encoding
            // (the world model predicts corrections; apply them for the next step)
            currentPhysics = currentPhysics + result.residual
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
}
