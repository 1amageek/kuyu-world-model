import MLX
import MLXNN

enum LinearInitializers {
    static func zero(inputDimensions: Int, outputDimensions: Int) -> Linear {
        Linear(
            weight: MLXArray.zeros([outputDimensions, inputDimensions]),
            bias: MLXArray.zeros([outputDimensions])
        )
    }

    static func identity(dimensions: Int) -> Linear {
        Linear(
            weight: MLXArray.identity(dimensions),
            bias: MLXArray.zeros([dimensions])
        )
    }
}
