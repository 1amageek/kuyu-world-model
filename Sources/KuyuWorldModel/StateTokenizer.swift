import MLX
import MLXNN

/// 1D causal convolutional encoder/decoder for time-series sensor observations.
///
/// Encoder: [batch, time_steps, sensorDim] -> [batch, time_steps, latentDim]
/// Decoder: [batch, time_steps, latentDim] -> [batch, time_steps, sensorDim]
///
/// Uses causal padding (left-pad only) so each time step only depends on past data.
/// Conv1d expects channels-last format: [batch, seq_len, channels].
public final class StateTokenizer: Module {

    @ModuleInfo public var encoderConvs: [Conv1d]
    @ModuleInfo public var encoderNorms: [LayerNorm]
    @ModuleInfo public var encoderProjection: Linear

    @ModuleInfo public var decoderProjection: Linear
    @ModuleInfo public var decoderLinears: [Linear]
    @ModuleInfo public var decoderNorms: [LayerNorm]

    private let kernelSize: Int
    private let numLayers: Int
    private let sensorDimensions: Int
    private let intermediateDimensions: Int

    public init(config: WorldModelConfig) {
        self.kernelSize = config.tokenizerKernelSize
        self.numLayers = config.tokenizerLayers
        self.sensorDimensions = config.sensorDimensions
        // Intermediate channel count for conv layers
        self.intermediateDimensions = config.physicsEmbedDimensions

        // Encoder: Conv1d stack with increasing channels
        var convs: [Conv1d] = []
        var norms: [LayerNorm] = []

        var currentChannels = config.sensorDimensions
        for i in 0..<config.tokenizerLayers {
            let outputChannels = (i == config.tokenizerLayers - 1)
                ? config.physicsEmbedDimensions
                : config.physicsEmbedDimensions
            // Conv1d with no built-in padding (we apply causal padding manually)
            convs.append(Conv1d(
                inputChannels: currentChannels,
                outputChannels: outputChannels,
                kernelSize: config.tokenizerKernelSize,
                padding: 0
            ))
            norms.append(LayerNorm(dimensions: outputChannels))
            currentChannels = outputChannels
        }
        self._encoderConvs.wrappedValue = convs
        self._encoderNorms.wrappedValue = norms
        // Final projection from conv output to physicsEmbedDim
        self._encoderProjection.wrappedValue = Linear(
            config.physicsEmbedDimensions, config.physicsEmbedDimensions
        )

        // Decoder: Linear stack to reconstruct sensor space (no transposed conv needed
        // because we preserve sequence length via causal padding in encoder)
        self._decoderProjection.wrappedValue = Linear(
            config.physicsEmbedDimensions, config.physicsEmbedDimensions
        )
        var decLinears: [Linear] = []
        var decNorms: [LayerNorm] = []
        var decChannels = config.physicsEmbedDimensions
        for i in 0..<config.tokenizerLayers {
            let outChannels = (i == config.tokenizerLayers - 1)
                ? config.sensorDimensions
                : config.physicsEmbedDimensions
            decLinears.append(Linear(decChannels, outChannels))
            decNorms.append(LayerNorm(dimensions: outChannels))
            decChannels = outChannels
        }
        self._decoderLinears.wrappedValue = decLinears
        self._decoderNorms.wrappedValue = decNorms
    }

    /// Encode sensor observations into latent space.
    /// Input: [batch, time_steps, sensorDim] -> Output: [batch, time_steps, physicsEmbedDim]
    public func encode(_ input: MLXArray) -> MLXArray {
        // input shape: [batch, time, channels] (channels-last for Conv1d)
        var h = input
        for i in 0..<numLayers {
            // Causal padding: pad left side only by (kernelSize - 1)
            let padAmount = kernelSize - 1
            let padded = padCausal(h, amount: padAmount)
            h = encoderConvs[i](padded)
            h = encoderNorms[i](h)
            h = relu(h)
        }
        h = encoderProjection(h)
        return h
    }

    /// Decode latent representations back to sensor space.
    /// Input: [batch, time_steps, physicsEmbedDim] -> Output: [batch, time_steps, sensorDim]
    public func decode(_ latent: MLXArray) -> MLXArray {
        var h = decoderProjection(latent)
        h = relu(h)
        for i in 0..<numLayers {
            h = decoderLinears[i](h)
            if i < numLayers - 1 {
                h = decoderNorms[i](h)
                h = relu(h)
            }
        }
        return h
    }

    /// Applies causal (left-only) zero padding along the sequence dimension (axis=1).
    /// Input: [batch, seq, channels] -> Output: [batch, seq + amount, channels]
    private func padCausal(_ x: MLXArray, amount: Int) -> MLXArray {
        guard amount > 0 else { return x }
        let batch = x.dim(0)
        let channels = x.dim(2)
        let zeros = MLXArray.zeros([batch, amount, channels])
        return concatenated([zeros, x], axis: 1)
    }
}
