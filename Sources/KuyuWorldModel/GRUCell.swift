import Foundation
import MLX
import MLXNN

/// A single-step GRU cell for use in the transition model.
///
/// Unlike MLXNN.GRU which processes full sequences, this operates on
/// a single time step, making it suitable for step-by-step rollouts.
///
/// Input: x [batch, inputDim], h [batch, hiddenDim]
/// Output: h_new [batch, hiddenDim]
public final class GRUCell: Module {

    public let hiddenSize: Int

    /// Gate weights: [3 * hiddenSize, inputDim]
    @ParameterInfo(key: "Wx") public var wx: MLXArray
    /// Recurrent weights: [3 * hiddenSize, hiddenSize]
    @ParameterInfo(key: "Wh") public var wh: MLXArray
    /// Input bias: [3 * hiddenSize]
    public let b: MLXArray?
    /// Recurrent bias for the new gate: [hiddenSize]
    public let bhn: MLXArray?

    public init(inputSize: Int, hiddenSize: Int, bias: Bool = true) {
        self.hiddenSize = hiddenSize

        let scale: Float = 1.0 / Foundation.sqrt(Float(hiddenSize))
        self._wx.wrappedValue = MLXRandom.uniform(
            low: -scale, high: scale, [3 * hiddenSize, inputSize]
        )
        self._wh.wrappedValue = MLXRandom.uniform(
            low: -scale, high: scale, [3 * hiddenSize, hiddenSize]
        )
        if bias {
            self.b = MLXRandom.uniform(low: -scale, high: scale, [3 * hiddenSize])
            self.bhn = MLXRandom.uniform(low: -scale, high: scale, [hiddenSize])
        } else {
            self.b = nil
            self.bhn = nil
        }
    }

    /// Single-step GRU forward.
    /// - Parameters:
    ///   - x: input [batch, inputDim]
    ///   - h: previous hidden state [batch, hiddenDim]
    /// - Returns: new hidden state [batch, hiddenDim]
    public func callAsFunction(x: MLXArray, h: MLXArray) -> MLXArray {
        // Project input
        var xProj: MLXArray
        if let b {
            xProj = addMM(b, x, wx.T)
        } else {
            xProj = matmul(x, wx.T)
        }

        // Split into reset/update gates and new content
        let xRZ = xProj[.ellipsis, .stride(to: 2 * hiddenSize)]
        let xN = xProj[.ellipsis, .stride(from: 2 * hiddenSize)]

        // Recurrent projection
        let hProj = matmul(h, wh.T)
        let hProjRZ = hProj[.ellipsis, .stride(to: 2 * hiddenSize)]
        var hProjN = hProj[.ellipsis, .stride(from: 2 * hiddenSize)]

        if let bhn {
            hProjN = hProjN + bhn
        }

        // Gates
        let rz = sigmoid(xRZ + hProjRZ)
        let parts = split(rz, parts: 2, axis: -1)
        let r = parts[0]
        let z = parts[1]

        // New gate
        let n = tanh(xN + r * hProjN)

        // Update
        let hNew = (1 - z) * n + z * h
        return hNew
    }
}
