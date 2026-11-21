// Streaming Liquid-Codec encoder. One call is one codec frame: 1920 samples plus
// 480 samples of lookahead (20 ms). The module returns 12 int codes and the next
// cache tensors (hist, k, v, conv state, position). Codes match Liquid's PyTorch
// encode() in fp32 for the first `cacheFrames` frames; see convert/lc_step.py.
// Weights are Liquid AI's (LFM Open License v1.0).
import CoreML
import Foundation

public final class LiquidANEEncoder: @unchecked Sendable {
    let enc: MLModel
    let manifest: LiquidManifest
    var hist: MLMultiArray
    var kCache: MLMultiArray
    var vCache: MLMultiArray
    var conv: MLMultiArray
    var pos: MLMultiArray

    public init(dir: URL, manifest: LiquidManifest, units: MLComputeUnits = .cpuAndNeuralEngine) throws {
        self.manifest = manifest
        let c = MLModelConfiguration(); c.computeUnits = units
        enc = try MLModel(contentsOf: dir.appendingPathComponent(manifest.encoderModel), configuration: c)
        hist = try MLMultiArray(shape: [1, 1, 1, NSNumber(value: manifest.lookback)], dataType: .float16)
        kCache = try MLMultiArray(shape: [3, 2, 64, NSNumber(value: manifest.cacheFrames)], dataType: .float16)
        vCache = try MLMultiArray(shape: [3, 2, 64, NSNumber(value: manifest.cacheFrames)], dataType: .float16)
        conv = try MLMultiArray(shape: [5, 512, 6], dataType: .float16)
        pos = try MLMultiArray(shape: [1], dataType: .float32)
        zero(hist); zero(kCache); zero(vCache); zero(conv); zero(pos)
    }

    public func reset() { zero(hist); zero(kCache); zero(vCache); zero(conv); zero(pos) }

    /// `samples` length is encoder_input_samples (2400): the codec frame, then 20 ms of lookahead, in [-1, 1].
    public func step(_ samples: [Float]) throws -> [Int32] {
        precondition(samples.count == manifest.stepSamples, "encoder step wants \(manifest.stepSamples) samples")
        let audio = try MLMultiArray(shape: [1, 1, 1, NSNumber(value: manifest.stepSamples)], dataType: .float16)
        let p = audio.dataPointer.bindMemory(to: Float16.self, capacity: samples.count)
        for i in 0..<samples.count { p[i] = Float16(samples[i]) }
        let provider = try MLDictionaryFeatureProvider(dictionary: [
            manifest.encoderInput: audio,
            "hist": hist, "k_cache": kCache, "v_cache": vCache, "conv_state": conv, "pos": pos,
        ])
        let out = try enc.prediction(from: provider)
        guard let codes = out.featureValue(for: manifest.encoderOutput)?.multiArrayValue else {
            throw NSError(domain: "LiquidANE", code: 1, userInfo: [NSLocalizedDescriptionKey: "encoder output missing"])
        }
        if let h = out.featureValue(for: "hist_out")?.multiArrayValue { copy(h, into: &hist) }
        if let h = out.featureValue(for: "k_cache_out")?.multiArrayValue { copy(h, into: &kCache) }
        if let h = out.featureValue(for: "v_cache_out")?.multiArrayValue { copy(h, into: &vCache) }
        if let h = out.featureValue(for: "conv_state_out")?.multiArrayValue { copy(h, into: &conv) }
        if let h = out.featureValue(for: "pos_out")?.multiArrayValue { copy(h, into: &pos) }
        let flat = LiquidANEDecoder.read(codes)
        return flat.map { Int32(lrintf($0)) }
    }
}

private func zero(_ a: MLMultiArray) {
    memset(a.dataPointer, 0, a.count * a.elementBytes)
}

private func copy(_ src: MLMultiArray, into dst: inout MLMultiArray) {
    if src.count == dst.count && src.dataType == dst.dataType {
        memcpy(dst.dataPointer, src.dataPointer, src.count * src.elementBytes)
    } else {
        dst = src
    }
}

private extension MLMultiArray {
    var elementBytes: Int {
        switch dataType {
        case .float16: return 2
        case .float32, .int32: return 4
        case .float64: return 8
        default: return 4
        }
    }
}
