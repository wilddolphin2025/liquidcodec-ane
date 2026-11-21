// Liquid-Codec decoder on the Apple Neural Engine (Core ML, fp16 LFM stack) + CPU fp32 tail. Codes [12, T] (12.5 frames/s) -> 24 kHz audio.
// New code (Swift port of convert/lc_ane.py chunks + Tail), using Liquid AI's converted weights (LFM Open License v1.0; see NOTICE).
import Accelerate
import CoreML
import Foundation

public final class LiquidANEDecoder: @unchecked Sendable {
    let dec: MLModel
    let W: [Float], b: [Float]                      // lin [1282, 512] row-major, bias [1282]
    let basis: [Float]                              // [1280, 1282]: windowed inverse real DFT for [Re(641); Im(641)]
    let env: [Float]                                // window^2 for overlap-add
    /// Codec frames of left context and of new audio. The legacy package is 20 + 20.
    public let ctx: Int
    public let newFrames: Int
    public var fr: Int { ctx + newFrames }
    public var t75: Int { fr * 6 }
    static let NB = 641, NF = 1280, HOP = 320, LV = [8, 7, 6, 6]

    public init(dir: URL, model: String = "lc_dec_ane_int8.mlmodelc", ctx: Int = 20, newFrames: Int = 20, units: MLComputeUnits = .cpuAndNeuralEngine) throws {
        self.ctx = ctx
        self.newFrames = newFrames
        let c = MLModelConfiguration(); c.computeUnits = units
        dec = try MLModel(contentsOf: dir.appendingPathComponent(model), configuration: c)
        let d = try Data(contentsOf: dir.appendingPathComponent("lc_tail_f32.bin"))
        let all = d.withUnsafeBytes { Array($0.bindMemory(to: Float.self)) }
        W = Array(all[0..<(1282 * 512)]); b = Array(all[(1282 * 512)...])
        var win = [Float](repeating: 0, count: Self.NF)
        for t in 0..<Self.NF { win[t] = Float(0.5 - 0.5 * cos(2 * Double.pi * Double(t) / Double(Self.NF))) }   // torch.hann_window (periodic)
        var B = [Float](repeating: 0, count: Self.NF * 2 * Self.NB)
        for t in 0..<Self.NF {
            for k in 0..<Self.NB {
                let ck: Double = (k == 0 || k == Self.NB - 1) ? 1 : 2, ang = 2 * Double.pi * Double(k * t) / Double(Self.NF)
                B[t * 2 * Self.NB + k] = Float(ck * cos(ang) / Double(Self.NF)) * win[t]
                B[t * 2 * Self.NB + Self.NB + k] = (k == 0 || k == Self.NB - 1) ? 0 : Float(-ck * sin(ang) / Double(Self.NF)) * win[t]
            }
        }
        basis = B; env = win.map { $0 * $0 }
    }

    /// codes [12, T] (row q = codebook q, values 0..<2016) -> Liquid's 48-d FSQ latent [48, T]
    public static func latent(codes: [Int32], T: Int) -> [Float] {
        let basisD: [Int32] = [1, 8, 56, 336]; var lat = [Float](repeating: 0, count: 48 * T)
        for t in 0..<T { for q in 0..<12 { let c = codes[q * T + t]
            for (j, lv) in LV.enumerated() { let d = (c / basisD[j]) % Int32(lv), hw = Float(lv / 2); lat[(q * 4 + j) * T + t] = (Float(d) - hw) / hw } } }
        return lat
    }

    /// whole decode: codes [12, T] -> audio (T * 1920 samples)
    public func decode(codes: [Int32], T: Int) throws -> [Float] {
        var out: [Float] = []; try audio(Self.latent(codes: codes, T: T), T: T, read: Self.read) { out += $0 }; return out
    }

    /// an output as a contiguous [Float], honouring strides (Core ML may pad them) and fp16 storage
    static func read(_ m: MLMultiArray) -> [Float] {
        let sh = m.shape.map { $0.intValue }, st = m.strides.map { $0.intValue }, n = sh.reduce(1, *); var out = [Float](repeating: 0, count: n)
        let inner = sh.last!, outer = n / inner, cap = zip(sh, st).map { ($0 - 1) * $1 }.reduce(0, +) + 1
        let h = m.dataType == .float16 ? m.dataPointer.bindMemory(to: Float16.self, capacity: cap) : nil
        let g = m.dataType == .float16 ? nil : m.dataPointer.bindMemory(to: Float.self, capacity: cap)
        for o in 0..<outer {
            var off = 0, r = o
            for d in stride(from: sh.count - 2, through: 0, by: -1) { off += (r % sh[d]) * st[d]; r /= sh[d] }
            for k in 0..<inner { let idx = off + k * st.last!; out[o * inner + k] = h != nil ? Float(h![idx]) : g![idx] }
        }
        return out
    }

    /// latent [48, T] -> audio (T * 1920 samples), streamed per call of the Liquid stack (20 codec frames = 1.6 s); onChunk gets each
    /// finished stretch of audio. Same output as LiquidCodecModel.decode (non-streaming), see g2410/lc_ane.py.
    public func audio(_ lat: [Float], T: Int, read: (MLMultiArray) -> [Float], onChunk: ([Float]) -> Void) throws {
        let n75 = T * 6, total = (n75 - 1) * Self.HOP + Self.NF
        var acc = [Float](repeating: 0, count: total), en = [Float](repeating: 0, count: total); var emitted = 480
        let la = try MLMultiArray(shape: [1, 48, 1, NSNumber(value: fr)], dataType: .float16), va = try MLMultiArray(shape: [1, 1, 1, NSNumber(value: t75)], dataType: .float16)
        let lp = la.dataPointer.bindMemory(to: Float16.self, capacity: 48 * fr), vp = va.dataPointer.bindMemory(to: Float16.self, capacity: t75)
        var s = 0
        while s < T {
            for c in 0..<48 { for f in 0..<fr { let g = s - ctx + f; lp[c * fr + f] = (g >= 0 && g < T) ? Float16(lat[c * T + g]) : 0 } }
            for f in 0..<t75 { let g = s - ctx + f / 6; vp[f] = (g >= 0 && g < T) ? 1 : 0 }
            let hid = try autoreleasepool { read(try dec.prediction(from: MLDictionaryFeatureProvider(dictionary: ["lat": la, "valid": va])).featureValue(for: "hidden")!.multiArrayValue!) }
            let keep = min(newFrames, T - s) * 6, c0 = ctx * 6
            var hk = [Float](repeating: 0, count: 512 * keep)
            for c in 0..<512 { for j in 0..<keep { hk[c * keep + j] = hid[c * t75 + c0 + j] } }
            // y [1282, keep] = W hk + b
            var y = [Float](repeating: 0, count: 1282 * keep)
            for r in 0..<1282 { for j in 0..<keep { y[r * keep + j] = b[r] } }
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 1282, Int32(keep), 512, 1, W, 512, hk, Int32(keep), 1, &y, Int32(keep))
            var sp = [Float](repeating: 0, count: 2 * Self.NB * keep)          // [Re(641); Im(641)] x keep
            for k in 0..<Self.NB { for j in 0..<keep {
                let m = exp(y[k * keep + j]), a = y[(Self.NB + k) * keep + j]
                sp[k * keep + j] = m * cos(a); sp[(Self.NB + k) * keep + j] = m * sin(a)
            } }
            var fr = [Float](repeating: 0, count: Self.NF * keep)              // windowed frames [1280, keep]
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, Int32(Self.NF), Int32(keep), Int32(2 * Self.NB), 1, basis, Int32(2 * Self.NB), sp, Int32(keep), 0, &fr, Int32(keep))
            for j in 0..<keep {
                let f0 = (s * 6 + j) * Self.HOP
                for t in 0..<Self.NF { acc[f0 + t] += fr[t * keep + j]; en[f0 + t] += env[t] }
            }
            s += newFrames
            let ready = s >= T ? total - 480 : s * 6 * Self.HOP              // samples before the next frame's start are final
            if ready > emitted { onChunk((emitted..<ready).map { acc[$0] / en[$0] }); emitted = ready }
        }
    }
}
