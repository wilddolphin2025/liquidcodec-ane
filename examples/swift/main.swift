// Two commands:
//   liquidcodec MODELS_DIR CODES.bin T OUT.wav
//       legacy: decode int32 codes [12, T] with the 40-frame package in MODELS_DIR
//   liquidcodec stream MODELS_DIR IN.wav OUT.wav
//       encode IN.wav (24 kHz mono) with the streaming Core ML encoder and decode
//       it with the manifest's decoder. Prints the algorithmic latency. Prediction
//       runs only on macOS.
//
// Build:
//   swiftc -O -o liquidcodec examples/swift/LiquidManifest.swift examples/swift/LiquidANEDecoder.swift \
//       examples/swift/LiquidANEEncoder.swift examples/swift/main.swift
import Foundation

let args = CommandLine.arguments
if args.count == 5, args[1] == "stream" {
    try stream(models: args[2], wavIn: args[3], wavOut: args[4])
} else if args.count == 5, let t = Int(args[3]) {
    try legacyDecode(models: args[1], codes: args[2], frames: t, wavOut: args[4])
} else {
    fputs("usage:\n  liquidcodec MODELS_DIR CODES.bin T OUT.wav\n  liquidcodec stream MODELS_DIR IN.wav OUT.wav\n", stderr)
    exit(2)
}

func legacyDecode(models: String, codes: String, frames: Int, wavOut: String) throws {
    let d = try Data(contentsOf: URL(fileURLWithPath: codes))
    let raw = d.withUnsafeBytes { Array($0.bindMemory(to: Int32.self)) }
    precondition(raw.count == 12 * frames, "expected 12*T codes")
    let t0 = Date()
    let dec = try LiquidANEDecoder(dir: URL(fileURLWithPath: models))
    let tLoad = Date().timeIntervalSince(t0)
    let t1 = Date()
    let y = try dec.decode(codes: raw, T: frames)
    let tDec = Date().timeIntervalSince(t1)
    try writeWav(y, to: wavOut)
    print(String(format: "wrote %@: %.2f s audio; load %.2f s; decode %.3f s (%.0fx real time)", wavOut, Double(y.count) / 24000, tLoad, tDec, Double(y.count) / 24000 / tDec))
}

func stream(models: String, wavIn: String, wavOut: String) throws {
    let dir = URL(fileURLWithPath: models)
    let manifest = try LiquidManifest.load(dir)
    let pcm = try readWav24k(wavIn)
    let frame = 1920
    let frames = pcm.count / frame
    precondition(frames >= 1, "need at least one 80 ms frame")
    let enc = try LiquidANEEncoder(dir: dir, manifest: manifest)
    let dec = try LiquidANEDecoder(dir: dir, model: manifest.decoderModel, ctx: manifest.ctx, newFrames: manifest.newFrames)
    var codes = [Int32]()
    let t0 = Date()
    for s in 0..<frames {
        let start = s * frame
        let body = Array(pcm[start..<start + frame])
        let look = lookahead(pcm, from: start + frame, n: manifest.lookahead)
        var block = body + look
        if block.count < manifest.stepSamples { block += [Float](repeating: 0, count: manifest.stepSamples - block.count) }
        codes += try enc.step(block)
    }
    let tEnc = Date().timeIntervalSince(t0)
    let t1 = Date()
    let y = try dec.decode(codes: codes, T: frames)
    let tDec = Date().timeIntervalSince(t1)
    try writeWav(y, to: wavOut)
    let latency = manifest.newFrames * 80 + (manifest.newFrames - 1) * 80 + 20 + 20
    print(String(format: "wrote %@: %d frames, %.2f s; encode %.3f s; decode %.3f s", wavOut, frames, Double(y.count) / 24000, tEnc, tDec))
    print("algorithmic latency \(latency) ms (encoder 80 ms frame + 20 ms analysis lookahead, decoder \(manifest.newFrames) new frame(s), 20 ms overlap-add holdback). Cache is \(manifest.cacheFrames) frames / \(String(format: "%.1f", Double(manifest.cacheFrames) / 12.5)) s and is not extra delay.")
}

/// Samples [from, from+n), reflecting past the end the way torch pad mode reflect does (index N-1+k -> N-1-k).
func lookahead(_ pcm: [Float], from: Int, n: Int) -> [Float] {
    var out = [Float]()
    out.reserveCapacity(n)
    for k in 0..<n {
        let i = from + k
        if i < pcm.count {
            out.append(pcm[i])
        } else {
            let back = i - pcm.count + 1          // k' = 1, 2, ... maps to N-1-k' = N-2, N-3, ...
            let j = pcm.count - 1 - back
            out.append(pcm[max(0, j)])
        }
    }
    return out
}

func writeWav(_ y: [Float], to path: String) throws {
    var w = Data()
    func u32(_ v: UInt32) { var x = v.littleEndian; w.append(Data(bytes: &x, count: 4)) }
    func u16(_ v: UInt16) { var x = v.littleEndian; w.append(Data(bytes: &x, count: 2)) }
    w.append("RIFF".data(using: .ascii)!); u32(UInt32(36 + 2 * y.count)); w.append("WAVEfmt ".data(using: .ascii)!); u32(16); u16(1); u16(1); u32(24000); u32(48000); u16(2); u16(16)
    w.append("data".data(using: .ascii)!); u32(UInt32(2 * y.count))
    for s in y { u16(UInt16(bitPattern: Int16(max(-1, min(1, s)) * 32767))) }
    try w.write(to: URL(fileURLWithPath: path))
}

func readWav24k(_ path: String) throws -> [Float] {
    let data = try Data(contentsOf: URL(fileURLWithPath: path))
    precondition(data.count > 44, "wav too small")
    // 16-bit PCM, mono or we take channel 0. Sample rate must already be 24 kHz.
    let rate = data.subdata(in: 24..<28).withUnsafeBytes { $0.load(as: UInt32.self) }
    let channels = data.subdata(in: 22..<24).withUnsafeBytes { $0.load(as: UInt16.self) }
    let bits = data.subdata(in: 34..<36).withUnsafeBytes { $0.load(as: UInt16.self) }
    precondition(rate == 24000 && bits == 16, "stream wants 24 kHz 16-bit PCM; resample first")
    var samples = [Float]()
    var i = 44
    let bps = Int(channels) * 2
    while i + bps <= data.count {
        let v = data.subdata(in: i..<(i + 2)).withUnsafeBytes { $0.load(as: Int16.self) }
        samples.append(Float(v) / 32768)
        i += bps
    }
    return samples
}
