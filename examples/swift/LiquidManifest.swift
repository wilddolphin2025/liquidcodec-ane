// Flat reader for models/liquidcodec.json. Unknown keys are ignored.
// The manifest is part of this conversion (see NOTICE); it is not Liquid AI's file.
import Foundation

struct LiquidManifest {
    var ctx = 20
    var newFrames = 1
    var encoderModel = "stream/lc_enc_step_int8.mlpackage"
    var decoderModel = "stream/lc_dec_n1_int8.mlpackage"
    var tail = "lc_tail_f32.bin"
    var encoderInput = "audio"
    var encoderOutput = "codes"
    var lookahead = 480
    var lookback = 1120
    var stepSamples = 2400
    var cacheFrames = 4096
    var encoderKind = "stateful_io"

    static func load(_ dir: URL) throws -> LiquidManifest {
        let url = dir.appendingPathComponent("liquidcodec.json")
        let text = try String(contentsOf: url, encoding: .utf8)
        var m = LiquidManifest()
        func str(_ key: String, _ dest: inout String) {
            if let v = jsonString(text, key), !v.isEmpty { dest = v }
        }
        func num(_ key: String, _ dest: inout Int) {
            if let v = jsonInt(text, key) { dest = v }
        }
        num("ctx_frames", &m.ctx)
        num("new_frames", &m.newFrames)
        num("lookahead_samples", &m.lookahead)
        num("stft_lookback_samples", &m.lookback)
        num("encoder_input_samples", &m.stepSamples)
        num("encoder_cache_frames", &m.cacheFrames)
        str("encoder_model", &m.encoderModel)
        str("decoder_model", &m.decoderModel)
        str("tail", &m.tail)
        str("encoder_input", &m.encoderInput)
        str("encoder_output", &m.encoderOutput)
        str("encoder_kind", &m.encoderKind)
        return m
    }
}

private func jsonString(_ text: String, _ key: String) -> String? {
    guard let range = text.range(of: "\"\(key)\"") else { return nil }
    let rest = text[range.upperBound...]
    guard let colon = rest.firstIndex(of: ":"), let q1 = rest[colon...].firstIndex(of: "\"") else { return nil }
    let start = rest.index(after: q1)
    guard let q2 = rest[start...].firstIndex(of: "\"") else { return nil }
    return String(rest[start..<q2])
}

private func jsonInt(_ text: String, _ key: String) -> Int? {
    guard let range = text.range(of: "\"\(key)\"") else { return nil }
    let rest = text[range.upperBound...]
    guard let colon = rest.firstIndex(of: ":") else { return nil }
    var i = rest.index(after: colon)
    while i < rest.endIndex, rest[i] == " " || rest[i] == "\n" { i = rest.index(after: i) }
    var digits = ""
    while i < rest.endIndex, rest[i].isNumber { digits.append(rest[i]); i = rest.index(after: i) }
    return Int(digits)
}
