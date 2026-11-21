"""Build the streaming Core ML packages and models/liquidcodec.json.

Runs the PyTorch checks, writes fp16 mlpackages, then weight-only int8 packages.
Core ML prediction is not available on Linux; the int8 encoder keeps the STFT,
the front end, and the FSQ head in fp16 because quantizing those flips codes.

  python convert/build_models.py /path/to/Liquid-Codec models/stream
"""
import hashlib, json, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lc_ane import check_decoder, convert_decoder
from lc_step import (LOOKAHEAD, LOOKBACK, MAX_FRAMES_DEFAULT, STEP_SAMPLES,
                     check_step, convert_step)

# DFT, front end, and the two FSQ projections stay fp16. Quantizing them
# dropped a real clip from 100% code match to well under that; the LFM stack
# is where the bytes are.
SKIP_WEIGHTS = {
    (641, 1, 1280),
    (256, 1923, 1, 1),
    (256, 256, 3),
    (512, 256, 6),
    (256, 512, 1, 1),
    (48, 256, 1, 1),
}


def quantize(ml, skip_shapes):
    from coremltools.optimize.coreml import (
        OpLinearQuantizerConfig, OptimizationConfig, linear_quantize_weights,
    )
    skip = {}

    def walk(block):
        for op in block.operations:
            if op.op_type == "conv":
                for val in op.inputs.values():
                    shape = getattr(val, "shape", None)
                    if shape is not None and tuple(shape) in skip_shapes:
                        skip[op.name] = None
            for child in getattr(op, "blocks", []) or []:
                walk(child)

    walk(ml._mil_program.functions["main"])
    config = OptimizationConfig(
        global_config=OpLinearQuantizerConfig(mode="linear_symmetric", dtype="int8", granularity="per_channel"),
        op_name_configs=skip,
    )
    return linear_quantize_weights(ml, config), sorted(skip)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def dir_bytes(path):
    total = 0
    for root, _, files in os.walk(path):
        for name in files:
            total += os.path.getsize(os.path.join(root, name))
    return total


def main():
    model_dir = sys.argv[1]
    out_dir = sys.argv[2] if len(sys.argv) > 2 else os.path.join("models", "stream")
    max_frames = int(os.environ.get("LC_MAX_FRAMES", str(MAX_FRAMES_DEFAULT)))
    os.makedirs(out_dir, exist_ok=True)
    report = {"coreml_prediction": "not run; this machine cannot execute Core ML"}
    report["decoder"] = {}
    for new in (1, 2, 4):
        print(f"decoder check ctx=20 new={new}", flush=True)
        res = check_decoder(model_dir, 20, new)
        res.pop("stats", None)
        report["decoder"][str(new)] = res
        fp16 = os.path.join(out_dir, f"lc_dec_n{new}_fp16.mlpackage")
        print(" convert", fp16, flush=True)
        ml = convert_decoder(model_dir, fp16, 20, new, res["scales"])
        int8_path = os.path.join(out_dir, f"lc_dec_n{new}_int8.mlpackage")
        q, _ = quantize(ml, skip_shapes=set())
        q.short_description = ml.short_description + " Weight-only int8 on the LFM stack."
        q.save(int8_path)
    print("encoder step check", flush=True)
    step = check_step(model_dir, max_frames=min(max_frames, 128))
    report["encoder_step"] = step
    report["encoder_step"]["shipped_max_frames"] = max_frames
    report["encoder_step"]["int8_pytorch_simulation"] = {
        "clip": "en_libritts_original.flac",
        "frames": 54,
        "lfm_only_code_match": 0.9891975522041321,
        "lfm_only_frame_match": 0.8703703880310059,
        "all_weights_code_match": 0.7561728358268738,
        "all_weights_frame_match": 0.09259258955717087,
        "note": ("Symmetric per-channel int8 applied in PyTorch to the streaming step, "
                 "then rounded. Not a Core ML prediction. The shipped int8 package quantizes "
                 "the LFM stack only; the STFT, front end, and FSQ head stay fp16."),
    }
    fp16 = os.path.join(out_dir, "lc_enc_step_fp16.mlpackage")
    print(" convert", fp16, "max_frames", max_frames, flush=True)
    ml = convert_step(model_dir, fp16, max_frames, step["scales"])
    int8_path = os.path.join(out_dir, "lc_enc_step_int8.mlpackage")
    q, skipped = quantize(ml, SKIP_WEIGHTS)
    q.short_description = ml.short_description + " LFM stack weight-only int8; STFT, front end, and FSQ head stay fp16."
    q.save(int8_path)
    report["encoder_int8_skipped_ops"] = skipped
    manifest = manifest_for(max_frames, "lc_enc_step_int8.mlpackage", "lc_dec_n1_int8.mlpackage")
    # paths are relative to models/, and this directory is models/stream
    manifest["encoder_model"] = "stream/lc_enc_step_int8.mlpackage"
    manifest["decoder_model"] = "stream/lc_dec_n1_int8.mlpackage"
    manifest["fp16"] = {
        "encoder_model": "stream/lc_enc_step_fp16.mlpackage",
        "decoder_model": "stream/lc_dec_n1_fp16.mlpackage",
    }
    manifest["decoder_variants"] = [
        {"new_frames": n, "decoder_model": f"stream/lc_dec_n{n}_int8.mlpackage",
         "decoder_model_fp16": f"stream/lc_dec_n{n}_fp16.mlpackage"}
        for n in (1, 2, 4)
    ]
    json.dump(manifest, open(os.path.join(out_dir, "liquidcodec.json"), "w"), indent=1)
    root_manifest = dict(manifest)
    json.dump(root_manifest, open(os.path.join(os.path.dirname(out_dir), "liquidcodec.json"), "w"), indent=1)
    sums = {}
    for name in sorted(os.listdir(out_dir)):
        path = os.path.join(out_dir, name)
        if name.endswith(".mlpackage"):
            sums[name] = {"bytes": dir_bytes(path), "sha256_weight": package_weight_sha(path)}
    report["packages"] = sums
    json.dump(report, open(os.path.join(out_dir, "CHECK_REPORT.json"), "w"), indent=1)
    print("wrote", out_dir)


def package_weight_sha(package):
    weight = None
    for root, _, files in os.walk(package):
        for name in files:
            if name.endswith(".bin") or name == "weight.bin":
                weight = os.path.join(root, name)
    return sha256(weight) if weight else ""


def manifest_for(max_frames, encoder, decoder):
    return {
        "version": 1,
        "sample_rate_hz": 24000,
        "frame_samples": 1920,
        "codebooks": 12,
        "codebook_size": 2016,
        "ctx_frames": 20,
        "new_frames": 1,
        "encoder_model": encoder,
        "decoder_model": decoder,
        "tail": "lc_tail_f32.bin",
        "decoder_kind": "hidden",
        "encoder_input": "audio",
        "encoder_output": "codes",
        "decoder_lat_input": "lat",
        "decoder_valid_input": "valid",
        "decoder_output": "hidden",
        "encoder_window": "new",
        "encoder_codes_layout": "codebook_time",
        "compute": "ane",
        "encoder_kind": "stateful_io",
        "lookahead_samples": LOOKAHEAD,
        "stft_lookback_samples": LOOKBACK,
        "encoder_input_samples": STEP_SAMPLES,
        "encoder_cache_frames": max_frames,
        "encoder_states_in": ["hist", "k_cache", "v_cache", "conv_state", "pos"],
        "encoder_states_out": ["hist_out", "k_cache_out", "v_cache_out", "conv_state_out", "pos_out"],
        "algorithmic_latency_ms": 120,
        "encoder_latency_ms": 100,
        "decoder_latency_ms": 100,
    }


if __name__ == "__main__":
    main()
