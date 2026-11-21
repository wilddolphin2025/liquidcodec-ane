"""Stream a 24 kHz wav through the Liquid-Codec encoder step and the chunked decoder.

On any machine this runs the PyTorch reimplementation and, with --compare, checks
codes against Liquid's model. Core ML prediction is macOS-only; pass --coreml there.

  python examples/stream_wav.py --wav samples/hi_rasa_original.flac --out /tmp/out.wav --compare
  python examples/stream_wav.py --wav in.wav --out out.wav --coreml --models models --units ane
"""
import argparse, os, sys
import numpy as np
import soundfile as sf
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "convert"))
from lc_ane import Tail, check_decoder, latent_from_codes, load_sd, run_chunked, DecANE, snr_db
from lc_enc import load_liquid, official_codes
from lc_step import FRAME, LOOKAHEAD, stream_file

REV = "b5abfac46ef92275f191a52c41013315f237301d"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wav", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--compare", action="store_true")
    ap.add_argument("--coreml", action="store_true")
    ap.add_argument("--models", default=os.path.join(HERE, "..", "models"))
    ap.add_argument("--liquid", default=os.environ.get("LIQUID_DIR", ""))
    ap.add_argument("--ctx", type=int, default=20)
    ap.add_argument("--new", type=int, default=1)
    ap.add_argument("--units", default="ane", choices=["ane", "cpu"])
    a = ap.parse_args()
    audio, sr = sf.read(a.wav, dtype="float32")
    audio = np.asarray(audio, np.float32)
    if audio.ndim > 1:
        audio = audio.mean(1)
    if sr != 24000:
        raise SystemExit(f"{a.wav} is {sr} Hz; resample to 24000 first")
    frames = len(audio) // FRAME
    audio = audio[:frames * FRAME]
    liquid = a.liquid or os.path.expanduser("~/Liquid-Codec")
    if not os.path.isdir(liquid):
        # the pinned snapshot, if the environment already downloaded it
        cand = "/tmp/liquid"
        liquid = cand if os.path.isdir(cand) else liquid
    if not os.path.isdir(liquid):
        from huggingface_hub import snapshot_download
        liquid = snapshot_download("LiquidAI/Liquid-Codec", revision=REV)
    sd = load_sd(liquid)
    got, _ = stream_file(sd, torch.tensor(audio))
    codes = got.numpy().astype(np.int32)
    print(f"pytorch step encoder: {codes.shape[1]} frames")
    if a.compare:
        ref = load_liquid(liquid)
        truth = official_codes(ref, torch.tensor(audio)).numpy()
        elem = codes == truth
        print(f"vs Liquid encode: code match {elem.mean():.6f}  frame match {elem.all(0).mean():.6f}  equal {bool(elem.all())}")
        with torch.no_grad():
            ref_audio = ref.decode(torch.from_numpy(truth)[None]).audio_values[0].numpy()
        lat = latent_from_codes(codes)
        dec = DecANE(sd, n=(a.ctx + a.new) * 6).eval()
        with torch.no_grad():
            fn = lambda w, v: dec(torch.from_numpy(w), torch.from_numpy(v))[0, :, 0].numpy()
            hid = run_chunked(fn, lat, a.ctx, a.new)
        y = Tail(sd).audio(hid)
        n = min(len(y), len(ref_audio))
        print(f"chunked fp32 decode vs Liquid decode: SNR {snr_db(ref_audio[:n], y[:n]):.2f} dB")
        sf.write(a.out, np.clip(y, -1, 1), 24000, subtype="PCM_16")
        print("wrote", a.out)
    if a.coreml:
        import coremltools as ct
        man_path = os.path.join(a.models, "liquidcodec.json")
        print("Core ML packages are loaded from", a.models)
        print("Prediction of an mlprogram requires macOS. This process will try and report the error if it cannot.")
        enc_name = "stream/lc_enc_step_fp16.mlpackage"
        if os.path.isfile(man_path):
            import json
            man = json.load(open(man_path))
            enc_name = man.get("fp16", {}).get("encoder_model", man["encoder_model"])
        path = os.path.join(a.models, enc_name) if not os.path.isabs(enc_name) else enc_name
        if not os.path.exists(path):
            # manifest paths are relative to models/
            path = os.path.join(a.models, os.path.basename(enc_name))
        units = ct.ComputeUnit.CPU_AND_NE if a.units == "ane" else ct.ComputeUnit.CPU_ONLY
        try:
            model = ct.models.MLModel(path, compute_units=units)
            print("spec loaded", path)
            print("inputs:", [f.name for f in model.get_spec().description.input])
            cache = int(model.get_spec().description.input[2].type.multiArrayType.shape[-1])
            feed = {
                "audio": np.zeros((1, 1, 1, 2400), np.float16),
                "hist": np.zeros((1, 1, 1, 1120), np.float16),
                "k_cache": np.zeros((3, 2, 64, cache), np.float16),
                "v_cache": np.zeros((3, 2, 64, cache), np.float16),
                "conv_state": np.zeros((5, 512, 6), np.float16),
                "pos": np.zeros((1,), np.float32),
            }
            pred = model.predict(feed)
            print("predicted one silent frame, codes", np.asarray(pred["codes"]).reshape(-1).astype(np.int32).tolist())
        except Exception as e:
            print(f"Core ML prediction did not run: {type(e).__name__}: {e}")
            print("On a Mac: python examples/stream_wav.py --coreml --models models --units ane "
                  "--wav <24k.wav> --out out.wav --compare --liquid \"$LIQUID_DIR\"")


if __name__ == "__main__":
    main()
