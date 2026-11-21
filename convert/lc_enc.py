"""Liquid-Codec encoder -> Core ML (Apple Neural Engine where the ops map).

MODIFIED WORK NOTICE: this file re-implements the encoder of Liquid AI's Liquid-Codec
(modeling_liquid_codec.py / modeling_lfm3.py,
https://huggingface.co/LiquidAI/Liquid-Codec, revision b5abfac46ef92275f191a52c41013315f237301d)
in the same ANE layout as convert/lc_ane.py and loads Liquid AI's weights. Changes from the
original: the analysis STFT is a stride-320 depthwise-free convolution (the windowed real DFT);
1x1 convolutions stand in for linears; the LFM stack is a fixed-shape stateless window
(left context + new frames) instead of a growing KV cache; each RMSNorm has a power-of-2
prescale so fp16 x^2 cannot overflow; finite-scalar quantization runs in this graph and
emits integer code values. The encoder's attention is causal over the whole window
(Liquid's encoder is full causal attention, not the decoder's 30-frame slide).

Liquid-Codec and its weights are licensed under the LFM Open License v1.0
(LICENSE-LFM-Open-v1.0.txt).

A call collects `new` codec frames (80 ms each) before this model can run. Left context is
past audio and is not extra delay. The analysis window hangs 480 samples (20 ms) past the
frame; with lookahead=0 the newest frame is reflect-padded the way Liquid's encode() pads
the end of a clip, so it is not identical to that frame inside a longer clip.

usage:
  python lc_enc.py check   MODEL_DIR [--ctx 20 --new 1]
  python lc_enc.py convert MODEL_DIR OUT.mlpackage [--ctx 20 --new 1] [--lookahead 0]
"""
import json, math, os, sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lc_ane import Layer, c1, load_sd, scales_from_stats, snr_db, HD

FRAME = 1920
N_FFT = 1280
HOP = 320
PAD = (N_FFT - HOP) // 2          # 480 samples, 20 ms
BINS = N_FFT // 2 + 1             # 641
STACK = 6
LEVELS = (8, 7, 6, 6)
FSQ_EPS = 1e-3


def pad_time(x, left, right, mode="constant"):
    """Pad the time axis of (N, C, 1, T). Reflect pad is not implemented for a 2-element pad on 4D."""
    y = x.reshape(x.shape[0], x.shape[1], -1)
    y = F.pad(y, (left, right), mode=mode)
    return y.reshape(x.shape[0], x.shape[1], 1, y.shape[-1])


def stft_weight():
    """Windowed real DFT, matching torch.stft(..., center=False, norm='backward', onesided)."""
    n = torch.arange(N_FFT).float()
    window = torch.hann_window(N_FFT)
    k = torch.arange(BINS).float()[:, None]
    ang = (2 * math.pi) * k * n[None] / N_FFT
    real = window[None] * torch.cos(ang)
    imag = -window[None] * torch.sin(ang)
    return torch.cat((real, imag), 0)[:, None, None, :].contiguous()


def fsq_constants():
    levels = torch.tensor(LEVELS).float()
    half_levels = (levels - 1) * (1 + FSQ_EPS) / 2
    offset = torch.where(levels % 2 == 0, torch.tensor(0.5), torch.tensor(0.0))
    shift = torch.atanh(offset / half_levels)
    half_width = torch.tensor([lv // 2 for lv in LEVELS]).float()
    basis = []
    prod = 1
    for lv in LEVELS:
        basis.append(prod)
        prod *= lv
    return half_levels, offset, shift, half_width, torch.tensor(basis).float()


class EncANE(nn.Module):
    """24 kHz mono PCM (1, 1, 1, n*1920 + lookahead) -> codes (1, 12, 1, n), integer-valued.

    n is ctx+new for a streaming window, or the whole clip for an offline check.
    Codes are the FSQ indices in [0, 2016). The last `new` frames are the ones a call emits;
    this module returns the whole window so the time axis is unambiguous.
    """

    def __init__(self, sd, n, scales=None, lookahead=0):
        super().__init__()
        scales = scales or {}
        self.lookahead = int(lookahead)
        self.register_buffer("hann", torch.hann_window(N_FFT), persistent=False)
        iw, ib = sd["_frontend.in_proj.weight"], sd["_frontend.in_proj.bias"]
        self.in_proj = c1(iw[..., 0] if iw.dim() == 3 else iw, ib)
        cw, cb = sd["_frontend.conv.weight"], sd["_frontend.conv.bias"]
        self.fe_conv = nn.Conv2d(256, 256, (1, 3), bias=True)
        self.fe_conv.weight.data = cw[:, :, None, :].clone()
        self.fe_conv.bias.data = cb.clone()
        self.fe_conv.requires_grad_(False)
        ow, ob = sd["_frontend.out.weight"], sd["_frontend.out.bias"]
        self.fe_out = nn.Conv2d(256, 512, (1, STACK), stride=(1, STACK), bias=True)
        self.fe_out.weight.data = ow[:, :, None, :].clone()
        self.fe_out.bias.data = ob.clone()
        self.fe_out.requires_grad_(False)
        self.layers = nn.ModuleList([Layer(sd, i, scales, "encoder.lfm.layers.") for i in range(8)])
        from lc_ane import RMS
        self.fn = RMS(sd["encoder.lfm.embedding_norm.weight"], 1, scales.get("final", 1.0))
        self.to_q = c1(sd["to_quantizer.weight"], sd["to_quantizer.bias"])
        self.proj = c1(sd["quantizer.fsq.project_in.weight"], sd["quantizer.fsq.project_in.bias"])
        half_levels, offset, shift, half_width, basis = fsq_constants()
        for name, val in (("half_levels", half_levels), ("offset", offset), ("shift", shift),
                          ("half_width", half_width), ("basis", basis)):
            self.register_buffer(name, val, persistent=False)
        self.set_len(n)

    def set_len(self, n):
        self.n = int(n)
        inv = 1.0 / (1e6 ** (torch.arange(0, HD, 2).float() / HD))
        fr = torch.arange(self.n).float()[:, None] * inv[None]
        emb = torch.cat((fr, fr), -1).T
        self.register_buffer("cos", emb.cos()[None, None], persistent=False)
        self.register_buffer("sin", emb.sin()[None, None], persistent=False)
        qi, ki = torch.arange(self.n)[:, None], torch.arange(self.n)[None]
        ok = ki <= qi
        self.register_buffer("base", torch.where(ok, 0.0, -1e4)[None, None], persistent=False)
        self.register_buffer("valid", torch.ones(1, 1, 1, self.n), persistent=False)

    def forward(self, audio, stats=None):
        # torch.stft matches Liquid's analysis. coremltools lowers it to a DFT conv.
        wave = pad_time(audio, PAD, PAD, mode="reflect").reshape(1, -1)
        spec_c = torch.stft(wave, N_FFT, hop_length=HOP, win_length=N_FFT, window=self.hann,
                            center=False, return_complex=True)
        mag = spec_c.abs()
        den = mag.clamp_min(1e-5)
        feat = torch.cat((torch.log1p(mag), spec_c.real / den, spec_c.imag / den), 1)
        feat = feat.reshape(1, feat.shape[1], 1, feat.shape[-1])
        h = F.silu(self.in_proj(feat))
        h = F.silu(self.fe_conv(pad_time(h, 2, 0)))
        h = self.fe_out(h)[..., : self.n]
        mask = self.base
        for i, layer in enumerate(self.layers):
            h = layer(h, self.valid, mask, self.cos, self.sin, stats, str(i))
        if stats is not None:
            stats["final.in"] = max(stats.get("final.in", 0.0), float(h.abs().max()))
        h = self.fn(h)
        lat = self.proj(self.to_q(h)).reshape(1, 12, 4, self.n)
        bound = torch.tanh(lat + self.shift.view(1, 1, 4, 1)) * self.half_levels.view(1, 1, 4, 1)
        bound = bound - self.offset.view(1, 1, 4, 1)
        digits = torch.round(bound) + self.half_width.view(1, 1, 4, 1)
        codes = torch.round((digits * self.basis.view(1, 1, 4, 1)).sum(2))
        return codes[:, :, None, :]


def load_liquid(model_dir):
    from transformers import AutoModel
    model = AutoModel.from_pretrained(model_dir, trust_remote_code=True, dtype=torch.float32).eval()
    impl = getattr(model.encoder.lfm.config, "_attn_implementation", None)
    try:
        model.encoder.lfm.set_attn_implementation("eager")
    except Exception:
        pass
    model._ane_attn = (impl, getattr(model.encoder.lfm.config, "_attn_implementation", None))
    return model


def pcm_tensor(samples):
    x = torch.as_tensor(samples, dtype=torch.float32).reshape(-1)
    return x


def official_codes(ref, audio):
    with torch.no_grad():
        y = ref.encode(audio.reshape(1, -1)).audio_codes[0]
    return y.to(torch.int64)


@torch.no_grad()
def window_audio(audio, start_frame, ctx, new, lookahead):
    """Left-align a (ctx+new) frame window on `start_frame`, then optional future samples."""
    need = (ctx + new) * FRAME + lookahead
    start = (start_frame - ctx) * FRAME
    end = (start_frame + new) * FRAME + lookahead
    w = audio.new_zeros(need)
    src_lo, src_hi = max(0, start), min(int(audio.numel()), end)
    if src_hi > src_lo:
        dst = src_lo - start
        w[dst:dst + (src_hi - src_lo)] = audio[src_lo:src_hi]
    return w


def match_rates(got, ref):
    """got, ref: (12, T) integer."""
    n = min(got.shape[-1], ref.shape[-1])
    g, r = got[:, :n], ref[:, :n]
    elem = g == r
    frame = elem.all(0)
    return dict(codes=int(elem.sum()), code_slots=int(elem.numel()),
                code_match=float(elem.float().mean()) if elem.numel() else 1.0,
                frames=int(frame.numel()), frame_match=float(frame.float().mean()) if frame.numel() else 1.0,
                frames_exact=int(frame.sum()))


def stream_codes(enc, audio, ctx, new, lookahead):
    """Glue-shaped streaming: zero left pad, reflect only inside the model, emit `new` frames at a time."""
    t = int(audio.numel()) // FRAME
    enc.set_len(ctx + new)
    cols = []
    for s in range(0, t, new):
        n_new = min(new, t - s)
        # A short tail repeats the fixed-shape call; only the real prefix is kept.
        w = window_audio(audio[:t * FRAME], s, ctx, new, lookahead)
        pred = enc(w.reshape(1, 1, 1, -1))[0, :, 0]
        take_from = ctx if n_new == new else ctx
        cols.append(pred[:, take_from:take_from + n_new].round().to(torch.int64))
    return torch.cat(cols, 1) if cols else torch.zeros(12, 0, dtype=torch.int64)


def fake_int8_per_out(weight):
    """Symmetric per-output-channel int8, the same rule coremltools linear_quantize_weights uses."""
    w = weight.detach().float()
    flat = w.reshape(w.shape[0], -1)
    scale = flat.abs().amax(1).clamp_min(1e-12) / 127.0
    q = torch.round(flat / scale[:, None]).clamp(-128, 127)
    # coremltools symmetric int8 uses [-127, 127] or [-128, 127] depending on version; both are reported.
    q = q.clamp(-127, 127)
    return (q * scale[:, None]).reshape(w.shape)


def apply_fake_int8(module, skip_names):
    for name, mod in module.named_modules():
        if not hasattr(mod, "weight") or mod.weight is None:
            continue
        if any(s in name for s in skip_names):
            continue
        if mod.weight.ndim >= 2:
            mod.weight.data.copy_(fake_int8_per_out(mod.weight.data))


def stage_report(sd, ref, audio):
    """Max abs at each encoder stage versus Liquid's modules. audio is (N,) a whole number of frames."""
    n = int(audio.numel()) // FRAME
    enc = EncANE(sd, n).eval()
    x = audio.reshape(1, -1)
    with torch.no_grad():
        feat_ref = ref.stft(x)
        wave = pad_time(x.reshape(1, 1, 1, -1), PAD, PAD, mode="reflect").reshape(1, -1)
        spec_c = torch.stft(wave, N_FFT, hop_length=HOP, win_length=N_FFT, window=enc.hann,
                            center=False, return_complex=True)
        mag = spec_c.abs()
        den = mag.clamp_min(1e-5)
        mine = torch.cat((torch.log1p(mag), spec_c.real / den, spec_c.imag / den), 1)[0]
        out = dict(stft_maxabs=float((mine - feat_ref[0]).abs().max()),
                   stft_frames=list(mine.shape))
        h_ref = ref.frontend(feat_ref)
        h = enc.fe_out(F.silu(enc.fe_conv(pad_time(F.silu(enc.in_proj(
            mine.reshape(1, mine.shape[0], 1, mine.shape[1]))), 2, 0))))
        h = h[..., :n]
        out["frontend_maxabs"] = float((h[0, :, 0] - h_ref[0]).abs().max())
        hid_ref = ref.encoder(h_ref.mT)
        # run our LFM on the official frontend so a frontend error does not hide an LFM error
        h2 = h_ref.reshape(1, 512, 1, n)
        for i, layer in enumerate(enc.layers):
            h2 = layer(h2, enc.valid, enc.base, enc.cos, enc.sin, None, str(i))
        h2 = enc.fn(h2)
        out["lfm_given_frontend_maxabs"] = float((h2[0, :, 0].T - hid_ref[0]).abs().max())
        codes = enc(audio.reshape(1, 1, 1, n * FRAME))[0, :, 0].round().to(torch.int64)
        ref_codes = official_codes(ref, audio[:n * FRAME])
        out["full_window"] = match_rates(codes, ref_codes)
        # distance of the pre-round FSQ value to the nearest integer, official vs ours
        lat = enc.proj(enc.to_q(h2)).reshape(1, 12, 4, n)
        bound = torch.tanh(lat + enc.shift.view(1, 1, 4, 1)) * enc.half_levels.view(1, 1, 4, 1)
        bound = bound - enc.offset.view(1, 1, 4, 1)
        frac = (bound - torch.round(bound)).abs()
        out["fsq_min_abs_dist_to_half"] = float((0.5 - frac).min()) if bound.numel() else None
    return out


def streaming_report(sd, ref, audio, ctx, new):
    t = int(audio.numel()) // FRAME
    ref_codes = official_codes(ref, audio[:t * FRAME])
    enc = EncANE(sd, ctx + new, lookahead=0).eval()
    got = stream_codes(enc, audio, ctx, new, 0)
    enc_l = EncANE(sd, ctx + new, lookahead=PAD).eval()
    got_l = stream_codes(enc_l, audio, ctx, new, PAD)
    # frames whose entire left context is real audio
    def sliced(codes, warmup):
        if codes.shape[-1] <= warmup:
            return codes[:, :0], ref_codes[:, :0]
        return codes[:, warmup:], ref_codes[:, warmup:codes.shape[-1]]
    warm = ctx
    a, b = sliced(got, warm)
    c, d = sliced(got_l, warm)
    # last new-frame of each chunk versus the earlier new frames (lookahead sensitivity)
    last = []
    early = []
    for s in range(warm, t, new):
        n_new = min(new, t - s)
        for k in range(n_new):
            f = s + k
            if f >= got.shape[-1]:
                break
            ok = bool(torch.equal(got[:, f], ref_codes[:, f]))
            (last if k == n_new - 1 else early).append(ok)
    return dict(
        ctx_frames=ctx, new_frames=new, clip_frames=t,
        lookahead_0_all=match_rates(got, ref_codes),
        lookahead_0_after_warmup=match_rates(a, b),
        lookahead_480_after_warmup=match_rates(c, d),
        lookahead_0_last_frame_of_chunk_match=float(np.mean(last)) if last else None,
        lookahead_0_earlier_frame_of_chunk_match=float(np.mean(early)) if early else None,
        algorithmic_latency_ms=dict(
            collect_ms=new * 80,
            stft_lookahead_if_waited_ms=20,
            left_context_ms=ctx * 80,
            left_context_is_delay=False,
            glue_formula_with_decoder_tail_ms=new * 80 + (new - 1) * 80 + 20,
        ),
    )


def load_clips():
    """24 kHz mono clips shipped with the repo, plus a longer synthetic clip. No resampling."""
    import soundfile as sf
    root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "samples")
    clips = []
    for name in ("hi_rasa_original.flac", "en_libritts_original.flac"):
        path = os.path.join(root, name)
        if not os.path.exists(path):
            continue
        x, sr = sf.read(path, dtype="float32")
        x = np.asarray(x, np.float32)
        if x.ndim > 1:
            x = x.mean(1)
        if sr != 24000:
            continue
        clips.append((name, x))
    rng = np.random.default_rng(0)
    n = FRAME * 80
    t = np.arange(n, dtype=np.float32) / 24000
    synth = (0.2 * np.sin(2 * np.pi * 180 * t) + 0.1 * np.sin(2 * np.pi * 740 * t)
             + 0.05 * rng.standard_normal(n).astype(np.float32))
    # a crude amplitude envelope so it is not a stationary tone
    env = 0.3 + 0.7 * (0.5 + 0.5 * np.sin(2 * np.pi * 3 * t))
    clips.append(("synthetic_80f", (synth * env).astype(np.float32)))
    return clips


def check_encoder(model_dir, ctx=20, new=1):
    torch.set_num_threads(int(os.environ.get("NT", "4")))
    torch.manual_seed(0)
    sd = load_sd(model_dir)
    ref = load_liquid(model_dir)
    clips = load_clips()
    # offline identity: our full-clip graph versus Liquid.encode on the Hindi clip (or the first clip)
    name, audio_np = clips[0]
    audio = pcm_tensor(audio_np)
    t = int(audio.numel()) // FRAME
    audio = audio[:t * FRAME]
    stages = stage_report(sd, ref, audio)
    streams = []
    for c in (4, 8, 12, 20, 40):
        streams.append(streaming_report(sd, ref, audio, c, new))
    # the manifest shape, on every clip, after warmup
    per_clip = []
    enc = EncANE(sd, ctx + new).eval()
    stats = {}
    for cname, arr in clips:
        a = pcm_tensor(arr)
        tt = int(a.numel()) // FRAME
        a = a[:tt * FRAME]
        got = stream_codes(enc, a, ctx, new, 0)
        truth = official_codes(ref, a)
        warm_g, warm_t = got[:, ctx:], truth[:, ctx:got.shape[-1]]
        # activation stats on this windowed run (speech, not only noise)
        with torch.no_grad():
            enc(a[: (ctx + new) * FRAME].reshape(1, 1, 1, -1), stats)
        per_clip.append(dict(clip=cname, frames=tt, after_warmup=match_rates(warm_g, warm_t),
                             all_frames=match_rates(got, truth)))
    # fp16 weight/activation simulation: the prescale is algebraically exact in fp32,
    # so this casts the fp32 codes' source by running the module in autocast-disabled
    # half weights where it is safe. Report a direct half() forward of the LFM+FSQ.
    fp16 = fp16_code_match(sd, ref, audio, ctx, new)
    fake = fake_quant_match(sd, ref, audio, ctx, new)
    res = dict(
        revision="b5abfac46ef92275f191a52c41013315f237301d",
        attn_implementation=list(ref._ane_attn),
        stage_clip=name,
        stages=stages,
        streaming_by_ctx_on_stage_clip=streams,
        manifest_shape=dict(ctx_frames=ctx, new_frames=new),
        per_clip_lookahead0=per_clip,
        fp16_sim=fp16,
        int8_weight_sim=fake,
        scales=scales_from_stats(stats),
        coreml_prediction="not run; coremltools on Linux converts but does not predict",
    )
    return res


def fp16_code_match(sd, ref, audio, ctx, new):
    """Run the streaming window with fp16 weights and fp16 math, FSQ still rounded in fp32."""
    enc = EncANE(sd, ctx + new).eval().half()
    # buffers that must stay fp32 for the trig tables get cast back; half convs are the point
    for name in ("half_levels", "offset", "shift", "half_width", "basis", "base", "valid"):
        buf = getattr(enc, name)
        if name in ("half_levels", "offset", "shift", "half_width", "basis"):
            setattr(enc, name, buf.float())
    t = int(audio.numel()) // FRAME
    truth = official_codes(ref, audio[:t * FRAME])
    got_cols = []
    enc_f = EncANE(sd, ctx + new).eval()
    enc_f.half()
    for p in ("half_levels", "offset", "shift", "half_width", "basis"):
        getattr(enc_f, p).data = getattr(enc_f, p).data.float()
    # cos/sin in half is what Core ML will store. Keep them half.
    with torch.no_grad():
        for s in range(0, t, new):
            n_new = min(new, t - s)
            w = window_audio(audio[:t * FRAME], s, ctx, new, 0).half().reshape(1, 1, 1, -1)
            pred = enc_f(w).float()[0, :, 0].round().to(torch.int64)
            got_cols.append(pred[:, ctx:ctx + n_new])
    got = torch.cat(got_cols, 1)
    warm = got[:, ctx:]
    return dict(after_warmup=match_rates(warm, truth[:, ctx:got.shape[-1]]),
                all_frames=match_rates(got, truth[:, :got.shape[-1]]),
                note="fp16 weights and activations through the stack; FSQ round is fp32 of the fp16 latent")


def fake_quant_match(sd, ref, audio, ctx, new):
    enc = EncANE(sd, ctx + new).eval()
    apply_fake_int8(enc, skip_names=("stft",))
    t = int(audio.numel()) // FRAME
    truth = official_codes(ref, audio[:t * FRAME])
    got = stream_codes(enc, audio, ctx, new, 0)
    enc2 = EncANE(sd, ctx + new).eval()
    apply_fake_int8(enc2, skip_names=())
    got2 = stream_codes(enc2, audio, ctx, new, 0)
    warm = ctx
    return dict(
        int8_except_stft_after_warmup=match_rates(got[:, warm:], truth[:, warm:got.shape[-1]]),
        int8_including_stft_after_warmup=match_rates(got2[:, warm:], truth[:, warm:got2.shape[-1]]),
        note="PyTorch stand-in for weight-only symmetric per-channel int8. Not a Core ML prediction.",
    )


def convert_encoder(model_dir, out_path, ctx=20, new=1, lookahead=0, scales=None):
    import coremltools as ct
    sd = load_sd(model_dir)
    if scales is None:
        scales = {}
    n = ctx + new
    m = EncANE(sd, n, scales=scales, lookahead=lookahead).eval()
    samples = n * FRAME + lookahead
    audio = torch.zeros(1, 1, 1, samples)
    traced = torch.jit.trace(m, (audio,))

    def keep_fp32(op):
        # The quantizer head is small and decides the integer code. Keep it in fp32.
        scope = " ".join(getattr(op, "scopes", []) or [])
        blob = f"{op.name} {getattr(op, 'op_type', '')} {scope}"
        if any(s in blob for s in ("proj", "to_q", "fsq")):
            return False
        return True

    ml = ct.convert(
        traced,
        inputs=[ct.TensorType("audio", (1, 1, 1, samples), dtype=np.float16)],
        outputs=[ct.TensorType("codes", dtype=np.float32)],
        convert_to="mlprogram",
        compute_precision=ct.transform.FP16ComputePrecision(op_selector=keep_fp32),
        minimum_deployment_target=ct.target.macOS15,
        compute_units=ct.ComputeUnit.CPU_ONLY,
    )
    ml.short_description = (
        f"Liquid-Codec encoder, fp16 stack, fp32 FSQ head, ctx={ctx} new={new} "
        f"lookahead={lookahead} samples. Codes are integer-valued float32 [0, 2016). "
        f"Liquid AI weights, LFM Open License v1.0."
    )
    ml.author = "Liquid AI, Inc. (weights); Core ML conversion"
    ml.license = "LFM Open License v1.0"
    ml.save(out_path)
    meta = dict(ctx_frames=ctx, new_frames=new, lookahead_samples=lookahead, samples=samples,
                input="audio", input_shape=[1, 1, 1, samples], output="codes",
                output_shape=[1, 12, 1, n], scales=scales)
    json.dump(meta, open(out_path + ".meta.json", "w"), indent=1)
    return ml


def main():
    import argparse
    p = argparse.ArgumentParser(description="Liquid-Codec encoder check and Core ML conversion")
    p.add_argument("mode", choices=["check", "convert"])
    p.add_argument("model_dir")
    p.add_argument("out", nargs="?")
    p.add_argument("--ctx", type=int, default=20)
    p.add_argument("--new", type=int, default=1)
    p.add_argument("--lookahead", type=int, default=0)
    p.add_argument("--report", default="")
    a = p.parse_args()
    if a.mode == "check":
        res = check_encoder(a.model_dir, a.ctx, a.new)
        text = json.dumps(res, indent=1)
        print(text)
        json.dump(res, open(a.report or "encoder_check.json", "w"), indent=1)
    else:
        if not a.out:
            p.error("convert needs OUT.mlpackage")
        scales = None
        src = a.report or "encoder_check.json"
        if os.path.exists(src):
            scales = json.load(open(src)).get("scales")
        convert_encoder(a.model_dir, a.out, a.ctx, a.new, a.lookahead, scales)
        print("SAVED", a.out)


if __name__ == "__main__":
    main()
