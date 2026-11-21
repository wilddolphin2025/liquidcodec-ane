"""One-frame Liquid-Codec encoder with a fixed KV cache, for a live call.

MODIFIED WORK NOTICE: re-implementation of Liquid AI's Liquid-Codec encoder
(modeling_liquid_codec.py / modeling_lfm3.py, revision
b5abfac46ef92275f191a52c41013315f237301d) as a single-frame step. The analysis
STFT and the stride-6 front end are local. The LFM stack keeps a cache: conv
state for the five short convolutions, and keys/values for the three full-causal
attention layers. Liquid's encoder attention is not windowed, so a stateless
left-context slice does not reproduce a long clip. This step does, until the
cache length is exceeded, after which the oldest keys are replaced and the
match is no longer exact.

Each call takes 2400 samples (one 80 ms codec frame plus 480 samples / 20 ms of
lookahead, which the analysis window hangs into the next frame) and returns that
frame's 12 codes. The first call builds the same left reflect pad Liquid's
encode() uses. Later calls read 1120 samples of PCM history from the module
state. Weights are Liquid AI's, under the LFM Open License v1.0.

usage:
  python lc_step.py check MODEL_DIR
  python lc_step.py convert MODEL_DIR OUT.mlpackage [--max-frames 1024]
"""
import json, os, sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lc_ane import Layer, RMS, c1, load_sd, scales_from_stats
from lc_enc import (BINS, FRAME, HOP, N_FFT, PAD, fsq_constants, load_liquid, official_codes)

LOOKBACK = 1120                 # 2 STFT frames of real history + the 480-sample analysis pad
LOOKAHEAD = PAD                 # 480 samples, 20 ms
STEP_SAMPLES = FRAME + LOOKAHEAD
MAX_FRAMES_DEFAULT = 4096       # 327.68 s of exact full attention; older keys are dropped after that
CONV_LAYERS = (0, 1, 3, 5, 7)
ATTN_LAYERS = (2, 4, 6)


class StepEncoder(nn.Module):
    """audio (1, 1, 1, 2400) -> codes (1, 12, 1, 1). State lives on the module.

    State buffers are what Core ML exports as MLState when this module is converted
    with `states=`. They start at zero. `reset_state` starts a new stream.
    """

    def __init__(self, sd, max_frames=MAX_FRAMES_DEFAULT, scales=None):
        super().__init__()
        scales = scales or {}
        self.max_frames = int(max_frames)
        self.register_buffer("hann", torch.hann_window(N_FFT), persistent=False)
        iw, ib = sd["_frontend.in_proj.weight"], sd["_frontend.in_proj.bias"]
        self.in_proj = c1(iw[..., 0] if iw.dim() == 3 else iw, ib)
        self.fe_conv = nn.Conv1d(256, 256, 3, bias=True)
        self.fe_conv.weight.data = sd["_frontend.conv.weight"].clone()
        self.fe_conv.bias.data = sd["_frontend.conv.bias"].clone()
        self.fe_conv.requires_grad_(False)
        self.fe_out = nn.Conv1d(256, 512, 6, stride=6, bias=True)
        self.fe_out.weight.data = sd["_frontend.out.weight"].clone()
        self.fe_out.bias.data = sd["_frontend.out.bias"].clone()
        self.fe_out.requires_grad_(False)
        self.layers = nn.ModuleList([Layer(sd, i, scales, "encoder.lfm.layers.") for i in range(8)])
        self.fn = RMS(sd["encoder.lfm.embedding_norm.weight"], 1, scales.get("final", 1.0))
        self.to_q = c1(sd["to_quantizer.weight"], sd["to_quantizer.bias"])
        self.proj = c1(sd["quantizer.fsq.project_in.weight"], sd["quantizer.fsq.project_in.bias"])
        half_levels, offset, shift, half_width, basis = fsq_constants()
        for name, val in (("half_levels", half_levels), ("offset", offset), ("shift", shift),
                          ("half_width", half_width), ("basis", basis)):
            self.register_buffer(name, val, persistent=False)
        z = torch.zeros
        self.register_buffer("hist", z(1, 1, 1, LOOKBACK))
        self.register_buffer("k_cache", z(3, 2, 64, self.max_frames))
        self.register_buffer("v_cache", z(3, 2, 64, self.max_frames))
        self.register_buffer("conv_state", z(5, 512, 6))
        self.register_buffer("pos", z(1))

    def reset_state(self):
        self.hist.zero_()
        self.k_cache.zero_()
        self.v_cache.zero_()
        self.conv_state.zero_()
        self.pos.zero_()

    def _features(self, wave):
        spec = torch.stft(wave, N_FFT, hop_length=HOP, win_length=N_FFT, window=self.hann,
                          center=False, return_complex=True)
        mag = spec.abs()
        den = mag.clamp_min(1e-5)
        return torch.cat((torch.log1p(mag), spec.real / den, spec.imag / den), 1)

    def _embed(self, audio, hist, pos):
        """One codec-frame embedding (1, 512, 1). `pos` selects the first-frame reflect path."""
        wave = audio.reshape(1, -1)
        first_pad = torch.cat((wave[:, 1:LOOKAHEAD + 1].flip(-1), wave), -1)
        feat0 = self._features(first_pad)
        h0 = F.silu(self.in_proj_1d(feat0))
        hc0 = F.silu(self.fe_conv(F.pad(h0, (2, 0))))
        emb0 = self.fe_out(hc0)
        later = torch.cat((hist.reshape(1, -1), wave), -1)
        feat1 = self._features(later)
        h1 = F.silu(self.in_proj_1d(feat1))
        hc1 = F.silu(self.fe_conv(h1))
        emb1 = self.fe_out(hc1)
        first = (pos.reshape(()) < 0.5).to(emb0.dtype)
        return emb0 * first + emb1 * (1.0 - first)

    def in_proj_1d(self, feat):
        # feat (1, 1923, T) -> (1, 256, T). The stored projection is a 1x1 conv2d.
        y = self.in_proj(feat.reshape(1, feat.shape[1], 1, feat.shape[-1]))
        return y.reshape(1, y.shape[1], y.shape[-1])

    def _rope(self, t, pos):
        # t (1, heads, 64, 1), pos scalar float tensor
        inv = 1.0 / (1e6 ** (torch.arange(0, 64, 2, device=t.device, dtype=torch.float32) / 64))
        ang = pos.reshape(()).float() * inv
        cos = torch.cat((ang.cos(), ang.cos())).view(1, 1, 64, 1).to(t.dtype)
        sin = torch.cat((ang.sin(), ang.sin())).view(1, 1, 64, 1).to(t.dtype)
        rot = torch.cat((-t[:, :, 32:], t[:, :, :32]), 2)
        return t * cos + rot * sin

    def _write_cache(self, cache, layer, value, slot):
        """cache (3, 2, 64, MAX), value (1, 2, 64, 1) -> updated cache, functional."""
        slots = torch.arange(self.max_frames, device=cache.device)
        oh = (slots == slot.reshape(()).long()).to(cache.dtype).view(1, 1, 1, -1)
        cur = cache[layer:layer + 1]
        updated = cur * (1.0 - oh) + value * oh
        return torch.cat((cache[:layer], updated, cache[layer + 1:]), 0)

    def _attend(self, h, layer_i, cache_i, pos, k_cache, v_cache):
        layer = self.layers[layer_i]
        n = h.shape[-1]
        q = layer.qn(layer.q(h).reshape(1, 8, 64, n))
        k = layer.kn(layer.k(h).reshape(1, 2, 64, n))
        v = layer.v(h).reshape(1, 2, 64, n)
        q, k = self._rope(q, pos), self._rope(k, pos)
        slot = torch.remainder(pos.reshape(()), self.max_frames)
        k_cache = self._write_cache(k_cache, cache_i, k, slot)
        v_cache = self._write_cache(v_cache, cache_i, v, slot)
        kk = torch.cat([k_cache[cache_i:cache_i + 1, 0:1]] * 4 + [k_cache[cache_i:cache_i + 1, 1:2]] * 4, 1)
        vv = torch.cat([v_cache[cache_i:cache_i + 1, 0:1]] * 4 + [v_cache[cache_i:cache_i + 1, 1:2]] * 4, 1)
        score = torch.matmul(q.transpose(2, 3), kk) * (64 ** -0.5)
        idx = torch.arange(self.max_frames, device=h.device).float()
        p = pos.reshape(()).float()
        # Before the cache wraps, slots past the write are empty. After it wraps, every
        # slot holds one of the most recent max_frames keys (already RoPE'd at its position).
        valid = torch.clamp((idx <= p).to(score.dtype) + (p >= float(self.max_frames)).to(score.dtype), max=1.0)
        score = score + (1.0 - valid.view(1, 1, 1, -1)) * score.new_full((), -1e4)
        a = torch.softmax(score, -1)
        o = torch.matmul(a, vv.transpose(2, 3)).transpose(2, 3)
        o = o * torch.sigmoid(layer.gp(h)).reshape(1, 8, 1, n)
        return layer.o(o.reshape(1, 512, 1, n)), k_cache, v_cache

    def _conv(self, h, layer_i, cache_i, conv_state):
        layer = self.layers[layer_i]
        bcx = layer.inp(h)
        b, c, xx = bcx[:, :512], bcx[:, 512:1024], bcx[:, 1024:]
        u = b * xx
        prev = conv_state[cache_i]
        seq = torch.cat((prev, u.reshape(512, 1)), -1).reshape(1, 512, 1, 7)
        y = layer.dw(seq)
        new_prev = seq.reshape(512, 7)[:, 1:]
        conv_state = torch.cat((conv_state[:cache_i], new_prev.unsqueeze(0), conv_state[cache_i + 1:]), 0)
        return layer.out(c * y), conv_state

    def eager_step(self, audio, stats=None):
        """Run one frame and keep the result on this module's buffers. Eager checks only."""
        codes, hist, k_cache, v_cache, conv_state, pos = self.forward(
            audio, self.hist, self.k_cache, self.v_cache, self.conv_state, self.pos, stats)
        self.hist.copy_(hist)
        self.k_cache.copy_(k_cache)
        self.v_cache.copy_(v_cache)
        self.conv_state.copy_(conv_state)
        self.pos.copy_(pos)
        return codes

    def forward(self, audio, hist, k_cache, v_cache, conv_state, pos, stats=None):
        emb = self._embed(audio, hist, pos).reshape(1, 512, 1, 1)
        h = emb
        attn_i = 0
        conv_i = 0
        for i, layer in enumerate(self.layers):
            x = h
            op = layer.n_op(h)
            if layer.att:
                op, k_cache, v_cache = self._attend(op, i, attn_i, pos, k_cache, v_cache)
                attn_i += 1
            else:
                op, conv_state = self._conv(op, i, conv_i, conv_state)
                conv_i += 1
            if stats is not None:
                stats[f"{i}.in"] = max(stats.get(f"{i}.in", 0.0), float(h.abs().max()))
                stats[f"{i}.op"] = max(stats.get(f"{i}.op", 0.0), float(op.abs().max()))
            h = x + layer.n_pop(op)
            if stats is not None:
                stats[f"{i}.mid"] = max(stats.get(f"{i}.mid", 0.0), float(h.abs().max()))
            f = layer.n_ffn(h)
            f = layer.w2(F.silu(layer.w1(f)) * layer.w3(f))
            if stats is not None:
                stats[f"{i}.ffo"] = max(stats.get(f"{i}.ffo", 0.0), float(f.abs().max()))
            h = h + layer.n_pffn(f)
        if stats is not None:
            stats["final.in"] = max(stats.get("final.in", 0.0), float(h.abs().max()))
        h = self.fn(h)
        lat = self.proj(self.to_q(h)).reshape(1, 12, 4, 1)
        bound = torch.tanh(lat + self.shift.view(1, 1, 4, 1)) * self.half_levels.view(1, 1, 4, 1)
        bound = bound - self.offset.view(1, 1, 4, 1)
        digits = torch.round(bound) + self.half_width.view(1, 1, 4, 1)
        codes = torch.round((digits * self.basis.view(1, 1, 4, 1)).sum(2))
        hist_out = audio[..., FRAME - LOOKBACK:FRAME]
        pos_out = pos + 1
        return codes[:, :, None, :], hist_out, k_cache, v_cache, conv_state, pos_out


def stream_file(sd, audio, max_frames=MAX_FRAMES_DEFAULT, scales=None):
    """Step through a 1-D 24 kHz clip. The last frame uses end reflect as its lookahead."""
    step = StepEncoder(sd, max_frames=max_frames, scales=scales).eval()
    step.reset_state()
    n = int(audio.numel())
    frames = n // FRAME
    cols = []
    with torch.no_grad():
        for s in range(frames):
            future_lo = (s + 1) * FRAME
            future_hi = future_lo + LOOKAHEAD
            if future_hi <= n:
                lookahead = audio[future_lo:future_hi]
            else:
                tail = audio[max(0, n - (LOOKAHEAD - max(0, n - future_lo))):n]
                # reflect past the end: index n-1+k -> n-1-k, so the next samples are audio[-2], audio[-3], ...
                need = future_hi - min(future_lo, n)
                have = audio[future_lo:n] if future_lo < n else audio.new_zeros(0)
                if need:
                    src = audio[-(need + 1):-1].flip(0) if n > need else audio.flip(0)
                    # audio[-2], audio[-3], ... is flip of audio[:-1][-need:]
                    extra = audio[:-1].flip(0)[:need]
                    lookahead = torch.cat((have, extra))[:LOOKAHEAD]
                else:
                    lookahead = have
            block = torch.cat((audio[s * FRAME:(s + 1) * FRAME], lookahead))
            if block.numel() < STEP_SAMPLES:
                block = F.pad(block, (0, STEP_SAMPLES - block.numel()))
            codes = step.eager_step(block.reshape(1, 1, 1, STEP_SAMPLES))
            cols.append(codes.reshape(12).round().to(torch.int64))
    return torch.stack(cols, 1), step


def check_step(model_dir, max_frames=MAX_FRAMES_DEFAULT):
    torch.set_num_threads(int(os.environ.get("NT", "4")))
    sd = load_sd(model_dir)
    ref = load_liquid(model_dir)
    import soundfile as sf
    root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "samples")
    clips = []
    for name in ("hi_rasa_original.flac", "en_libritts_original.flac"):
        x, sr = sf.read(os.path.join(root, name), dtype="float32")
        if sr == 24000:
            clips.append((name, torch.tensor(np.asarray(x, np.float32))))
    reports = []
    stats = {}
    for name, audio in clips:
        frames = int(audio.numel()) // FRAME
        audio = audio[:frames * FRAME]
        got, step = stream_file(sd, audio, max_frames=max_frames)
        # one more pass collecting activation stats on this clip's first few seconds
        step2 = StepEncoder(sd, max_frames=max_frames).eval()
        with torch.no_grad():
            for s in range(min(frames, 16)):
                future = audio[(s + 1) * FRAME:(s + 1) * FRAME + LOOKAHEAD]
                if future.numel() < LOOKAHEAD:
                    future = torch.cat((future, audio[:-1].flip(0)[:LOOKAHEAD - future.numel()]))
                block = torch.cat((audio[s * FRAME:(s + 1) * FRAME], future[:LOOKAHEAD]))
                step2.eager_step(block.reshape(1, 1, 1, STEP_SAMPLES), stats)
        truth = official_codes(ref, audio)
        elem = got == truth[:, :got.shape[-1]]
        reports.append(dict(
            clip=name, frames=int(got.shape[-1]),
            code_match=float(elem.float().mean()),
            frame_match=float(elem.all(0).float().mean()),
            equal=bool(torch.equal(got, truth[:, :got.shape[-1]])),
        ))
    return dict(
        max_frames=max_frames,
        exact_stream_seconds=max_frames / 12.5,
        lookahead_samples=LOOKAHEAD,
        lookahead_ms=int(LOOKAHEAD / 24),
        step_samples=STEP_SAMPLES,
        lookback_samples=LOOKBACK,
        algorithmic_encoder_latency_ms=80 + 20,
        note=("Codes match Liquid's encode() for the first max_frames of a stream, "
              "including the 20 ms analysis lookahead. After that the cache keeps the "
              "most recent max_frames keys and the match is no longer exact. "
              "Left context is state, not delay."),
        clips=reports,
        scales=scales_from_stats(stats),
        coreml_prediction="not run; coremltools on Linux converts but does not predict",
    )


def convert_step(model_dir, out_path, max_frames=MAX_FRAMES_DEFAULT, scales=None):
    import coremltools as ct
    sd = load_sd(model_dir)
    m = StepEncoder(sd, max_frames=max_frames, scales=scales or {}).eval()
    m.reset_state()
    audio = torch.zeros(1, 1, 1, STEP_SAMPLES)
    hist = torch.zeros(1, 1, 1, LOOKBACK)
    k = torch.zeros(3, 2, 64, max_frames)
    v = torch.zeros(3, 2, 64, max_frames)
    conv = torch.zeros(5, 512, 6)
    pos = torch.zeros(1)
    traced = torch.jit.trace(m, (audio, hist, k, v, conv, pos))
    ml = ct.convert(
        traced,
        inputs=[
            ct.TensorType("audio", (1, 1, 1, STEP_SAMPLES), dtype=np.float16),
            ct.TensorType("hist", (1, 1, 1, LOOKBACK), dtype=np.float16),
            ct.TensorType("k_cache", (3, 2, 64, max_frames), dtype=np.float16),
            ct.TensorType("v_cache", (3, 2, 64, max_frames), dtype=np.float16),
            ct.TensorType("conv_state", (5, 512, 6), dtype=np.float16),
            ct.TensorType("pos", (1,), dtype=np.float32),
        ],
        outputs=[
            ct.TensorType("codes", dtype=np.float32),
            ct.TensorType("hist_out", dtype=np.float16),
            ct.TensorType("k_cache_out", dtype=np.float16),
            ct.TensorType("v_cache_out", dtype=np.float16),
            ct.TensorType("conv_state_out", dtype=np.float16),
            ct.TensorType("pos_out", dtype=np.float32),
        ],
        convert_to="mlprogram",
        compute_precision=ct.precision.FLOAT16,
        minimum_deployment_target=ct.target.macOS15,
        compute_units=ct.ComputeUnit.CPU_ONLY,
    )
    ml.short_description = (
        f"Liquid-Codec encoder step, fp16, cache of {max_frames} frames "
        f"({max_frames / 12.5:.2f} s exact), lookahead {LOOKAHEAD} samples. "
        f"Input audio is one 80 ms frame plus 20 ms lookahead. "
        f"Liquid AI weights, LFM Open License v1.0."
    )
    ml.author = "Liquid AI, Inc. (weights); Core ML conversion"
    ml.license = "LFM Open License v1.0"
    ml.save(out_path)
    json.dump(dict(max_frames=max_frames, step_samples=STEP_SAMPLES, lookahead_samples=LOOKAHEAD,
                   lookback_samples=LOOKBACK, state=["hist", "k_cache", "v_cache", "conv_state", "pos"],
                   scales=scales or {}), open(out_path + ".meta.json", "w"), indent=1)
    return ml


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=["check", "convert"])
    p.add_argument("model_dir")
    p.add_argument("out", nargs="?")
    p.add_argument("--max-frames", type=int, default=MAX_FRAMES_DEFAULT)
    p.add_argument("--report", default="")
    a = p.parse_args()
    if a.mode == "check":
        res = check_step(a.model_dir, a.max_frames)
        print(json.dumps(res, indent=1))
        json.dump(res, open(a.report or "step_check.json", "w"), indent=1)
    else:
        if not a.out:
            p.error("convert needs OUT.mlpackage")
        scales = None
        src = a.report or "step_check.json"
        if os.path.exists(src):
            scales = json.load(open(src)).get("scales")
        convert_step(a.model_dir, a.out, a.max_frames, scales)
        print("SAVED", a.out)


if __name__ == "__main__":
    main()
