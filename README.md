# Liquid-Codec on the Apple Neural Engine

A Core ML conversion of [Liquid AI's Liquid-Codec](https://huggingface.co/LiquidAI/Liquid-Codec) (24 kHz speech codec, 12 codebooks at
12.5 frames/s). The original packages are the decoder only, in a fixed 40-frame window. `models/stream/` adds the encoder and decoder windows of 1, 2, and 4 new frames for a real-time call.

**Licence:** the model weights and the decoder design are Liquid AI's and are used under the **LFM Open License v1.0**
(`LICENSE-LFM-Open-v1.0.txt`). That includes its commercial-use limit (Section 5: commercial use is licensed only below USD 10M annual revenue).
What we changed is listed in `NOTICE` and at the top of `convert/lc_ane.py`.

## Speed: CPU vs Neural Engine
**About 6x faster on the Neural Engine than Liquid's original model on the CPU.** fp16 and int8 run at about the same speed; int8's gain is size
(36 MB vs 72 MB), with no audible change.

MacBook Air, Apple M3, the same 10 speech clips (60.2 s of audio) in every row. x real time = audio seconds / decode seconds, after a warm-up call.

| decoder | size | quiet machine, 1 run | busy machine, 3 runs: mean (range) |
|---|---|---|---|
| Liquid's original PyTorch model, fp32, CPU (4 threads) | 147 MB (whole codec) | 53x | 33x (27-36) |
| this conversion, fp16, CPU only | 72 MB | 93x | 54x (43-60) |
| this conversion, int8, CPU only | 36 MB | 95x | 46x (35-56) |
| this conversion, fp16, Neural Engine | 72 MB | 320x | 180x (152-204) |
| **this conversion, int8, Neural Engine (recommended)** | **36 MB** | **316x** | **218x (188-248)** |

The "busy" runs had other processes using ~3 CPU cores (alternating row order, n = 10 clips per run), so all rows are slower there. The
Neural Engine-to-PyTorch ratio is ~6x on the quiet machine and 5.5-6.7x on the busy one. All 464 operations of both packages run on the
Neural Engine (Core ML compute plan). The first call after loading compiles for the ANE (~3-5 s once per process).

## Does it match the original?
- Against Liquid's PyTorch fp32 decode of the same codes (same 10 clips): fp16 SNR 38.9 dB (sd 5.4, min 26.0); int8 27.6 dB (sd 3.2, min 20.3).
- int8 was judged by ear as unchanged from fp16 and from PyTorch fp32 ("absolutely fine, no change in the voice"; 6 clips, one listener).
- In PyTorch fp32, the re-implemented decoder matches Liquid's hidden states to 1.4e-5 (max abs). Chunked decoding matches full-sequence decoding
  to 3.2e-5. A negative control (chunks without their left context) differs by 3.2, so the check can fail.
- The Swift example matches the Python example on the same codes and package at 71.6 dB SNR.
- Lower bit widths did not hold up without retraining: plain post-training int4 (per-block) dropped to 9.8 dB and left the ANE (386 of 464 ops on
  the CPU); 4-bit and 6-bit k-means palettes gave 4.4 and 16.6 dB.

## What is in the box
- `models/lc_dec_ane_int8.mlmodelc` (recommended, 36 MB): the same package with weight-only int8 (linear, symmetric, per-channel; coremltools).
- `models/lc_dec_ane.mlmodelc` (fp16, 72 MB): the decoder's LFM stack (8 layers: 5 short convolutions, 3 sliding-window attentions) as an fp16 Core ML program.
  Fixed shape: each call takes 40 codec frames (20 of left context + 20 new = 1.6 s of new audio) and a validity mask, and returns the 512-d hidden
  states at 75 Hz. It is stateless, so chunked output equals full decoding: the receptive field is 117 frames, under the 120-frame window.
- `models/lc_tail_f32.bin`: the output projection (512 -> 1282, float32). The CPU tail turns hidden states into audio: linear -> exp/polar ->
  inverse STFT (n_fft 1280, hop 320, Hann) -> overlap-add with the window envelope. It stays on the CPU in fp32 for precision.
- `convert/lc_ane.py`: re-implementation and conversion (`check` compares against Liquid's PyTorch model; `convert` writes the .mlpackage).
  `convert/export_tail.py` writes the tail file.
- `examples/decode_wav.py`: Python, wav -> codes (Liquid's PyTorch encoder) -> ANE decode -> wav, with `--compare`.
- `examples/swift/`: a Swift decoder and, with `liquidcodec stream`, a streaming encoder plus decoder. The legacy `liquidcodec MODELS CODES.bin T OUT.wav` path still drives the 40-frame package.
- `models/liquidcodec.json` and `models/stream/`: the one-frame encoder and the 1-, 2-, and 4-frame decoders. See Streaming below. The 40-frame packages in this list are unchanged.
- `samples/`: seven real human recordings in seven languages, each as `*_original.flac` and `*_ane_int8.flac` (see Samples), plus `codes_example.npy` (the English clip's codes).

## Samples
Real human speech, one clip per language: the original recording and its Liquid-Codec codes decoded on the Neural Engine with the int8 package.
SNR is against Liquid's PyTorch fp32 decode of the same codes (one clip each).

▶ **[Listen to six of them (48 s video)](samples/samples_demo.mp4)**: each language plays the original, then the Neural Engine int8 decode (French is left out of the video; its FLACs are in `samples/`). The first 3.8 s of speech in each clip are used, with one volume level per language shared by both versions.

| language | file prefix | source | speaker | licence | SNR |
|---|---|---|---|---|---|
| English | `en_libritts` | LibriTTS-R, test.clean, utterance 4446_2275_000002_000009 | 4446 | CC-BY-4.0 | 34.6 dB |
| Hindi | `hi_rasa` | AI4Bharat Rasa (Hindi), test clip HIN_F_CONV_01840 | the dataset's female speaker | CC-BY-4.0 | 26.3 dB |
| Spanish | `es_419` | Google FLEURS es_419, test, id 1870 | not published (male) | CC-BY-4.0 | 25.1 dB |
| French | `fr_fr` | Google FLEURS fr_fr, test, id 1679 | not published (male) | CC-BY-4.0 | 29.6 dB |
| German | `de_de` | Google FLEURS de_de, test, id 1885 | not published (male) | CC-BY-4.0 | 26.7 dB |
| Japanese | `ja_jp` | Google FLEURS ja_jp, test, id 1679 | not published (male) | CC-BY-4.0 | 26.3 dB |
| Mandarin | `cmn_hans_cn` | Google FLEURS cmn_hans_cn, test, id 1687 | not published (male) | CC-BY-4.0 | 27.9 dB |

FLEURS audio is 16 kHz (kept as is in `*_original.flac`); it was resampled to 24 kHz before encoding. The Rasa clip was resampled from 48 kHz to 24 kHz.

**Attribution.** LibriTTS-R: Y. Koizumi et al., "LibriTTS-R: A Restored Multi-Speaker Text-to-Speech Corpus", 2023 (from LibriTTS / LibriSpeech, public-domain
LibriVox audiobooks), CC-BY-4.0. Rasa: AI4Bharat, IIT Madras, "Rasa: Building Expressive Speech Synthesis Systems for Indian Languages", CC-BY-4.0.
FLEURS: A. Conneau et al., "FLEURS: Few-shot Learning Evaluation of Universal Representations of Speech", Google, 2022, CC-BY-4.0.

## Changes needed for the ANE
- The quantizer's two output projections (48 -> 256 -> 512) are folded into one, and linears are written as 1x1 convolutions on (1, C, 1, T) tensors.
- Each RMSNorm gets a power-of-2 prescale (from activation statistics), so x² cannot overflow fp16. The residual stream reaches ~2,300.
- Fixed-shape stateless chunks with a validity mask replace the KV-cache streaming path: the mask zeroes conv inputs before the start and masks attention keys.
- The STFT head (complex exp, inverse FFT, overlap-add) stays outside the package, on the CPU in fp32.

## Use
```
# Python (macOS 15+, Apple silicon): pip install numpy torch soundfile soxr coremltools transformers huggingface_hub
python examples/decode_wav.py --wav samples/en_libritts_original.flac --out out.wav --compare
python examples/decode_wav.py --codes samples/codes_example.npy --out out.wav            # int8 by default; --model lc_dec_ane.mlmodelc for fp16

# Swift
swiftc -O -o decode examples/swift/LiquidANEDecoder.swift examples/swift/main.swift
python -c "import numpy as np; np.load('samples/codes_example.npy').astype('<i4').tofile('codes.bin')"
./decode models codes.bin 55 out.wav

# Re-convert from Liquid's weights (needs coremltools; check first, then convert)
python convert/lc_ane.py check  /path/to/Liquid-Codec
python convert/lc_ane.py convert /path/to/Liquid-Codec lc_dec_ane.mlpackage
xcrun coremlcompiler compile lc_dec_ane.mlpackage models/
python convert/export_tail.py /path/to/Liquid-Codec models/lc_tail_f32.bin
```

## Streaming encoder and smaller decoder windows

The 40-frame decoder above is unchanged. A call needs an encoder and a much shorter decoder step. Both are in `models/stream/`, described by `models/liquidcodec.json`.

The encoder is a one-frame Core ML step, not a stateless window. Liquid's encoder attention is causal over the whole clip (the decoder's attention is the 30-frame slide; the encoder's is not). A fixed left context therefore does not reproduce Liquid's codes: at 20 frames of context and no lookahead, code match on a 16 s clip was about 32% and the decoded audio was about 3.5 dB SNR against decoding the true codes. With 480 samples of real lookahead and 120 frames of context it was still about 90% of code values and about half of frames. Stepping one frame at a time, with the conv state and the attention keys kept, matches Liquid's `encode()` exactly in fp32.

Each encoder call takes **2400 samples**: the 80 ms codec frame plus **480 samples (20 ms)** that the analysis window hangs into the next frame. The first call uses the same left reflect pad as Liquid's `encode()`. The cache holds 4096 frames (327.68 s). Past that, the oldest keys are replaced and the match is no longer exact. The cache is not delay.

Algorithmic latency for `new_frames=1`, not including Core ML runtime or a jitter buffer:

| piece | time |
|---|---|
| collect one codec frame | 80 ms |
| analysis lookahead | 20 ms |
| receiver collect | 0 (one packet) |
| decoder overlap-add holdback | 20 ms |
| **end to end** | **120 ms** |

The WebRTC glue's own formula (sender collect + receiver collect + 20 ms tail, no encoder lookahead) is 100 ms for `new_frames=1`. The extra 20 ms is the analysis lookahead above. `new_frames` of 2 or 4 are separate decoder packages (`lc_dec_n2_*`, `lc_dec_n4_*`); the glue's formula then adds `(new_frames - 1) * 80` ms of receiver collect. The shipped manifest uses `new_frames=1`.

The decoder stays stateless. Its receptive field is 117 frames at 75 Hz, and 20 codec frames of left context are 120 frames at that rate, so a chunk matches a full decode. Measured in fp32 against Liquid's model (63 random code frames): hidden-state max abs about 3e-5, audio SNR about 100 dB, and a negative control that drops the left context differs by about 5. A KV cache would not change those samples. The packages are fp16 and weight-only int8. The int8 encoder quantizes the LFM stack only; the STFT, the front end, and the FSQ head stay fp16. A PyTorch stand-in of that split, on the English sample, kept 98.9% of code values and 87% of frames. Quantizing every weight dropped that to 75.6% of codes and 9% of frames. Those int8 figures are not Core ML predictions.

Core ML **prediction was not run** (this conversion ran on Linux). Do not treat the table above as an ANE speed or an on-device SNR. The fp32 code match was run here; the founder checks the packages on an M-series Mac with the commands below.

### What the WebRTC glue must change

`scripts/liquidcodec/fetch_models.sh` in wilddolphin2025/webrtcsays.ai already copies `models/` and reads `liquidcodec.json`. The decoder side of that manifest matches the glue as it stands: `ctx_frames=20`, `new_frames=1`, inputs `lat` and `valid`, output `hidden`, tail `lc_tail_f32.bin`. The encoder does not. `encoder_window` is `"new"`, but the call is not `new_frames * 1920` samples and it is not a single `audio` tensor.

In `modules/audio_coding/codecs/liquidcodec/liquid_coreml.mm`:

1. Read `lookahead_samples` (480), `encoder_input_samples` (2400), and `encoder_cache_frames` from the manifest. The glue's parser currently ignores unknown keys, so those fields have to be added next to `encoder_window`.
2. `Start()`: the encoder input `audio` has last dimension 2400, not `new_frames * frame_samples`. Also require inputs `hist` `[1,1,1,1120]`, `k_cache` and `v_cache` `[3,2,64,cache]`, `conv_state` `[5,512,6]`, `pos` `[1]`, all kept by the backend and zeroed in `Reset()`.
3. `EncodeWindow`: take the last 1920 samples of the new audio plus 480 samples of lookahead. `StreamEncoder::Push` must not call the model until those 480 samples have arrived (hold the frame, do not reflect-pad it). Run the prediction with the five state tensors, read `codes` (integer-valued float, shape `[1,12,1,1]`), and replace the state with `hist_out`, `k_cache_out`, `v_cache_out`, `conv_state_out`, `pos_out`.
4. Leave `DecodeWindow` as it is. For `new_frames=1` it already builds a 21-frame latent and a 126-long validity mask and keeps the last 6 hidden frames at 75 Hz.

Until that patch, pointing the glue at this manifest fails the encoder shape check on purpose. Decoder-only use of `stream/lc_dec_n1_int8.mlpackage` with `--liquid-ctx 20 --liquid-new 1` matches the code that is already there.

### Build and Mac check

```
# any machine: PyTorch reference. LIQUID_DIR is a checkout of the HF repo at b5abfac
python convert/lc_step.py check "$LIQUID_DIR"          # codes equal Liquid encode()
python convert/lc_ane.py check "$LIQUID_DIR" --ctx 20 --new 1
python examples/stream_wav.py --wav samples/hi_rasa_original.flac --out /tmp/out.wav --compare --liquid "$LIQUID_DIR"

# rebuild the packages (coremltools converts on Linux; it does not predict there)
python convert/build_models.py "$LIQUID_DIR" models/stream

# Mac, Apple silicon, macOS 15+. Prediction and the ANE are only measurable here.
python examples/stream_wav.py --coreml --models models --units cpu \
    --wav samples/hi_rasa_original.flac --out /tmp/cpu.wav --compare --liquid "$LIQUID_DIR"
python examples/stream_wav.py --coreml --models models --units ane \
    --wav samples/hi_rasa_original.flac --out /tmp/ane.wav --compare --liquid "$LIQUID_DIR"
swiftc -O -o liquidcodec examples/swift/LiquidManifest.swift examples/swift/LiquidANEDecoder.swift \
    examples/swift/LiquidANEEncoder.swift examples/swift/main.swift
./liquidcodec stream models samples/hi_rasa_original.flac /tmp/swift.wav
# optional: xcrun compile, if a caller wants .mlmodelc instead of .mlpackage
bash scripts/compile_models.sh models/stream
```

`stream_wav.py --compare` prints the code match against Liquid and the SNR of the chunked fp32 decode. `--coreml` loads the fp16 step package; on Linux that load does not predict.

## Limits
- The legacy 40-frame packages are still decoder-only. The streaming encoder is `models/stream/lc_enc_step_*.mlpackage`.
- Not bit-exact on device: the fp16 decoder was ~39 dB and int8 ~28 dB SNR against fp32 on the 40-frame package (int8 judged unchanged by ear, n=6). The new windows were checked in fp32 only. The streaming encoder matches Liquid's codes in fp32; the fp16 and int8 packages were not executed here.
- Numbers come from one machine (M3 Air) and one set of 10 clips. Other chips will differ. The first call per process includes ANE compilation.
- macOS 15 or later (the package targets macOS 15).
