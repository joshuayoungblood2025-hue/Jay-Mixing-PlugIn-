# AI Mixing Assistant — Phase 1

A "virtual mixing engineer" that ingests exported multitrack **stems** and produces a
polished, release-ready **stereo mix** plus processed stems and a detailed report.

Phase 1 is the **analysis + rule-based mixing engine**. It is written entirely against the
Python **standard library** (no numpy/scipy/librosa) so it runs anywhere, and it is
structured so a faster numeric backend can be dropped in later without changing the API.

## What it does today

1. **Ingest** a folder of `.wav` stems (16/24/32-bit PCM, mono or stereo).
2. **Analyze** each stem — loudness (ITU-R BS.1770 integrated LUFS), sample peak,
   spectral features, tonal balance, crest factor.
3. **Classify** each track into a role (vocal / drums / bass / instrument / fx) using a
   hybrid of filename hints + spectral heuristics.
4. **Mix** using role-aware rules toward a genre target:
   - gain staging & balancing (with a vocal-prominence control),
   - role-based EQ (high-pass, tonal shaping, clarity vs. warmth),
   - role-based dynamics (compression scaled by a mix-intensity control),
   - panning / stereo placement,
   - optional **reference-track tonal matching**.
5. **Bus processing** — light glue compression + a safety limiter targeting a mastering
   headroom (default -1 dBTP-ish / -16 LUFS).
6. **Render** processed stems + a stereo mix, and write a JSON + human-readable **report**
   explaining every decision.

## Install

```bash
cd mixing-assistant
pip install -e . --no-build-isolation   # or: pip install -e ".[dev]" for tooling
```

No network needed — there are no runtime dependencies. (`--no-build-isolation` avoids a
build-time fetch of setuptools; you can drop it if you have internet access.) You can also
run it without installing, straight from the source tree:

```bash
PYTHONPATH=src python -m mixassist.cli genres
```

## Quick start

```bash
# 1. Generate a synthetic 5-second multitrack session to play with
python -m mixassist.cli demo --out ./demo_session

# 2. Mix it
python -m mixassist.cli mix ./demo_session/stems \
    --genre pop \
    --out ./demo_session/mixed \
    --intensity 0.6 --vocal 0.7 --tone 0.1

# 3. Read the report
cat ./demo_session/mixed/mix_report.txt
```

To match a reference track's tonal balance:

```bash
python -m mixassist.cli mix ./stems --genre rnb --reference ./ref/song.wav --out ./out
```

## Phase 2: feedback, visualization & control surface

Every `mix` run now also produces:

- **`dashboard.html`** — a self-contained visual dashboard (no internet/CDN needed):
  loudness & peak meters with the target marker, a frequency-balance graph vs. the genre
  target curve, a short-term loudness timeline, a stereo correlation gauge + per-band
  width, a color-coded **suggestions** panel, and a per-track breakdown. Open it in any
  browser.
- **ASCII meters + suggestions** inside `mix_report.txt`, and a `metrics`/`suggestions`
  block in `mix_report.json`.

The **suggestions engine** flags things like over-limiting, tonal deviations from target,
wide/mono low-end and phase problems, low dynamic range (PLR), and midrange masking of the
vocal — each with an explanation.

### Control surface (per-track tuning)

Adjust individual tracks from the CLI (all repeatable):

```bash
python -m mixassist.cli mix ./stems --genre pop \
    --solo "Lead Vocal"          # solo one or more tracks \
    --mute "Shaker"              # drop a track from the sum \
    --trim "Hats=-2"             # gain trim in dB, on top of auto-balance \
    --pan  "Guitar=0.4"          # hard override the pan (-1..1)
```

Save/restore the whole control-surface state as JSON:

```bash
python -m mixassist.cli mix ./stems --genre pop --vocal 0.7 --save-config mix.json
# ...tweak mix.json (global controls + per-track overrides + user EQ)...
python -m mixassist.cli mix ./stems --config mix.json --out ./out
```

A config captures the global sliders (intensity / vocal prominence / tone), the reference
path, and per-track overrides including custom EQ bands. Use `--no-dashboard` to skip the
HTML.

## Phase 3: learning & personalization

This phase adds **data-derived targets** and a **feedback loop** — pure-stdlib statistical
learning (medians, rating-weighted averages), not a neural net.

### Learn a target from reference mixes

Point it at a folder of finished songs you like; it measures their tonal balance, loudness,
dynamics and width and distills a **target profile** (median across the corpus):

```bash
python -m mixassist.cli learn ./refs/*.wav --name my_style --base pop --out ./profiles
python -m mixassist.cli profiles --dir ./profiles          # list saved profiles
python -m mixassist.cli mix ./stems --profile ./profiles/my_style.profile.json --out ./out
```

The learned profile becomes the mix's tonal target and loudness goal (per-track level
balance and panning are inherited from the `--base` genre, since finished stereo files have
no stems to learn balance from).

### Teach it your taste (feedback loop)

Rate finished mixes 1-5; the assistant remembers the control settings behind your
highly-rated mixes and nudges future mixes toward them:

```bash
python -m mixassist.cli mix ./stems --genre pop --intensity 0.8 --vocal 0.8 --out ./m1
python -m mixassist.cli rate ./m1 --stars 5                # learn from this mix
# ...rate a few more...
python -m mixassist.cli mix ./stems --genre pop --personalize --out ./m2
# -> controls are blended toward your liked settings, and it prints exactly what changed
```

Ratings persist in `mix_prefs.json` (override with `--prefs`). Personalization is
genre-aware, weighted by rating and recency, capped so it never lurches, and it degrades to
a no-op until you've given positive feedback.

## Phase 4: real-time processing + plugin

The offline engine is the "mixing brain." Phase 4 makes its **bus chain** run in real time
and ships it as a plugin.

### Block-accurate real-time core + preset bridge

The offline master chain (EQ -> glue compression -> output gain -> limiter) can be
snapshotted or built as a portable **`.chain.json` preset**, then applied with a stateful,
block-based processor whose output is bit-identical regardless of block size:

```bash
# export a bus-chain preset from a genre or a learned profile
python -m mixassist.cli preset --genre pop --intensity 0.6 --output-gain -1 --out pop.chain.json
python -m mixassist.cli preset --profile ./profiles/my_style.profile.json --out my.chain.json

# apply it to a stereo file in fixed-size blocks (the same math the plugin runs)
python -m mixassist.cli process master.wav --preset pop.chain.json --out out.wav --block 512
```

### Numeric backend seam

The vectorizable array math goes through `dsp/backend.py`, which auto-selects a **numpy**
backend when numpy is importable and otherwise falls back to pure Python — so the codebase
runs everywhere now and speeds up for free where numpy exists. No other code changes.

### The plugin (AU / VST3 / Standalone)

`plugin/` is a complete **JUCE** project that applies the same `.chain.json` chain as a
real-time insert (its DSP is a direct port of `mixassist/dsp`). Build it on macOS:

```bash
cd plugin && cmake -B build -G Xcode && cmake --build build --config Release
```

See `plugin/docs/LOGIC_PRO.md` for building, `auval`, inserting it in Logic, and an honest
account of DAW-integration scope (it's a bus/master insert; per-track intelligence stays in
the offline engine because Logic exposes no session/mixer API).

> The plugin is provided as **source, built on a Mac** — the full AU/VST3 isn't compiled in
> this repo's generation environment (no C++/JUCE toolchain there). But its DSP core is
> compile-verified against the Python engine (parity ~5e-13), and a GitHub Actions
> `macos-latest` job actually builds the AU/VST3/Standalone and runs `auval` on every push
> (`.github/workflows/build.yml`). The Python real-time core is fully tested here.

## Drum separation (Demucs / DrumSep)

`mixassist separate` splits a drum loop (or full song) into per-instrument stems using an
external neural model, then drops them into a folder ready for `mixassist mix` / the app:

```bash
# full-song split (drums/bass/vocals/other) — needs `pip install demucs`
mixassist separate song.wav --out ./separated

# drum-piece split (kick/snare/cymbals/toms) — needs the DrumSep model via --repo/--model
mixassist separate drumloop.wav --repo ./drumsep_model --model modelo_final --out ./separated
```

It shells out to Demucs (which must be installed on the machine — it pulls in PyTorch) and
renames the outputs (the DrumSep labels are Spanish: bombo→Kick, redoblante→Snare,
platillos→Cymbals, toms→Toms). No model ships with this repo; professional separation is a
trained neural network, so it runs on your machine, not in this package.

## Mix-prep mode (balance/EQ/comp/pan only — master in your DAW)

Set `master=False` (CLI `--no-master`, and the **default in the web app**) to get a clean,
**un-mastered** mix for finishing in your DAW. It:

- **Gain-stages every track to the industry-standard ~-18 dBFS RMS** (the "-18 dBFS = 0 VU"
  sweet spot), with peaks kept under -6 dBFS, **before any EQ/compression** — so the
  processing hits at correct levels.
- **Classifies, balances, EQs, compresses, and pans** each track (intentional panning:
  kick/snare/bass/lead-vocal centered, everything else placed like an engineer would).
- **Skips** bus glue, loudness maximizing, the limiter, and creative FX.
- Leaves the bus peaking ~ -6 dBFS — headroom to **master and add reverb/FX yourself**.

The web app returns the **processed stems as a downloadable `.zip`** (re-import them into
Logic to keep mixing) plus an in-page **play button** to preview the balanced mix. One-click
**Auto Mix** uses genre-appropriate settings so you don't touch a fader.

## Creative FX (reverb / delay / saturation)

Beyond the technical mix, the engine adds tasteful **creative processing** via role-aware
send effects and bus drive:

- **Reverb** — a Freeverb-style algorithm on a return bus (vocals get the most, drums a
  short room, bass none).
- **Delay** — a tempo-friendly stereo/ping-pong delay send (mostly vocals + lead).
- **Saturation** — gentle tanh drive on the bus for warmth and glue.
- **Kick → Bass side-chain** — `--sidechain 0..1` ducks the bass whenever the kick hits, so
  the low end stays clean and the kick punches through (best paired with an isolated kick
  from `separate`).

Controls (`0..1`) are on the CLI (`--reverb`, `--delay`, `--drive`), the web app sliders,
and the mix config. They default to subtle amounts in the CLI/app and to `0` in the library
API. These are solid algorithmic effects — musical and real, though not a clone of a
boutique commercial reverb. They currently live in the engine/app; porting them into the
C++ plugin is a follow-up.

## Architecture

```
src/mixassist/
  audio/      AudioBuffer + WAV I/O + resampling
  dsp/        biquad filters, FFT, loudness (BS.1770), EQ, compressor, gain/pan, backend
  analysis/   spectral features, tonal balance, track classification, master metrics
  mixing/     genre targets, reference match, engine, suggestions, config, profiles, report
  learn/      corpus learning (targets from references) + preference/feedback model
  realtime/   block-based bus chain + portable .chain.json presets (mirrors the plugin)
  viz/        inline-SVG primitives + self-contained HTML dashboard
  cli.py      command line entry point
plugin/       JUCE AU/VST3/Standalone plugin source (build on macOS) + Logic Pro guide
```

## Roadmap

- Phase 1 (done): analysis + rule-based mixing engine.
- Phase 2 (done): control-surface tuning + visual dashboard, metrics & suggestions.
- Phase 3 (done): learned targets from a reference corpus + rating-based personalization.
- Phase 4 (done): block-accurate real-time core, preset bridge, numpy-ready backend seam,
  and a JUCE AU/VST3 plugin (build on macOS).
- Beyond: a compiled/validated plugin binary, an active numpy/native DSP backend, and a
  learned neural tonal model once a real numeric/ML stack is available.

See the design discussion for the honest scope notes on Logic Pro integration.
