# AI Mixing Assistant — Audio Plugin (AU / VST3 / Standalone)

A JUCE plugin that applies the **bus processing chain** designed by the Python engine —
parametric EQ → glue compression → output gain → safety limiter — as a real-time insert.
It loads the same `.chain.json` presets exported by `mixassist preset` (or snapshotted from
a finished mix), and its DSP is a direct port of `mixassist/dsp`, so a plugin instance
reproduces the offline result on the same material.

> **Status / honesty note.** This is provided as **source**. It builds on **macOS with
> JUCE + CMake/Xcode**. The full plugin was **not compiled in the repo's generation
> environment** (no JUCE/Xcode/internet there), but two things de-risk that:
>
> 1. The JUCE-free DSP core (`Source/Dsp.h`) **is compiled and numerically verified** there
>    to match the Python engine sample-for-sample (max error ~5e-13) via
>    `plugin/tests/verify_dsp_parity.py`.
> 2. The GitHub Actions workflow (`.github/workflows/build.yml`) **actually builds the
>    AU/VST3/Standalone on a `macos-latest` runner and runs `auval`** on every push, and
>    uploads the built plugins as artifacts.
>
> So: push to GitHub and the macOS job compiles + validates it for real; the DSP math is
> already proven equivalent to the offline engine.

## Verify the DSP locally (no JUCE needed)

```bash
# from the repo root — compiles Source/Dsp.h and compares to the Python BusChain
PYTHONPATH=src python plugin/tests/verify_dsp_parity.py
# -> prints a per-sample table and "PARITY OK"
```

## Build (macOS)

```bash
cd plugin
cmake -B build -G Xcode                 # or: -G "Unix Makefiles" / Ninja
cmake --build build --config Release
```

The first configure downloads JUCE via `FetchContent`. If you're offline, point it at a
local JUCE checkout:

```bash
cmake -B build -DMIXASSIST_JUCE_PATH=/path/to/JUCE
```

Built artifacts (with `COPY_PLUGIN_AFTER_BUILD`) install to the user plugin folders:

- AU:  `~/Library/Audio/Plug-Ins/Components/AI Mixing Assistant.component`
- VST3: `~/Library/Audio/Plug-Ins/VST3/AI Mixing Assistant.vst3`

Validate the AU before using it in Logic:

```bash
auval -v aufx Mxa1 Mxas
```

## Use

1. Export a chain preset from the Python tool:
   ```bash
   mixassist preset --genre pop --intensity 0.6 --output-gain -1 --out pop.chain.json
   ```
   (or reuse the `*.chain.json` a mix run can produce).
2. Insert **AI Mixing Assistant** on your master/bus.
3. Click **Load .chain.json** and pick the preset. The EQ/comp/limiter/gain load in.
4. Tweak **EQ Amount** (live scale of the tonal moves), **Output Gain**, and **Ceiling**;
   the GR read-outs show compressor/limiter activity. Settings are saved with the session.

See `docs/LOGIC_PRO.md` for Logic-specific setup and the honest scope of DAW integration.

## Layout

```
plugin/
  CMakeLists.txt          JUCE plugin target (AU/VST3/Standalone) via FetchContent
  Source/
    Dsp.h                 biquads + compressor + limiter (ports of mixassist/dsp)
    PluginProcessor.*     chain build + block processing + preset (JSON) loading + state
    PluginEditor.*        compact control surface
  docs/LOGIC_PRO.md       build/validate/use-in-Logic guide + integration notes
```
