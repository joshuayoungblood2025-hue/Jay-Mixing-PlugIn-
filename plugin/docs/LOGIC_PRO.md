# Using the AI Mixing Assistant plugin in Logic Pro

This guide covers building the AU, getting it to appear in Logic, and how the plugin fits
into a real Logic workflow — including an honest account of what DAW integration can and
cannot do.

## 1. Build & install the Audio Unit

Logic only loads **Audio Units (AU)**. From `plugin/`:

```bash
cmake -B build -G Xcode
cmake --build build --config Release
```

`COPY_PLUGIN_AFTER_BUILD` copies the component to
`~/Library/Audio/Plug-Ins/Components/`. Then validate it (Logic will refuse an AU that
fails validation):

```bash
auval -v aufx Mxa1 Mxas
```

`aufx` = audio effect, `Mxa1` = plugin code, `Mxas` = manufacturer code (see
`CMakeLists.txt`). If `auval` passes, launch Logic; it re-scans AUs on startup (or use
**Logic Pro → Settings → Plug-in Manager** to reset the cache).

## 2. Insert it on a bus or the stereo out

- Put your session's tracks into groups/busses as usual.
- On the **Stereo Out** (or a mix bus) channel strip, add an insert:
  **Audio FX → AI Mixing Assistant** (under the `MixAssist` manufacturer).
- Click **Load .chain.json** in the plugin UI and choose a preset exported by the Python
  tool (`mixassist preset ...`). The EQ, glue compression, output gain, and limiter ceiling
  populate from the preset.

## 3. Recommended workflow (offline brain + real-time hands)

The strongest way to use this system today:

1. **Export stems** from Logic (select tracks → *Bounce → Export as audio files*, or drag
   the region stems out).
2. Run the **offline engine** on the stems (`mixassist mix ... --out mixed/`). It analyzes,
   classifies, balances, and designs a master bus chain, and writes a dashboard + report.
3. Either use the produced master directly, or take the **bus-chain preset** it can export
   and load it into this plugin on your Logic master to keep mixing *inside* Logic with the
   assistant's chain in real time. The plugin's DSP matches the offline chain.

## 4. What the plugin does / does not do (honest scope)

**Does:**
- Real-time bus/master processing: EQ + glue compression + output gain + safety limiter.
- Loads presets exported by the Python tool; reproduces that chain sample-accurately.
- Saves its state (including the loaded preset) with the Logic project.

**Does not (by design / platform limits):**
- It does **not** read your Logic session, routing, or per-track plugin state. Logic exposes
  no public API for that; there is no supported way for a plugin to enumerate other tracks
  or rewrite the mixer. The per-track intelligence (classification, per-stem EQ/comp,
  panning, balancing) therefore lives in the **offline engine** that works on exported
  stems, not in this insert.
- It is **not** an auto-mixer that "listens to the mix and rebalances tracks." A single
  insert only sees the audio flowing through it.

This split — an offline "mixing brain" that designs decisions on stems, plus a real-time
plugin that applies the resulting bus chain — is the realistic architecture for Logic. A
deeper integration (e.g., an app that reads/writes sessions) would require Apple to provide
session-level APIs, or fragile scripting via MIDI/OSC and the Logic Scripter, which is out
of scope here.

## 5. Other DAWs

The same target also builds **VST3** and a **Standalone** app, so the plugin works in
Ableton Live, Cubase, Reaper, Studio One, FL Studio, etc. Only the AU is used by Logic.
