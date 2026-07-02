"""Command-line interface for the AI Mixing Assistant.

Subcommands:
    genres   list supported genre targets
    demo     generate a synthetic multitrack session to experiment with
    analyze  analyze + classify the stems in a folder (no rendering)
    mix      mix a folder of stems into a master + processed stems + report + dashboard
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from mixassist import __version__
from mixassist.analysis.classify import classify_track
from mixassist.analysis.features import extract_features
from mixassist.analysis.metrics import compute_master_metrics
from mixassist.analysis.spectrum import tonal_balance
from mixassist.audio.io import load_stems, load_wav, save_wav
from mixassist.learn.corpus import learn_profile_from_references
from mixassist.learn.feedback import PreferenceModel, personalize_settings, record_feedback
from mixassist.mixing import config as config_mod
from mixassist.mixing import profiles as profiles_mod
from mixassist.mixing.engine import MixSettings, TrackOverride, mix
from mixassist.mixing.report import build_report, render_text
from mixassist.mixing.suggestions import generate_suggestions
from mixassist.mixing.targets import available_genres, get_target
from mixassist.realtime.preset import BusChainPreset
from mixassist.viz.dashboard import render_dashboard

_DEFAULT_PROFILE_DIR = "./profiles"
_DEFAULT_PREFS = "./mix_prefs.json"


def _cmd_genres(_: argparse.Namespace) -> int:
    from mixassist.mixing.targets import get_target

    print("Available genre targets:\n")
    for g in available_genres():
        t = get_target(g)
        print(f"  {g:10s} {t.bus_lufs:6.1f} LUFS   {t.notes}")
    return 0


def _cmd_demo(args: argparse.Namespace) -> int:
    from mixassist.examples.generate_stems import generate_session

    print(f"Generating synthetic session ({args.seconds:.1f}s) ...")
    stems_dir = generate_session(args.out, seconds=args.seconds)
    files = sorted(p.name for p in stems_dir.glob("*.wav"))
    print(f"Wrote {len(files)} stems to {stems_dir}:")
    for name in files:
        print(f"  - {name}")
    print(f"\nNext:\n  mixassist mix {stems_dir} --genre pop --out {Path(args.out) / 'mixed'}")
    return 0


def _cmd_analyze(args: argparse.Namespace) -> int:
    stems = load_stems(args.stems)
    results = []
    print(f"Analyzing {len(stems)} stems from {args.stems} ...\n")
    for name, buf in stems.items():
        feat = extract_features(name, buf)
        cls = classify_track(feat)
        results.append({"name": name, "classification": cls.as_dict(), "features": feat.summary()})
        print(
            f"{name:16s} -> {cls.role.upper():10s}/{cls.subtype:12s} "
            f"conf {cls.confidence:.2f} [{cls.method}]"
        )
        print(f"                 {feat.summary()}")
    if args.json:
        Path(args.json).write_text(json.dumps(results, indent=2))
        print(f"\nWrote analysis JSON to {args.json}")
    return 0


def _merge_track_flag(overrides: dict[str, TrackOverride], name: str) -> TrackOverride:
    ov = overrides.get(name)
    if ov is None:
        ov = TrackOverride()
        overrides[name] = ov
    return ov


def _parse_track_flags(args: argparse.Namespace) -> dict[str, TrackOverride]:
    """Turn --mute/--solo/--trim/--pan flags into per-track overrides."""
    overrides: dict[str, TrackOverride] = {}
    for name in args.mute or []:
        _merge_track_flag(overrides, name).mute = True
    for name in args.solo or []:
        _merge_track_flag(overrides, name).solo = True
    for spec in args.trim or []:
        name, _, val = spec.rpartition("=")
        if not name:
            raise ValueError(f"--trim expects NAME=DB, got '{spec}'")
        _merge_track_flag(overrides, name).gain_trim_db = float(val)
    for spec in args.pan or []:
        name, _, val = spec.rpartition("=")
        if not name:
            raise ValueError(f"--pan expects NAME=VALUE, got '{spec}'")
        _merge_track_flag(overrides, name).pan = float(val)
    return overrides


def _cmd_mix(args: argparse.Namespace) -> int:
    stems = load_stems(args.stems)
    print(f"Loaded {len(stems)} stems from {args.stems}")

    reference_path = args.reference
    if args.config:
        print(f"Loading control-surface config: {args.config}")
        settings, cfg_ref = config_mod.load_config(args.config)
        reference_path = reference_path or cfg_ref
    else:
        settings = MixSettings(
            genre=args.genre,
            intensity=args.intensity,
            vocal_prominence=args.vocal,
            tone=args.tone,
            target_lufs=args.target_lufs,
            peak_ceiling_db=args.peak,
            locked=frozenset(args.lock or []),
            reverb=args.reverb,
            delay=args.delay,
            drive=args.drive,
        )

    # Merge per-track CLI flags on top of whatever came from config/defaults.
    flag_overrides = _parse_track_flags(args)
    if flag_overrides:
        merged = dict(settings.track_overrides)
        for name, ov in flag_overrides.items():
            base = merged.get(name, TrackOverride())
            if ov.mute:
                base.mute = True
            if ov.solo:
                base.solo = True
            if ov.gain_trim_db:
                base.gain_trim_db = ov.gain_trim_db
            if ov.pan is not None:
                base.pan = ov.pan
            merged[name] = base
        settings.track_overrides = merged
    settings.locked = settings.locked | frozenset(args.lock or [])

    if reference_path:
        print(f"Analyzing reference track: {reference_path}")
        settings.reference = tonal_balance(load_wav(reference_path))

    if args.profile:
        target = profiles_mod.load_profile(args.profile)
        settings.target_override = target
        origin = f"learned from {target.learned_from}" if target.is_learned else "custom"
        print(f"Using target profile '{target.name}' ({origin}, {target.bus_lufs:.1f} LUFS).")

    if args.personalize:
        model = PreferenceModel.load(args.prefs)
        settings, notes = personalize_settings(settings, model)
        for line in notes:
            print(line)

    if args.save_config:
        config_mod.save_config(args.save_config, settings, reference_path)
        print(f"Saved control-surface config to {args.save_config}")

    print(f"Mixing (genre={settings.genre}) ...")
    result = mix(stems, settings)

    metrics = compute_master_metrics(result.master)
    suggestions = generate_suggestions(result, metrics)

    out_dir = Path(args.out)
    stems_out = out_dir / "stems"
    out_dir.mkdir(parents=True, exist_ok=True)

    save_wav(
        out_dir / "master.wav", result.master, bit_depth=args.bit_depth, float_output=args.float
    )
    for name, buf in result.stems.items():
        save_wav(stems_out / f"{name}.wav", buf, bit_depth=args.bit_depth, float_output=args.float)

    report = build_report(result, settings, metrics, suggestions)
    (out_dir / "mix_report.json").write_text(json.dumps(report, indent=2))
    text = render_text(result, settings, metrics, suggestions)
    (out_dir / "mix_report.txt").write_text(text + "\n")

    dashboard_note = ""
    if not args.no_dashboard:
        html = render_dashboard(result, metrics, suggestions, settings)
        (out_dir / "dashboard.html").write_text(html)
        dashboard_note = f"\nDashboard: {out_dir / 'dashboard.html'} (open in a browser)"

    print(f"\nWrote master + {len(result.stems)} processed stems to {out_dir}")
    print(f"Report: {out_dir / 'mix_report.txt'} and mix_report.json{dashboard_note}\n")
    print(text)
    return 0


def _cmd_learn(args: argparse.Namespace) -> int:
    refs = list(args.refs)
    print(f"Learning profile '{args.name}' from {len(refs)} reference mix(es) ...")
    target, summary = learn_profile_from_references(args.name, refs, base_genre=args.base)
    path = profiles_mod.save_profile(args.out, target)
    print(f"Analyzed {summary.num_references} usable reference(s).")
    print(f"  learned loudness : {target.bus_lufs:.1f} LUFS")
    print(f"  learned dynamics : {target.target_plr_db:.1f} dB PLR")
    print(f"  learned width    : {target.target_width:.2f}")
    print("  learned tonal curve (dB vs. average):")
    for band, val in target.tonal_curve_db.items():
        print(f"    {band:11s} {val:+.1f}")
    print(f"\nSaved profile to {path}")
    print(f"Use it with:  mixassist mix <stems> --profile {path}")
    return 0


def _cmd_profiles(args: argparse.Namespace) -> int:
    profs = profiles_mod.list_profiles(args.dir)
    if not profs:
        print(f"No profiles found in {args.dir}")
        return 0
    print(f"Profiles in {args.dir}:\n")
    for t in profs:
        origin = f"learned from {t.learned_from}" if t.is_learned else "hand-authored"
        print(f"  {t.name:16s} {t.bus_lufs:6.1f} LUFS   ({origin})  {t.notes}")
    return 0


def _cmd_rate(args: argparse.Namespace) -> int:
    report_path = Path(args.report)
    if report_path.is_dir():
        report_path = report_path / "mix_report.json"
    data = json.loads(report_path.read_text())
    rs = data.get("settings", {})
    metrics = data.get("metrics") or {}
    settings = MixSettings(
        genre=rs.get("genre", "default"),
        intensity=float(rs.get("intensity", 0.5)),
        vocal_prominence=float(rs.get("vocal_prominence", 0.5)),
        tone=float(rs.get("tone", 0.0)),
    )
    model = record_feedback(
        args.prefs,
        settings,
        rating=float(args.stars),
        integrated_lufs=metrics.get("integrated_lufs"),
        plr_db=metrics.get("plr_db"),
    )
    print(
        f"Recorded {args.stars}-star rating for a '{settings.genre}' mix "
        f"(intensity {settings.intensity:.2f}, vocal {settings.vocal_prominence:.2f}, "
        f"tone {settings.tone:+.2f})."
    )
    print(f"Preference model now has {len(model.examples)} example(s) at {args.prefs}.")
    return 0


def _cmd_preset(args: argparse.Namespace) -> int:
    """Export a real-time bus-chain preset from a genre or learned profile."""
    if args.profile:
        target = profiles_mod.load_profile(args.profile)
    else:
        target = get_target(args.genre)
    preset = BusChainPreset.from_target(
        target, intensity=args.intensity, output_gain_db=args.output_gain
    )
    path = preset.save(args.out)
    print(f"Exported bus-chain preset '{preset.name}' -> {path}")
    print(f"  EQ bands   : {len(preset.eq)}")
    if preset.comp:
        print(f"  glue comp  : {preset.comp.ratio:.1f}:1 @ {preset.comp.threshold_db:.0f} dB")
    print(f"  limiter    : {preset.limiter_ceiling_db:.1f} dBFS ceiling")
    print(f"  output gain: {preset.output_gain_db:+.1f} dB")
    print("\nApply it in real-time blocks with:  mixassist process <in.wav> --preset " + path)
    print("Or load the same JSON in the JUCE plugin (see plugin/docs/LOGIC_PRO.md).")
    return 0


def _cmd_process(args: argparse.Namespace) -> int:
    """Apply a saved bus-chain preset to a stereo WAV using block-based processing."""
    from mixassist.dsp.loudness import integrated_lufs, peak_dbfs

    preset = BusChainPreset.load(args.preset)
    buf = load_wav(args.input).to_stereo()
    chain = preset.build_chain(buf.sample_rate, num_channels=2)
    print(
        f"Processing {buf.duration_seconds:.1f}s @ {buf.sample_rate} Hz "
        f"in {args.block} -sample blocks with preset '{preset.name}' ..."
    )
    chain.process(buf, block_size=args.block)
    save_wav(args.out, buf, bit_depth=args.bit_depth, float_output=args.float)
    lufs = integrated_lufs(buf)
    print(f"Wrote {args.out}")
    print(f"  integrated {lufs:.1f} LUFS | peak {peak_dbfs(buf):.1f} dBFS")
    if chain.comp is not None:
        print(f"  glue comp max GR {chain.comp.max_gr_db:.1f} dB")
    print(f"  limiter max GR {chain.limiter.max_gr_db:.1f} dB")
    return 0


def _cmd_serve(args: argparse.Namespace) -> int:
    from mixassist.webapp import serve

    serve(host=args.host, port=args.port, open_browser=not args.no_open)
    return 0


def _cmd_separate(args: argparse.Namespace) -> int:
    from mixassist.separate import separate

    print(f"Separating {args.input} with Demucs (device={args.device}) ...")
    if args.repo or args.model:
        print(f"  model: {args.model or '(default)'}  repo: {args.repo or '(none)'}")
    print("  (this can take a while on CPU — that is normal)")
    res = separate(args.input, args.out, model=args.model, repo=args.repo, device=args.device)
    if not res["ok"]:
        print("error: " + res["error"], file=sys.stderr)
        return 1
    print(f"\nSeparated into {len(res['stems'])} stems in {res['stems_dir']}:")
    for s in res["stems"]:
        print(f"  - {s}.wav")
    print(f"\nNow mix them:\n  mixassist mix {res['stems_dir']} --genre pop --out ./mixed")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="mixassist", description="AI Mixing Assistant")
    p.add_argument("--version", action="version", version=f"mixassist {__version__}")
    sub = p.add_subparsers(dest="command", required=True)

    sub.add_parser("genres", help="list supported genre targets").set_defaults(func=_cmd_genres)

    d = sub.add_parser("demo", help="generate a synthetic multitrack session")
    d.add_argument("--out", default="./demo_session", help="output directory")
    d.add_argument("--seconds", type=float, default=5.0, help="length of the session")
    d.set_defaults(func=_cmd_demo)

    a = sub.add_parser("analyze", help="analyze + classify stems without mixing")
    a.add_argument("stems", help="folder containing .wav stems")
    a.add_argument("--json", help="also write analysis to this JSON file")
    a.set_defaults(func=_cmd_analyze)

    m = sub.add_parser("mix", help="mix stems into a master + report")
    m.add_argument("stems", help="folder containing .wav stems")
    m.add_argument("--out", default="./mixed", help="output directory")
    m.add_argument("--genre", default="default", choices=available_genres(), help="genre target")
    m.add_argument("--intensity", type=float, default=0.5, help="processing intensity 0..1")
    m.add_argument("--vocal", type=float, default=0.5, help="vocal prominence 0..1")
    m.add_argument("--tone", type=float, default=0.0, help="tone -1 (warm) .. +1 (clarity)")
    m.add_argument("--reverb", type=float, default=0.25, help="creative reverb amount 0..1")
    m.add_argument("--delay", type=float, default=0.12, help="creative delay amount 0..1")
    m.add_argument("--drive", type=float, default=0.15, help="bus saturation/warmth 0..1")
    m.add_argument("--reference", help="reference WAV to match tonal balance to")
    m.add_argument("--target-lufs", type=float, default=None, help="override bus LUFS target")
    m.add_argument("--peak", type=float, default=None, help="override peak ceiling (dBFS)")
    m.add_argument(
        "--lock", action="append", help="stem name to pass through unprocessed (repeatable)"
    )
    m.add_argument("--config", help="load full control-surface state from a JSON config")
    m.add_argument("--save-config", help="save the resolved control-surface state to JSON")
    m.add_argument("--mute", action="append", metavar="NAME", help="mute a stem (repeatable)")
    m.add_argument("--solo", action="append", metavar="NAME", help="solo a stem (repeatable)")
    m.add_argument(
        "--trim", action="append", metavar="NAME=DB", help="gain trim a stem, e.g. 'Vocal=1.5'"
    )
    m.add_argument(
        "--pan", action="append", metavar="NAME=VALUE", help="pan a stem -1..1, e.g. 'Gtr=0.4'"
    )
    m.add_argument("--no-dashboard", action="store_true", help="skip writing the HTML dashboard")
    m.add_argument("--profile", help="use a learned/custom target profile JSON as the target")
    m.add_argument(
        "--personalize",
        action="store_true",
        help="blend controls toward your rated preferences (see 'rate')",
    )
    m.add_argument(
        "--prefs", default=_DEFAULT_PREFS, help="preference model path (for --personalize)"
    )
    m.add_argument(
        "--bit-depth", type=int, default=24, choices=[16, 24, 32], help="output PCM bit depth"
    )
    m.add_argument("--float", action="store_true", help="write 32-bit float WAVs")
    m.set_defaults(func=_cmd_mix)

    lr = sub.add_parser("learn", help="learn a target profile from reference mixes")
    lr.add_argument("refs", nargs="+", help="reference WAV files to learn from")
    lr.add_argument("--name", required=True, help="name for the learned profile")
    lr.add_argument(
        "--base", default="default", choices=available_genres(), help="genre for level/pan balance"
    )
    lr.add_argument("--out", default=_DEFAULT_PROFILE_DIR, help="directory to save the profile in")
    lr.set_defaults(func=_cmd_learn)

    pr = sub.add_parser("profiles", help="list saved target profiles")
    pr.add_argument("--dir", default=_DEFAULT_PROFILE_DIR, help="profiles directory")
    pr.set_defaults(func=_cmd_profiles)

    rt = sub.add_parser("rate", help="rate a finished mix to teach your preferences")
    rt.add_argument("report", help="mix output dir or mix_report.json to rate")
    rt.add_argument("--stars", type=float, required=True, help="rating 1..5")
    rt.add_argument("--prefs", default=_DEFAULT_PREFS, help="preference model path")
    rt.set_defaults(func=_cmd_rate)

    ps = sub.add_parser("preset", help="export a real-time bus-chain preset (for 'process'/plugin)")
    ps.add_argument("--out", default="./bus.chain.json", help="output preset JSON path")
    ps.add_argument("--genre", default="default", choices=available_genres(), help="genre target")
    ps.add_argument("--profile", help="use a learned/custom profile JSON instead of a genre")
    ps.add_argument("--intensity", type=float, default=0.5, help="processing intensity 0..1")
    ps.add_argument("--output-gain", type=float, default=0.0, help="fixed output gain (dB)")
    ps.set_defaults(func=_cmd_preset)

    pc = sub.add_parser("process", help="apply a bus-chain preset to a stereo WAV in blocks")
    pc.add_argument("input", help="input stereo WAV")
    pc.add_argument("--preset", required=True, help="bus-chain preset JSON (see 'preset')")
    pc.add_argument("--out", default="./processed.wav", help="output WAV path")
    pc.add_argument("--block", type=int, default=512, help="processing block size (samples)")
    pc.add_argument(
        "--bit-depth", type=int, default=24, choices=[16, 24, 32], help="output PCM bit depth"
    )
    pc.add_argument("--float", action="store_true", help="write 32-bit float WAV")
    pc.set_defaults(func=_cmd_process)

    sv = sub.add_parser("serve", help="launch the local browser app (drop stems, click Mix)")
    sv.add_argument("--host", default="127.0.0.1", help="host to bind (default 127.0.0.1)")
    sv.add_argument("--port", type=int, default=8765, help="port (default 8765)")
    sv.add_argument("--no-open", action="store_true", help="do not auto-open the browser")
    sv.set_defaults(func=_cmd_serve)

    sep = sub.add_parser(
        "separate", help="split a drum loop (or song) into stems via Demucs/DrumSep"
    )
    sep.add_argument("input", help="input audio file (drum loop or song)")
    sep.add_argument("--out", default="./separated", help="output directory")
    sep.add_argument("--model", help="Demucs model name (e.g. the DrumSep model)")
    sep.add_argument("--repo", help="folder containing the DrumSep model files")
    sep.add_argument(
        "--device", default="cpu", choices=["cpu", "cuda", "mps"], help="compute device"
    )
    sep.set_defaults(func=_cmd_separate)

    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except (FileNotFoundError, NotADirectoryError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
