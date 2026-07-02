"""A tiny local web app for the mixing engine — no terminal knowledge required to use.

Run ``mixassist serve`` and a page opens in the browser where you drop your stem WAVs,
pick a genre and a few sliders, click **Mix**, then view the dashboard and download the
mastered file. Built entirely on the standard library (``http.server``) so there is nothing
extra to install.
"""

from __future__ import annotations

import contextlib
import json
import shutil
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

from mixassist.analysis.metrics import compute_master_metrics
from mixassist.audio.io import load_stems, save_wav
from mixassist.mixing.engine import MixSettings, mix
from mixassist.mixing.report import build_report, render_text
from mixassist.mixing.suggestions import generate_suggestions
from mixassist.mixing.targets import available_genres
from mixassist.viz.dashboard import render_dashboard

_CONTENT_TYPES = {
    ".html": "text/html; charset=utf-8",
    ".wav": "audio/wav",
    ".txt": "text/plain; charset=utf-8",
    ".json": "application/json",
}


# --------------------------------------------------------------------- multipart parsing


def _hval(header: str, key: str) -> str | None:
    idx = header.lower().find(key.lower())
    if idx < 0:
        return None
    rest = header[idx + len(key) :]
    if rest.startswith('"'):
        end = rest.find('"', 1)
        return rest[1:end] if end > 0 else None
    end = rest.find(";")
    return (rest if end < 0 else rest[:end]).strip()


def parse_multipart(body: bytes, boundary: bytes):
    """Return (fields: dict[str, str], files: list[(filename, bytes)])."""
    fields: dict[str, str] = {}
    files: list[tuple[str, bytes]] = []
    delim = b"--" + boundary
    for part in body.split(delim):
        part = part.strip(b"\r\n")
        if not part or part == b"--" or b"\r\n\r\n" not in part:
            continue
        raw_headers, content = part.split(b"\r\n\r\n", 1)
        if content.endswith(b"\r\n"):
            content = content[:-2]
        headers = raw_headers.decode("utf-8", "replace")
        disposition = ""
        for line in headers.split("\r\n"):
            if line.lower().startswith("content-disposition"):
                disposition = line
        name = _hval(disposition, "name=")
        filename = _hval(disposition, "filename=")
        if filename:
            files.append((Path(filename).name, content))
        elif name:
            fields[name] = content.decode("utf-8", "replace").strip()
    return fields, files


# ------------------------------------------------------------------------------ page


def _index_html() -> str:
    genre_opts = "".join(
        f'<option value="{g}"{" selected" if g == "pop" else ""}>{g}</option>'
        for g in available_genres()
    )
    return _PAGE.replace("__GENRES__", genre_opts)


# ------------------------------------------------------------------------------ mixing


def _run_mix(runs_root: Path, fields: dict, files: list[tuple[str, bytes]]) -> dict:
    if not files:
        return {"ok": False, "error": "No stem files were uploaded."}

    run_id = time.strftime("%Y%m%d-%H%M%S")
    run_dir = runs_root / run_id
    stems_dir = run_dir / "stems"
    stems_dir.mkdir(parents=True, exist_ok=True)
    saved = 0
    for filename, content in files:
        if not filename.lower().endswith((".wav", ".wave")):
            continue
        (stems_dir / filename).write_bytes(content)
        saved += 1
    if saved == 0:
        return {"ok": False, "error": "None of the uploaded files were WAV files."}

    def fnum(key: str, default: float) -> float:
        try:
            return float(fields.get(key, default))
        except (TypeError, ValueError):
            return default

    settings = MixSettings(
        genre=fields.get("genre", "pop"),
        intensity=fnum("intensity", 0.5),
        vocal_prominence=fnum("vocal", 0.5),
        tone=fnum("tone", 0.0),
        reverb=fnum("reverb", 0.25),
        delay=fnum("delay", 0.12),
        drive=fnum("drive", 0.15),
        sidechain=fnum("sidechain", 0.0),
    )

    stems = load_stems(str(stems_dir))
    result = mix(stems, settings)
    metrics = compute_master_metrics(result.master)
    suggestions = generate_suggestions(result, metrics)

    out_dir = run_dir / "mixed"
    (out_dir / "stems").mkdir(parents=True, exist_ok=True)
    save_wav(out_dir / "master.wav", result.master, bit_depth=24)
    for name, buf in result.stems.items():
        save_wav(out_dir / "stems" / f"{name}.wav", buf, bit_depth=24)
    (out_dir / "dashboard.html").write_text(
        render_dashboard(result, metrics, suggestions, settings)
    )
    (out_dir / "mix_report.txt").write_text(
        render_text(result, settings, metrics, suggestions) + "\n"
    )
    (out_dir / "mix_report.json").write_text(
        json.dumps(build_report(result, settings, metrics, suggestions), indent=2)
    )

    return {
        "ok": True,
        "id": run_id,
        "tracks": saved,
        "lufs": round(metrics.integrated_lufs, 1),
        "peak": round(metrics.peak_dbfs, 1),
        "dashboard": f"/runs/{run_id}/mixed/dashboard.html",
        "master": f"/runs/{run_id}/mixed/master.wav",
        "report": f"/runs/{run_id}/mixed/mix_report.txt",
    }


# ---------------------------------------------------------------------------- handler


def make_handler(runs_root: Path):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):  # quieter console
            pass

        def _send(self, code: int, body: bytes, ctype: str, download: str | None = None):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            if download:
                self.send_header("Content-Disposition", f'attachment; filename="{download}"')
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            path = urlparse(self.path).path
            if path == "/":
                self._send(200, _index_html().encode("utf-8"), _CONTENT_TYPES[".html"])
                return
            if path.startswith("/runs/"):
                target = (runs_root / path[len("/runs/") :]).resolve()
                if runs_root.resolve() not in target.parents or not target.is_file():
                    self._send(404, b"not found", "text/plain")
                    return
                ext = target.suffix.lower()
                ctype = _CONTENT_TYPES.get(ext, "application/octet-stream")
                dl = target.name if ext == ".wav" else None
                self._send(200, target.read_bytes(), ctype, download=dl)
                return
            self._send(404, b"not found", "text/plain")

        def do_POST(self):
            if urlparse(self.path).path != "/mix":
                self._send(404, b"not found", "text/plain")
                return
            ctype = self.headers.get("Content-Type", "")
            if "multipart/form-data" not in ctype or "boundary=" not in ctype:
                self._send(400, b'{"ok":false,"error":"bad form"}', _CONTENT_TYPES[".json"])
                return
            boundary = ctype.split("boundary=", 1)[1].strip().strip('"').encode()
            length = int(self.headers.get("Content-Length", "0"))
            body = self.rfile.read(length)
            fields, files = parse_multipart(body, boundary)
            try:
                result = _run_mix(runs_root, fields, files)
            except Exception as exc:  # surface mixing errors to the UI
                result = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
            self._send(200, json.dumps(result).encode("utf-8"), _CONTENT_TYPES[".json"])

    return Handler


def serve(host: str = "127.0.0.1", port: int = 8765, open_browser: bool = True) -> None:
    runs_root = Path("mixassist_runs")
    runs_root.mkdir(exist_ok=True)
    httpd = ThreadingHTTPServer((host, port), make_handler(runs_root))
    url = f"http://{host}:{port}/"
    print(f"MixAssist app running at {url}")
    print("Open that address in your browser. Press Ctrl-C here to stop.")
    if open_browser:
        with contextlib.suppress(Exception):
            webbrowser.open(url)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping MixAssist app.")
        httpd.shutdown()


def clear_runs() -> None:
    root = Path("mixassist_runs")
    if root.exists():
        shutil.rmtree(root)


_PAGE = """<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>AI Mixing Assistant</title>
<style>
*{box-sizing:border-box}
body{margin:0;background:#0f172a;color:#e2e8f0;font-family:ui-sans-serif,system-ui,-apple-system,Segoe UI,Roboto,sans-serif}
.wrap{max-width:720px;margin:0 auto;padding:28px 18px 64px}
h1{font-size:22px;margin:0 0 4px}
.tag{font-size:11px;color:#0f172a;background:#38bdf8;padding:2px 8px;border-radius:10px;margin-left:6px;vertical-align:middle}
p.sub{color:#94a3b8;margin:6px 0 20px}
.card{background:#1e293b;border-radius:12px;padding:18px;margin-top:16px}
label{display:block;font-size:13px;color:#cbd5e1;margin:14px 0 6px}
select{width:100%;padding:8px;border-radius:8px;background:#0b1220;color:#e2e8f0;border:1px solid #334155}
input[type=range]{width:100%}
.val{color:#38bdf8;font-variant-numeric:tabular-nums}
#drop{border:2px dashed #475569;border-radius:12px;padding:26px;text-align:center;color:#94a3b8;cursor:pointer}
#drop.hi{border-color:#38bdf8;color:#e2e8f0;background:#0b1220}
#files{font-size:12px;color:#cbd5e1;margin-top:8px;white-space:pre-line}
button{margin-top:18px;width:100%;padding:12px;border:0;border-radius:10px;background:#38bdf8;color:#04222f;font-weight:700;font-size:15px;cursor:pointer}
button:disabled{opacity:.5;cursor:default}
.row{display:flex;justify-content:space-between;align-items:center}
#status{margin-top:14px;color:#94a3b8;font-size:14px;min-height:20px}
a.dl{display:inline-block;margin-top:12px;margin-right:10px;padding:9px 14px;border-radius:9px;background:#22c55e;color:#04220f;font-weight:700;text-decoration:none}
a.rep{background:#334155;color:#e2e8f0}
iframe{width:100%;height:640px;border:0;border-radius:12px;margin-top:16px;background:#0b1220}
.hint{color:#64748b;font-size:12px;margin-top:8px}
</style></head><body><div class="wrap">
<h1>AI Mixing Assistant <span class="tag">local app</span></h1>
<p class="sub">Drop your stem WAVs, choose a style, and mix. Everything runs on your Mac.</p>

<form id="f" class="card">
  <div id="drop">Drag &amp; drop your stem .wav files here, or click to choose
    <div id="files"></div>
    <input id="stems" name="stems" type="file" accept=".wav,.wave" multiple hidden>
  </div>
  <div class="hint">Tip: name them so the engine recognizes roles — e.g. Kick.wav, Bass.wav, "Lead Vocal.wav", Snare.wav, Synth.wav.</div>

  <label for="genre">Genre</label>
  <select id="genre" name="genre">__GENRES__</select>

  <label class="row"><span>Intensity (subtle &rarr; aggressive)</span><span class="val" id="iv">0.50</span></label>
  <input type="range" id="intensity" name="intensity" min="0" max="1" step="0.05" value="0.5">

  <label class="row"><span>Vocal prominence</span><span class="val" id="vv">0.50</span></label>
  <input type="range" id="vocal" name="vocal" min="0" max="1" step="0.05" value="0.5">

  <label class="row"><span>Tone (warm &larr; 0 &rarr; clarity)</span><span class="val" id="tv">0.00</span></label>
  <input type="range" id="tone" name="tone" min="-1" max="1" step="0.05" value="0">

  <label class="row"><span>Reverb (space)</span><span class="val" id="rv">0.25</span></label>
  <input type="range" id="reverb" name="reverb" min="0" max="1" step="0.05" value="0.25">

  <label class="row"><span>Delay</span><span class="val" id="dv">0.12</span></label>
  <input type="range" id="delay" name="delay" min="0" max="1" step="0.05" value="0.12">

  <label class="row"><span>Drive (warmth)</span><span class="val" id="gv">0.15</span></label>
  <input type="range" id="drive" name="drive" min="0" max="1" step="0.05" value="0.15">

  <label class="row"><span>Kick &rarr; Bass ducking</span><span class="val" id="sv">0.00</span></label>
  <input type="range" id="sidechain" name="sidechain" min="0" max="1" step="0.05" value="0">

  <button id="go" type="submit">Mix</button>
  <div id="status"></div>
  <div id="links"></div>
</form>
<div id="dash"></div>
</div>
<script>
const $=s=>document.querySelector(s);
const drop=$('#drop'),inp=$('#stems');
drop.onclick=()=>inp.click();
['dragover','dragenter'].forEach(e=>drop.addEventListener(e,ev=>{ev.preventDefault();drop.classList.add('hi')}));
['dragleave','drop'].forEach(e=>drop.addEventListener(e,ev=>{ev.preventDefault();drop.classList.remove('hi')}));
drop.addEventListener('drop',ev=>{inp.files=ev.dataTransfer.files;showFiles()});
inp.addEventListener('change',showFiles);
function showFiles(){const n=inp.files.length;$('#files').textContent=n?[...inp.files].map(f=>f.name).join('\\n'):''}
$('#intensity').oninput=e=>$('#iv').textContent=(+e.target.value).toFixed(2);
$('#vocal').oninput=e=>$('#vv').textContent=(+e.target.value).toFixed(2);
$('#tone').oninput=e=>$('#tv').textContent=(+e.target.value).toFixed(2);
$('#reverb').oninput=e=>$('#rv').textContent=(+e.target.value).toFixed(2);
$('#delay').oninput=e=>$('#dv').textContent=(+e.target.value).toFixed(2);
$('#drive').oninput=e=>$('#gv').textContent=(+e.target.value).toFixed(2);
$('#sidechain').oninput=e=>$('#sv').textContent=(+e.target.value).toFixed(2);
$('#f').addEventListener('submit',async ev=>{
  ev.preventDefault();
  if(!inp.files.length){$('#status').textContent='Please choose some .wav stems first.';return;}
  const fd=new FormData();
  for(const f of inp.files)fd.append('stems',f);
  fd.append('genre',$('#genre').value);
  fd.append('intensity',$('#intensity').value);
  fd.append('vocal',$('#vocal').value);
  fd.append('tone',$('#tone').value);
  fd.append('reverb',$('#reverb').value);
  fd.append('delay',$('#delay').value);
  fd.append('drive',$('#drive').value);
  fd.append('sidechain',$('#sidechain').value);
  $('#go').disabled=true;$('#links').innerHTML='';$('#dash').innerHTML='';
  $('#status').textContent='Mixing '+inp.files.length+' stems… (this can take a bit)';
  try{
    const r=await fetch('/mix',{method:'POST',body:fd});
    const d=await r.json();
    if(!d.ok){$('#status').textContent='Error: '+d.error;$('#go').disabled=false;return;}
    $('#status').textContent='Done — '+d.tracks+' tracks mixed. Master: '+d.lufs+' LUFS, peak '+d.peak+' dBFS.';
    $('#links').innerHTML='<a class="dl" href="'+d.master+'" download>Download master.wav</a>'
      +'<a class="dl rep" href="'+d.report+'" target="_blank">View report</a>';
    $('#dash').innerHTML='<iframe src="'+d.dashboard+'"></iframe>';
  }catch(e){$('#status').textContent='Error: '+e;}
  $('#go').disabled=false;
});
</script></body></html>"""
