#!/usr/bin/env python3
"""
Build no-code browser annotation packages.

Each annotator gets a folder containing:
  - index.html: the annotation app with tasks embedded
  - audio/: the WAV files assigned to that annotator

Annotators can double-click index.html, annotate in their browser, and download
a JSON export that label_studio_bridge.py can import.
"""

import argparse
import json
import shutil
from datetime import datetime
from pathlib import Path


APP_TEMPLATE = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Cello Sound Annotation - {annotator}</title>
  <style>
    :root {{
      color-scheme: light;
      --bg: #f7f7f4;
      --panel: #ffffff;
      --ink: #202124;
      --muted: #666a70;
      --line: #d8d8d2;
      --accent: #1967d2;
      --accent-soft: #e8f0fe;
      --danger: #b3261e;
      --ok: #137333;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      color: var(--ink);
      background: var(--bg);
    }}
    header {{
      position: sticky;
      top: 0;
      z-index: 3;
      background: rgba(247, 247, 244, 0.96);
      border-bottom: 1px solid var(--line);
      padding: 14px 20px;
    }}
    .topbar {{
      max-width: 1120px;
      margin: 0 auto;
      display: flex;
      gap: 16px;
      align-items: center;
      justify-content: space-between;
    }}
    h1 {{
      margin: 0;
      font-size: 20px;
      font-weight: 650;
      letter-spacing: 0;
    }}
    main {{
      max-width: 1120px;
      margin: 0 auto;
      padding: 20px;
      display: grid;
      grid-template-columns: minmax(0, 1fr) 280px;
      gap: 20px;
    }}
    .panel {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 18px;
    }}
    .status {{
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
      align-items: center;
      color: var(--muted);
      font-size: 14px;
    }}
    .pill {{
      border: 1px solid var(--line);
      border-radius: 999px;
      padding: 5px 10px;
      background: #fff;
      color: var(--ink);
    }}
    audio {{
      width: 100%;
      margin: 12px 0 18px;
    }}
    h2 {{
      margin: 18px 0 8px;
      font-size: 16px;
      letter-spacing: 0;
    }}
    .hint {{
      color: var(--muted);
      font-size: 13px;
      line-height: 1.35;
      margin: 0 0 10px;
    }}
    .field {{
      border-top: 1px solid var(--line);
      padding: 14px 0;
    }}
    .field:first-of-type {{ border-top: 0; }}
    .field-title {{
      font-weight: 600;
      margin-bottom: 8px;
    }}
    .field-description {{
      color: var(--muted);
      font-size: 13px;
      line-height: 1.35;
      margin: -2px 0 10px;
    }}
    .choices {{
      display: grid;
      grid-template-columns: repeat(4, minmax(72px, 1fr));
      gap: 8px;
    }}
    .choices.five {{
      grid-template-columns: repeat(5, minmax(56px, 1fr));
    }}
    .choices.flags {{
      grid-template-columns: repeat(3, minmax(0, 1fr));
    }}
    button, .choice {{
      min-height: 42px;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #fff;
      color: var(--ink);
      font: inherit;
      cursor: pointer;
    }}
    button:hover, .choice:hover {{ border-color: var(--accent); }}
    .choice.selected {{
      border-color: var(--accent);
      background: var(--accent-soft);
      color: #174ea6;
      font-weight: 650;
    }}
    .actions {{
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
      margin-top: 18px;
    }}
    .primary {{
      background: var(--accent);
      color: white;
      border-color: var(--accent);
      padding: 0 16px;
      font-weight: 650;
    }}
    .secondary {{ padding: 0 14px; }}
    .danger {{
      color: var(--danger);
      border-color: #f0c8c4;
    }}
    textarea {{
      width: 100%;
      min-height: 76px;
      resize: vertical;
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 10px;
      font: inherit;
    }}
    .side h2 {{ margin-top: 0; }}
    .list {{
      max-height: 58vh;
      overflow: auto;
      border: 1px solid var(--line);
      border-radius: 8px;
    }}
    .row {{
      display: flex;
      justify-content: space-between;
      gap: 8px;
      padding: 8px 10px;
      border-bottom: 1px solid var(--line);
      cursor: pointer;
      font-size: 14px;
    }}
    .row:last-child {{ border-bottom: 0; }}
    .row.active {{ background: var(--accent-soft); }}
    .row.done .done-mark {{ color: var(--ok); }}
    .warning {{
      color: var(--danger);
      min-height: 20px;
      margin-top: 10px;
      font-size: 14px;
    }}
    .small {{
      color: var(--muted);
      font-size: 13px;
      line-height: 1.4;
    }}
    @media (max-width: 840px) {{
      main {{ grid-template-columns: 1fr; }}
      .choices, .choices.five, .choices.flags {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }}
    }}
  </style>
</head>
<body>
  <header>
    <div class="topbar">
      <h1>Cello Sound Annotation</h1>
      <div class="status">
        <span class="pill">Annotator: {annotator}</span>
        <span class="pill" id="progress">0 / 0 complete</span>
      </div>
    </div>
  </header>

  <main>
    <section class="panel">
      <div class="status">
        <span id="recordNumber"></span>
        <span id="dynamic"></span>
      </div>
      <audio id="audio" controls preload="auto"></audio>

      <h2>Technical Quality</h2>
      <p class="hint">Please rate every required item. Replay the start for attack and the end for release.</p>
      <div id="requiredFields"></div>

      <h2>Consistency Through The Stroke</h2>
      <p class="hint">Required. These questions describe how the sound changes from the beginning to the end of the stroke.</p>
      <div id="consistencyFields"></div>

      <h2>Tonal Character</h2>
      <p class="hint">Optional. Use these when the technical quality was clear enough to judge character.</p>
      <div id="optionalFields"></div>

      <h2>Flags / Notes</h2>
      <div id="flagField"></div>
      <textarea id="notes" placeholder="Optional comments"></textarea>

      <div class="warning" id="warning"></div>
      <div class="actions">
        <button class="primary" id="saveNext">Save and Next</button>
        <button class="secondary" id="prev">Previous</button>
        <button class="secondary" id="next">Next</button>
        <button class="secondary" id="download">Download Results</button>
      </div>
    </section>

    <aside class="panel side">
      <h2>Recordings</h2>
      <p class="small">Your progress is saved in this browser. Use Download Results when finished and send back the JSON file.</p>
      <div class="list" id="taskList"></div>
      <div class="actions">
        <button class="secondary" id="download2">Download Results</button>
        <button class="danger" id="clearLocal">Clear Saved Progress</button>
      </div>
    </aside>
  </main>

  <script>
    const ANNOTATOR = {annotator_json};
    const TASKS = {tasks_json};
    const STORAGE_KEY = "cello_annotation_" + ANNOTATOR + "_" + {package_id_json};
    const REQUIRED = [
      ["overall", "Overall quality", "Your holistic judgment of the whole recording. Do not average the other scores.", [
        ["1", "1 Poor"], ["2", "2 Fair"], ["3", "3 Good"], ["4", "4 Great"]
      ]],
      ["tone_quality", "Tone quality", "How resonant, clear, and satisfying the sustained tone is.", [
        ["1", "1 Poor"], ["2", "2 Fair"], ["3", "3 Good"], ["4", "4 Great"]
      ]],
      ["bow_control", "Bow control", "How controlled and even the bowing sounds during the middle of the stroke.", [
        ["1", "1 Poor"], ["2", "2 Fair"], ["3", "3 Good"], ["4", "4 Great"]
      ]],
      ["attack_quality", "Attack quality", "Focus on the first moment of sound: clean start versus scratch, thump, or delayed engagement.", [
        ["1", "1 Poor"], ["2", "2 Fair"], ["3", "3 Good"], ["4", "4 Great"]
      ]],
      ["release_quality", "Release quality", "Focus on the ending: clean finish versus abrupt, noisy, or uneven stop.", [
        ["1", "1 Poor"], ["2", "2 Fair"], ["3", "3 Good"], ["4", "4 Great"]
      ]],
      ["dynamic_accuracy", "Dynamic accuracy", "How closely the recording matches the intended dynamic shown above.", [
        ["1", "1 Off"], ["2", "2 Mostly off"], ["3", "3 Close"], ["4", "4 Matches"]
      ]]
    ];
    const CONSISTENCY = [
      ["dynamic_consistency", "Dynamic consistency", "How does loudness change during the sustained stroke?", [
        ["much_softer", "Gets much softer"],
        ["softer", "Gets softer"],
        ["equal", "Equal level throughout"],
        ["louder", "Gets louder"],
        ["much_louder", "Gets much louder"]
      ]],
      ["tone_consistency", "Tone quality consistency", "Where is the tone clearest or most focused?", [
        ["much_clearer_start", "Much clearer at start"],
        ["clearer_start", "Clearer at start"],
        ["consistent", "Consistent"],
        ["clearer_finish", "Clearer at finish"],
        ["much_clearer_finish", "Much clearer at finish"]
      ]],
      ["noise_consistency", "Surface-noise consistency", "Where is scratch, grit, or bow noise most noticeable?", [
        ["much_noisier_start", "Much noisier at start"],
        ["noisier_start", "Noisier at start"],
        ["even", "Even noise level"],
        ["noisier_finish", "Noisier at finish"],
        ["much_noisier_finish", "Much noisier at finish"]
      ]]
    ];
    const OPTIONAL = [
      ["grainy_smooth", "Grainy to Smooth", "Texture of the tone: rough/gritty versus smooth and even.", [
        ["1", "1 Grainy"], ["2", "2"], ["3", "3 Neutral"], ["4", "4"], ["5", "5 Smooth"]
      ]],
      ["harsh_sweet", "Harsh to Sweet", "Edge of the sound: biting or unpleasant versus rounded and pleasant.", [
        ["1", "1 Harsh"], ["2", "2"], ["3", "3 Neutral"], ["4", "4"], ["5", "5 Sweet"]
      ]],
      ["thin_rich", "Thin to Rich", "Body of the tone: small/weak/narrow versus full and rich.", [
        ["1", "1 Thin"], ["2", "2"], ["3", "3 Neutral"], ["4", "4"], ["5", "5 Rich"]
      ]],
      ["airy_clear", "Airy to Clear", "Airy means breathy or whispery bow noise in the tone; clear means focused pitch with little airy noise.", [
        ["1", "1 Airy"], ["2", "2"], ["3", "3 Neutral"], ["4", "4"], ["5", "5 Clear"]
      ]],
      ["dry_resonant", "Dry to Resonant", "How much the sound rings or blooms after it starts.", [
        ["1", "1 Dry"], ["2", "2"], ["3", "3 Neutral"], ["4", "4"], ["5", "5 Resonant"]
      ]]
    ];
    const FLAGS = [["skip_flag", "Flag", "Use these only when the recording has a technical problem that makes quality ratings misleading.", [
      ["normal", "Normal"],
      ["audio_problem", "Audio problem"],
      ["robot_problem", "Robot problem"]
    ]]];

    let current = 0;
    let answers = loadAnswers();

    function loadAnswers() {{
      try {{
        return JSON.parse(localStorage.getItem(STORAGE_KEY) || "{{}}");
      }} catch {{
        return {{}};
      }}
    }}

    function persist() {{
      localStorage.setItem(STORAGE_KEY, JSON.stringify(answers));
    }}

    function keyForTask(task) {{
      return task.data.stroke_id + "::" + task.data.repeat;
    }}

    function answerForCurrent() {{
      const key = keyForTask(TASKS[current]);
      if (!answers[key]) answers[key] = {{}};
      return answers[key];
    }}

    function renderChoiceGroup(container, fields, five=false, flags=false) {{
      container.innerHTML = "";
      for (const [name, title, description, choicesData] of fields) {{
        const field = document.createElement("div");
        field.className = "field";
        const heading = document.createElement("div");
        heading.className = "field-title";
        heading.textContent = title;
        field.appendChild(heading);
        if (description) {{
          const desc = document.createElement("div");
          desc.className = "field-description";
          desc.textContent = description;
          field.appendChild(desc);
        }}
        const choices = document.createElement("div");
        choices.className = "choices" + (five ? " five" : "") + (flags ? " flags" : "");
        for (const [value, label] of choicesData) {{
          const btn = document.createElement("button");
          btn.type = "button";
          btn.className = "choice";
          btn.dataset.field = name;
          btn.dataset.value = value;
          btn.textContent = label;
          btn.addEventListener("click", () => {{
            const ans = answerForCurrent();
            ans[name] = value;
            persist();
            render();
          }});
          choices.appendChild(btn);
        }}
        field.appendChild(choices);
        container.appendChild(field);
      }}
    }}

    function updateSelected() {{
      const ans = answerForCurrent();
      for (const btn of document.querySelectorAll(".choice")) {{
        btn.classList.toggle("selected", ans[btn.dataset.field] === btn.dataset.value);
      }}
      document.getElementById("notes").value = ans.notes || "";
    }}

    function isComplete(task) {{
      const ans = answers[keyForTask(task)] || {{}};
      if (ans.skip_flag === "audio_problem" || ans.skip_flag === "robot_problem") return true;
      return REQUIRED.every(([name]) => ans[name]) && CONSISTENCY.every(([name]) => ans[name]);
    }}

    function validateCurrent() {{
      const ans = answerForCurrent();
      if (ans.skip_flag === "audio_problem" || ans.skip_flag === "robot_problem") return true;
      const missing = REQUIRED.concat(CONSISTENCY).filter(([name]) => !ans[name]).map(([, title]) => title);
      if (missing.length) {{
        document.getElementById("warning").textContent = "Please complete: " + missing.join(", ");
        return false;
      }}
      document.getElementById("warning").textContent = "";
      return true;
    }}

    function renderTaskList() {{
      const list = document.getElementById("taskList");
      list.innerHTML = "";
      TASKS.forEach((task, index) => {{
        const row = document.createElement("div");
        row.className = "row" + (index === current ? " active" : "") + (isComplete(task) ? " done" : "");
        row.addEventListener("click", () => {{ saveNotes(); current = index; render(); }});
        row.innerHTML = "<span>" + (index + 1) + ". Recording</span><span class='done-mark'>" + (isComplete(task) ? "done" : "") + "</span>";
        list.appendChild(row);
      }});
    }}

    function saveNotes() {{
      const ans = answerForCurrent();
      ans.notes = document.getElementById("notes").value;
      persist();
    }}

    function render() {{
      const task = TASKS[current];
      const data = task.data;
      document.getElementById("recordNumber").textContent = "Recording " + (current + 1) + " of " + TASKS.length;
      document.getElementById("dynamic").textContent = "Intended dynamic: " + (data.dynamic_est || "unknown");
      document.getElementById("audio").src = data.audio;
      document.getElementById("warning").textContent = "";
      updateSelected();
      renderTaskList();
      const done = TASKS.filter(isComplete).length;
      document.getElementById("progress").textContent = done + " / " + TASKS.length + " complete";
    }}

    function toLabelStudioResult(fields) {{
      const result = [];
      for (const [name, value] of Object.entries(fields)) {{
        if (name === "notes") {{
          if (value) result.push({{ from_name: "notes", to_name: "audio", type: "textarea", value: {{ text: [value] }} }});
        }} else {{
          result.push({{ from_name: name, to_name: "audio", type: "choices", value: {{ choices: [String(value)] }} }});
        }}
      }}
      return result;
    }}

    function downloadResults() {{
      saveNotes();
      const exported = TASKS.map(task => {{
        const fields = answers[keyForTask(task)] || {{}};
        return {{
          data: task.data,
          annotations: Object.keys(fields).length ? [{{
            completed_at: new Date().toISOString(),
            result: toLabelStudioResult(fields)
          }}] : []
        }};
      }});
      const blob = new Blob([JSON.stringify(exported, null, 2)], {{ type: "application/json" }});
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = "annotations_" + ANNOTATOR + ".json";
      document.body.appendChild(a);
      a.click();
      a.remove();
      URL.revokeObjectURL(url);
    }}

    document.addEventListener("DOMContentLoaded", () => {{
      renderChoiceGroup(document.getElementById("requiredFields"), REQUIRED);
      renderChoiceGroup(document.getElementById("consistencyFields"), CONSISTENCY, true);
      renderChoiceGroup(document.getElementById("optionalFields"), OPTIONAL, true);
      renderChoiceGroup(document.getElementById("flagField"), FLAGS, false, true);
      document.getElementById("notes").addEventListener("input", saveNotes);
      document.getElementById("saveNext").addEventListener("click", () => {{
        saveNotes();
        if (!validateCurrent()) return;
        if (current < TASKS.length - 1) current += 1;
        render();
      }});
      document.getElementById("prev").addEventListener("click", () => {{ saveNotes(); if (current > 0) current -= 1; render(); }});
      document.getElementById("next").addEventListener("click", () => {{ saveNotes(); if (current < TASKS.length - 1) current += 1; render(); }});
      document.getElementById("download").addEventListener("click", downloadResults);
      document.getElementById("download2").addEventListener("click", downloadResults);
      document.getElementById("clearLocal").addEventListener("click", () => {{
        if (confirm("Clear saved progress for this package?")) {{
          answers = {{}};
          persist();
          render();
        }}
      }});
      render();
    }});
  </script>
</body>
</html>
"""


README_TEMPLATE = """# Cello Sound Annotation

Thank you for helping annotate these recordings.

## What to do

1. Open `index.html` in Chrome, Safari, Edge, or Firefox.
2. Listen to each recording.
3. Rate each required quality item.
4. Use the optional tonal-character ratings when you feel confident.
5. When you are finished, click **Download Results**.
6. Send back the downloaded file named `annotations_{annotator}.json`.

Your progress is saved in your browser as you work. If you close the window and
open `index.html` again on the same computer/browser, your previous ratings
should still be there.

## Rating scale

- 1: poor
- 2: fair
- 3: good
- 4: great

If the audio file has a technical problem or the robot clearly malfunctioned,
use the flag buttons instead of forcing a quality rating.
"""


def task_audio_name(task):
    data = task.get("data", {})
    if data.get("audio_file"):
        return data["audio_file"]
    audio = str(data.get("audio", ""))
    return audio.rsplit("/", 1)[-1]


def localize_tasks(tasks):
    localized = []
    for task in tasks:
        copied = json.loads(json.dumps(task))
        audio_name = task_audio_name(copied)
        copied["data"]["audio"] = f"audio/{audio_name}"
        copied["data"]["audio_file"] = audio_name
        localized.append(copied)
    return localized


def build_package(tasks_file, audio_dir, out_dir, annotator):
    tasks = json.loads(tasks_file.read_text())
    localized = localize_tasks(tasks)
    package_dir = out_dir / annotator
    package_audio = package_dir / "audio"
    if package_dir.exists():
        shutil.rmtree(package_dir)
    package_audio.mkdir(parents=True, exist_ok=True)

    missing = []
    for task in localized:
        audio_name = task_audio_name(task)
        src = audio_dir / audio_name
        if not src.exists():
            missing.append(audio_name)
            continue
        shutil.copy2(src, package_audio / audio_name)

    if missing:
        raise SystemExit(f"{annotator}: missing {len(missing)} audio files, first missing: {missing[0]}")

    package_id = f"{annotator}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    html = APP_TEMPLATE.format(
        annotator=annotator,
        annotator_json=json.dumps(annotator),
        package_id_json=json.dumps(package_id),
        tasks_json=json.dumps(localized, separators=(",", ":")),
    )
    (package_dir / "index.html").write_text(html)
    (package_dir / "README.txt").write_text(README_TEMPLATE.format(annotator=annotator))
    return package_dir, len(localized)


def main():
    parser = argparse.ArgumentParser(description="Create public, no-code annotation packages")
    parser.add_argument("--tasks-dir", required=True, help="Directory containing tasks_ANNOTATOR.json files")
    parser.add_argument("--audio-dir", required=True, help="Directory containing WAV files")
    parser.add_argument("--out-dir", required=True, help="Output directory for annotator packages")
    parser.add_argument("--annotators", nargs="+", required=True, help="Annotator IDs to package")
    args = parser.parse_args()

    tasks_dir = Path(args.tasks_dir)
    audio_dir = Path(args.audio_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "created_at": datetime.now().isoformat(),
        "tasks_dir": str(tasks_dir),
        "audio_dir": str(audio_dir),
        "packages": {},
    }

    for annotator in args.annotators:
        tasks_file = tasks_dir / f"tasks_{annotator}.json"
        if not tasks_file.exists():
            raise SystemExit(f"Missing task file: {tasks_file}")
        package_dir, n_tasks = build_package(tasks_file, audio_dir, out_dir, annotator)
        manifest["packages"][annotator] = {
            "path": str(package_dir),
            "n_tasks": n_tasks,
        }
        print(f"{annotator}: {n_tasks} tasks -> {package_dir}")

    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"Manifest: {out_dir / 'manifest.json'}")


if __name__ == "__main__":
    main()
