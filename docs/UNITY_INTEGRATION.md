# Open in Unity — Integration Guide

The Model Gallery has an **Open in Unity** action that spawns a generated GLB
into Unity Editor — either in a fresh empty scene, or in the scene that is
currently open. The launcher installs itself from the app on first click; no
terminal commands required.

> macOS only for now. Windows / Linux noted at the bottom.

---

## How it works

```
[Open in Unity button]
        │  (first click only) POST /api/v1/unity/register-launcher
        │      → installs UnityLauncher.app in ~/Applications
        │      → registers unity3dgen:// with Launch Services
        ▼
window.location.href = "unity3dgen://spawn?url=<glb>&scene=new|existing"
        │
        ▼
UnityLauncher.app  (macOS routes the URL here)
        │ writes SpawnRequest JSON into
        │ UnityProject/Assets/SpawnRequests/
        │ launches Unity on UnityProject/ if not running
        ▼
SpawnBridge.cs  (Editor, [InitializeOnLoad])
        │ downloads the GLB, imports via glTFast,
        │ opens/keeps scene, instantiates prefab, frames it
        ▼
Model visible in the Scene view
```

No backend changes are needed for the happy path at runtime — the backend is
only used once to install the URL handler, and again whenever you move the
repo (the bundle has the repo path baked in and needs re-registering).

---

## Prerequisites

- **macOS** (Darwin).
- **Unity Hub + Unity 2022.3 LTS** — install from <https://unity.com/download>.
  The launcher shells out to `open -a Unity`; any 2022.3.x works.
- Backend running **natively** (not inside Docker), e.g. `uvicorn app.main:app
  --reload --port 8000` — the installer must touch `~/Applications/` on the
  host, which a container cannot do.

---

## Install from the UI

1. Start backend + frontend (`docker-compose up --build`, or the dev commands
   in `CLAUDE.md` — note the backend must run **natively** for the installer
   to work; generation endpoints can stay in Docker if you prefer).
2. Open the Model Gallery view. If the launcher isn't installed yet you'll
   see an amber banner:

   > Unity launcher is not installed yet. Install it to enable **Open in Unity**. &nbsp;&nbsp; [ Install now ]

3. Click **Install now**. This POSTs `/api/v1/unity/register-launcher`, which:
   - compiles `scripts/UnityLauncher/handler.applescript` into
     `~/Applications/UnityLauncher.app`,
   - injects `CFBundleURLTypes` for `unity3dgen://` into the bundle's
     Info.plist,
   - bakes the absolute path to this repo into
     `UnityLauncher.app/Contents/Resources/handler.sh`,
   - registers the bundle with Launch Services.

   Takes a second or two. On success the banner disappears and the **Unity ▾**
   dropdowns on each model card become active.

4. First Unity open: launch `UnityProject/` once through Unity Hub (or just
   let the launcher open it on the first click). Unity resolves the
   `com.unity.cloud.gltfast` package and compiles the `SpawnBridge` editor
   script — takes a minute the first time.

If you skip step 3 explicitly, the first click on **Unity ▾ → New/Current
scene** runs the installer automatically before firing the URI.

### Re-installing

The installer is idempotent — click it again any time. You'll want to re-run
it if you move the repo to a different path (the `repo_match` field in
`GET /api/v1/unity/launcher-status` goes false and the banner reappears).

---

## Usage

On each card in the Model Gallery:

- **Unity ▾** → **New scene**: save current scene if dirty, open an empty
  scene, drop the model in.
- **Unity ▾** → **Current scene**: spawn into whatever scene is active right
  now. If Unity isn't running, it's launched and the spawn fires as soon as
  the Editor finishes loading.

The full-screen viewer (click a card) exposes both options as large buttons.
The spawned GameObject is selected and framed in the Scene view. Imported
GLBs are cached under `UnityProject/Assets/Imported/<id>.glb`.

---

## Troubleshooting

**Banner says "Unity integration unavailable: Backend is running in Docker"**
- The installer must run on the host — a container can't touch `~/Applications/`.
  Run the backend with `uvicorn` natively. Generation workers can still be in
  Docker; only the backend API process needs host access for the install step.

**Button clicks but Unity never opens**
- Check `GET /api/v1/unity/launcher-status`. `registered: false` means Launch
  Services hasn't picked up the bundle — click **Install now** again.
- Chrome sometimes swallows unknown protocols silently. Try Safari once to
  confirm the scheme is bound.

**Unity opens but nothing spawns**
- Open `Window → Package Manager` in Unity and confirm `glTFast` is installed
  and resolved without errors. If it's missing, the import step produces no
  prefab and `SpawnBridge` logs
  `[SpawnBridge] glTFast did not produce a prefab for …` in the Console.
- Check the Unity Console for download errors. `SpawnBridge` logs
  `[SpawnBridge] downloading …` and `[SpawnBridge] spawned …` on success.
- Confirm the GLB URL is reachable from Unity. Open the Download link in the
  gallery — if that works, the URL the URI embeds works too.

**"Install failed: osacompile not found"**
- You're on a trimmed-down macOS (CI, minimal VM). Install Xcode Command Line
  Tools: `xcode-select --install`.

**Spawn requests pile up in `UnityProject/Assets/SpawnRequests/`**
- They should always be deleted, including on the error path. Anything left
  behind is a bug — check the Unity Console for a stack trace and file it.

---

## API reference

- `GET /api/v1/unity/launcher-status` →
  ```json
  { "supported": true, "installed": true, "registered": true,
    "repo_match": true, "app_path": "/Users/.../UnityLauncher.app",
    "repo_root": "/Users/.../3D-Generator" }
  ```
  On non-macOS or Docker: `supported: false` with a `reason` string.

- `POST /api/v1/unity/register-launcher` → installs/re-installs the bundle.
  Returns the same shape. 400 if macOS-only constraints are violated.

---

## Windows / Linux (not yet implemented)

The design ports cleanly but each platform needs its own installer branch:

- **Windows:** write a `.reg` fragment pointing the `unity3dgen` protocol at a
  `.ps1` or `.bat` that mirrors `handler.sh.template`. See
  [MS docs on custom URL protocols](https://learn.microsoft.com/en-us/previous-versions/windows/internet-explorer/ie-developer/platform-apis/aa767914(v=vs.85)).
- **Linux:** install a `.desktop` file with
  `MimeType=x-scheme-handler/unity3dgen;` via `xdg-mime default`.

Everything platform-agnostic — the SpawnRequest JSON format, the Unity editor
bridge, the frontend buttons, the status/install endpoints' shape — stays as
is; only the OS-specific branch inside `routes_unity.py::register_launcher`
changes.
