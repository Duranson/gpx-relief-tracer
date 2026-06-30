# gpx-relief-tracer
Provides a tool to dynamically trace a GPX file on a 3D relief as a video.

## Requirements

### Blender
Download and install from https://www.blender.org/download/

### Python dependencies (`.venv`)
```powershell
python -m venv .venv
.venv\Scripts\pip install numpy rasterio gpxpy pytest
```

### ffmpeg (required to assemble PNG frames into MP4)

**Option A — winget (Windows 10/11, recommended)**
```powershell
winget install Gyan.FFmpeg
```
Then open a new terminal — winget adds ffmpeg to your PATH automatically.

**Option B — manual install**
1. Download a Windows build from https://www.gyan.dev/ffmpeg/builds/ (choose `ffmpeg-release-essentials.zip`)
2. Extract the archive, e.g. to `C:\ffmpeg`
3. Add the `bin` folder to your PATH:
```powershell
# Run once in an Administrator PowerShell — persists across reboots
[Environment]::SetEnvironmentVariable(
    "PATH",
    [Environment]::GetEnvironmentVariable("PATH", "Machine") + ";C:\ffmpeg\bin",
    "Machine"
)
```
4. Open a new terminal and verify:
```powershell
ffmpeg -version
```
