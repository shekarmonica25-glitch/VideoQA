# VideoQA

A Windows app that **watches and listens to a video** and writes every defect it finds,
with its **timestamp**, into a log file (`.log`) and a spreadsheet (`.csv`).

Input: **only the video**. No script or metadata needed.

## Defects it finds

| Defect | Example |
|---|---|
| `MISSING_IMAGE` | Black or blank screen |
| `FROZEN_VIDEO` | Picture stuck for 3+ seconds |
| `VIDEO_BLUR` | Picture much less sharp than the rest of the video |
| `FRAME_FRAGMENT` | Torn / blocky / green-patch frames, decoder errors |
| `NO_AUDIO` | No sound, sound ends before the picture, **person talking but no voice** |
| `AUDIO_LAG` | Voice cuts out mid-speech, clicks/pops, stutter |
| `AUDIO_OVERLAP` | Two voices talking at the same time |
| `DOUBLE_BGM` | Two background songs at once, or the same music layered twice |
| `LANGUAGE_CHANGE` | Speech switches to another language |
| `AUDIO_DELAY` | Voice early/late compared to the lips (conservative) |
| `GENDER_MISMATCH` | Speaking face looks male/female but the voice sounds the opposite |

Items marked **(please check)** are the app's guesses and should be confirmed by a person.

## Run from source

```
py -3.14 -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe download_models.py
.venv\Scripts\python.exe app.py                     (window)
.venv\Scripts\python.exe app.py video.mp4 -o logs   (command line)
```

## Build the .exe

```
.venv\Scripts\python.exe build_exe.py
```

The app appears in `dist\VideoQA\` (about 1 GB, including the AI models). Copy the whole folder.

## More details

- [context.md](context.md): how every check works (explained simply), test results, known limits
- [README_USER.txt](README_USER.txt): instructions for people using the `.exe`

## Licence note

The face male/female model (InsightFace `genderage.onnx`) is for **non-commercial research only**.
For commercial use, replace it or turn off the "Wrong voice gender" check.
All other components use MIT / Apache / BSD licences.
