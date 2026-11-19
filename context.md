# VideoQA – Project Context

> This file explains the whole project in very simple words,
> like you would explain it to a 5-year-old. 🧸

---

## 1. What are we building?

Imagine you have a **robot friend** who watches your videos for you.

The robot watches **and** listens to the whole video.
Every time something goes wrong, it writes it down in a **notebook**,
along with **the exact time** it happened.

- The **robot** = our app, called **`VideoQA.exe`**
- The **notebook** = a **log file** (a text file)
- You give the robot **only the video**, nothing else. No script, no notes.

---

## 2. The decisions we already made

| Question | Answer |
|---|---|
| Which language do we write the robot in? | **Python** (it knows the most "AI tricks" for sound and video) |
| How will people open it? | As an **`.exe` file**: double-click and it runs. No Python needed. |
| What does the robot get? | **Only the video file** (no script, no metadata) |
| What does the robot give back? | A **log file** listing every problem **with its time** |
| Where does the project live? | `D:\RClone\VideoQA\` |
| What computer is it built on? | Windows 11, Python 3.14, NVIDIA RTX 3050 graphics card (4 GB) |

---

## 3. The problems the robot looks for (explained like you're 5)

### 🗣️ 1. Audio delay
**What it is:** The person's **mouth moves first**, and their **voice comes a little later**.
Like a cartoon where the lips and the words don't match.

**How the robot checks:**
1. It watches the **mouth** opening and closing.
2. It listens for when the **voice** gets loud and quiet.
3. It slides the two "patterns" until they match best.
4. If it has to slide them more than a tiny bit (about 1/10 of a second), the audio is late. ✍️ Write it down!

---

### 👥 2. Audio overlap
**What it is:** **Two voices talking at the same time** when only one should be talking.
Like two kids shouting at once.

**How the robot checks:**
1. Every voice has its own **"note"** (how high or low it sounds).
2. If the robot hears **two different notes at the same moment** for a while,
   two voices are overlapping. ✍️ Write it down!

---

### 🎵🎵 3. Double BGM (background music)
**What it is:** **Two songs playing together** in the background.
Like two radios playing in the same room.

**How the robot checks:**
1. Every song has a **beat** (boom… boom… boom…).
2. One song = **one steady beat**. Two songs = **two beats bumping into each other**.
3. It also checks if the **same song is playing twice**, a little bit apart (sounds like an echo).
4. If yes, ✍️ write it down!

---

### 🌫️ 4. Video blur
**What it is:** The picture becomes **fuzzy**, like looking through foggy glasses.

**How the robot checks:**
1. A sharp picture has lots of **clear edges** (lines between things).
2. A blurry picture has **soft, mushy edges**.
3. The robot counts the edges in every picture.
4. If a part of the video suddenly has far fewer edges than the rest, it's blurry. ✍️ Write it down!

---

### 🐢 5. Lag in audio (stutter, gaps, clicks)
**What it is:** The sound **stops for a moment in the middle of a sentence**,
**repeats itself** ("hel-hel-hello"), or makes a **"click/pop"** noise.

**How the robot checks:**
1. It finds the parts where someone is **talking**.
2. If there's a sudden **hole of silence** in the middle of talking, that's a gap.
3. If a tiny piece of sound **repeats** right after itself, that's a stutter.
4. If there's a sudden **sharp spike**, that's a click.
5. ✍️ Write each one down!

---

### 🧩 6. Frame fragmentation
**What it is:** The picture **breaks into pieces**: blocky squares, green or pink patches,
or a torn picture. Like a broken puzzle.

**How the robot checks:**
1. A video is many pictures shown very fast (like a flip book).
2. The robot compares each picture with the one before it.
3. If one picture is **suddenly very different** and then the next picture is **normal again**,
   that picture was broken.
4. It also looks for **big square blocks** and **weird green or pink areas**.
5. It also asks the video player: "Did you have trouble reading any part?"
6. ✍️ Write it down!

---

### 🖼️❌ 7. Missing image
**What it is:** The picture is **all black**, **all one colour**, or **frozen** (stuck).

**How the robot checks:**
1. **All black?** It checks if every dot in the picture is dark.
2. **All one colour?** It checks if every dot is almost the same colour.
3. **Frozen?** It checks if the picture **doesn't change at all** for a long time.
4. ✍️ Write it down!

---

### 👩🔄👨 8. Wrong voice (male voice for a woman, or female voice for a man)
**What it is:** A **woman is on screen talking**, but you hear a **man's voice**,
or the other way around.

**How the robot checks** (we have no script, so it must guess):
1. It finds **which face is talking** (whose mouth is moving).
2. It **guesses** whether that face looks like a man or a woman.
3. It listens to the voice. **Deep/low** voices are usually men, **high** voices are usually women.
4. If the face and the voice **don't match**, ✍️ write it down!

> ⚠️ **Important:** Guessing from a face or a voice is **not always right**.
> Some women have deep voices and some men have high voices.
> So the robot marks these as **"please check"**, not as a sure mistake.

---

### 🔇 9. No audio at the end
**What it is:** The video **keeps playing**, but the **sound stops**, usually near the end.

**How the robot checks:**
1. It measures **how long the picture is** and **how long the sound is**.
   If the sound is shorter, the end is silent. ✍️ Write it down!
2. It also listens to the **last part** of the video. If it's silent but the picture
   is still moving (or a mouth is moving), ✍️ write it down!

---

### 🌍 10. Audio language change
**What it is:** The video is in **one language** (for example English) and suddenly
switches to **another language** (for example Hindi).

**How the robot checks:**
1. It cuts the talking into **small pieces** (about 10 seconds each).
2. A smart listening AI called **Whisper** tells the robot which language each piece is in.
3. The language used **most of the time** = the "main language".
4. Any piece in a **different language** that the robot is sure about. ✍️ Write it down!

---

## 4. How the robot works, step by step

```
   🎬 Video file
        │
        ▼
 ① Open the video and read its "label" (how long, how many pictures per second)
        │
        ▼
 ② Separate it into two parts:
      🖼️ pictures (the flip-book pages)     🔊 sound (the audio track)
        │                                     │
        ▼                                     ▼
 ③ Picture checks:                       Sound checks:
    • missing image                        • audio lag / gaps / clicks
    • frozen picture                       • audio overlap
    • blur                                 • double BGM
    • frame fragmentation                  • language change
                                           • no audio at end
        │                                     │
        └──────────────┬──────────────────────┘
                       ▼
 ④ Checks that need BOTH picture and sound:
    • audio delay (mouth vs voice)
    • wrong voice gender (face vs voice)
                       │
                       ▼
 ⑤ 📒 Write everything into the log file, sorted by time
```

---

## 5. What the notebook (log file) will look like

```
=== Video QA Report ===
File     : D:\Videos\episode_12.mp4
Scanned  : 2026-11-18 14:32:05
Duration : 00:05:42.300

[00:00:12.500 - 00:00:14.000]  MISSING_IMAGE      Black frames for 1.5s              (confidence 0.98)
[00:01:03.200 - 00:01:05.800]  VIDEO_BLUR         Sharpness dropped 72% vs average   (confidence 0.85)
[00:02:10.000 - 00:02:16.400]  DOUBLE_BGM         Two music beats detected           (confidence 0.79)
[00:02:45.100]                 AUDIO_DELAY        Voice is 180 ms late vs mouth      (confidence 0.81)
[00:03:30.000 - 00:03:38.000]  LANGUAGE_CHANGE    Main: English, here: Hindi         (confidence 0.93)
[00:04:01.600]                 GENDER_MISMATCH    Face looks female, voice is male   (confidence 0.70 - please check)
[00:04:12.300]                 FRAME_FRAGMENT     Broken/blocky picture              (confidence 0.91)
[00:05:30.000 - 00:05:42.300]  NO_AUDIO_AT_END    Picture continues, sound is silent (confidence 0.97)

Total defects: 8
```

- The **time** tells you **where to look** in the video.
- **Confidence** = how sure the robot is (1.00 = totally sure, 0.50 = coin flip).

---

## 6. The "toy box": tools the robot uses

| Tool | What it does (simple) |
|---|---|
| **Python** | The language we write the robot in |
| **FFmpeg** | A Swiss-army knife for video: opens it, splits sound from pictures, finds black/frozen/silent parts |
| **OpenCV** | Looks at pictures: blur, broken frames, finding faces and mouths |
| **NumPy / SciPy** | Does the maths on sound (beats, voice notes, clicks) |
| **Whisper (faster-whisper)** | AI that listens to speech and says which language it is |
| **Silero VAD** (inside faster-whisper) | Finds the parts where someone is talking |
| **pyannote segmentation** | AI that hears when two people talk at the same time (overlap) |
| **YuNet** | Finds faces and mouth corners |
| **InsightFace genderage** | Guesses if a face looks male or female |
| **ONNX Runtime** | Runs the small AI "brains" above |
| **Tkinter** | Makes the simple window with buttons |
| **PyInstaller** | Wraps everything into one `VideoQA.exe` |

> 💡 FFmpeg is **packed inside the app**, so you **don't** need to install it yourself
> (the error "ffmpeg is not recognized" will not happen for the app).

---

## 7. The building plan (in order)

1. ✅ **Write this context file**
2. ✅ **Set up the project**: create a private Python "box" (`.venv`) and install the tools
3. ✅ **Picture checks**: missing image, frozen, blur, fragmentation
4. ✅ **Sound + AI checks**: lag, overlap, double BGM, language, no audio at end,
   audio delay, wrong voice gender
5. ✅ **The window and the notebook**: buttons to pick a video or folder, plus the log file writer
6. ✅ **Test it**: make a test video with **planted mistakes** and check the robot finds them
7. ✅ **Wrap it** into `VideoQA.exe`
8. ✅ **Try it on a real video** (tailoring lesson) and tune the settings: 142 → 13 alarms
9. ✅ **A person checked** "voice missing" at 2:55 and 4:54: **real defects** 🎉
10. ⬜ **Add SyncNet** (trained lip-sync AI) for a reliable audio-delay check

---

## 8. Things to remember ⚠️

- The robot is a **helper, not a judge**. A human should **double-check** what it finds,
  especially **wrong voice gender** and **overlap**, because those are guesses.
- The first scan can be **slower** because the AI "brains" need to wake up.
  The graphics card (RTX 3050) helps make it faster.
- The `.exe` folder will be **big** (hundreds of MB to a few GB) because the AI brains live inside it.
- Since there is **no script**, the robot learns what is "normal" **from the video itself**
  (for example, "main language = the language spoken most of the time").

---

## 9. Current status (2026-11-18)

**The robot is built!** 🎉 → `D:\RClone\VideoQA\dist\VideoQA\VideoQA.exe`

- Copy the **whole `dist\VideoQA` folder** (about 1.07 GB) to any Windows PC. Nothing else to install.
- Double-click `VideoQA.exe` → add videos → **Start checking** → a `.log` file (easy to read)
  and a `.csv` file (for Excel) are written for every video.
- It can also run from the command line for many videos at once (see `README.txt` in the folder).
- Speed: a 50-second video takes about **37 seconds**. The very first run is slower
  because Windows scans the new files.

### Test results
The test video (`tests\out\test_defects.mp4`) has **13 planted mistakes**.
The robot found **all 13**, with **no false alarms**:

| Planted mistake | Found? |
|---|---|
| Black screen, blank grey screen | ✅ ✅ |
| Blur | ✅ |
| Scrambled frame, green patch, blocky squares | ✅ ✅ ✅ |
| Frozen picture | ✅ |
| Voice dropout, stutter, click | ✅ ✅ ✅ |
| Two voices at once (overlap) | ✅ |
| Two songs at once (double BGM) | ✅ |
| Sound ends before the picture | ✅ |

**Lip-sync delay** and **wrong voice gender** need a **real face** on screen.
The test video has no real faces, so these were tested with a *pretend face*.
The robot correctly found "voice 200 ms late", "voice 160 ms early",
"face male + voice female" and "face female + voice male". They still need
testing on **real videos with people talking**.

### Real video test: "Chapter 1A - Tailoring tools introduction.mp4" (16 min)
**First try:** 142 alarms, mostly **false**, and it took 26 minutes. Fixes made:

| Problem | Why it happened | Fix |
|---|---|---|
| 76 × "male voice on a woman" | Background music made the robot hear a "ghost" note 3x too low | New pitch finder (autocorrelation) + a piece only counts as male if most of it is low |
| 26 × "two songs at once" | Her talking rhythm looked like a second beat | Beats are now measured only in the deep bass, where voices don't reach |
| Blur on cartoon slides | Flat cartoons have few fine details | Blur now measures the crispest edges, not the average |
| "Frozen" while she stood still | The old test was too loose | Frozen now needs (almost) zero pixels changing |
| Random lip-sync delays | Her face is small, so the simple mouth measure is too noisy | Delay reported only when many windows in a row strongly agree |
| Very slow | Language AI used slow int8 maths | Switched to float32 (3.5x faster) |

**Second try:** 13 alarms, 7 minutes:
- ✅ **14:28.16 broken (torn) frame**: real, confirmed by looking at the frame.
- ✅ **11 × "voice missing" (mouth moving but no voice)**: her mouth makes speech shapes while only music plays.
  **Confirmed real by the user** at 2:55 and 4:54. The other 9 are the same kind but not yet watched.
  The alarm's message is now clearer and its confidence was raised to 0.75.
- ❓ **5:34.586 click**: not checked yet.

### Known limit: lip sync
The simple mouth measure cannot reliably match lips to voice when the face is small.
A proper fix is a trained lip-sync AI (**SyncNet**). This is the suggested next step.

### Not tested yet
- **Language change**: Windows on this PC only has English voices, so we could not make
  a test video with two languages. On the test video the robot correctly says
  "main language = English" and does not flag anything wrongly.
- **Real-world videos**: thresholds were tuned on the test video. Real videos may give
  some false alarms, so they may need tuning.

### ⚠️ Licence warning
The face male/female model (**InsightFace genderage**) is licensed for
**non-commercial research only**. For commercial or company use, replace
`models\genderage.onnx` with a model your company is allowed to use (or untick the
"Wrong voice gender" check). All other parts are free for commercial use
(MIT / Apache / BSD licences).

### Project files (simple map)
| File | What it is |
|---|---|
| `app.py` | The window + command line (start here) |
| `videoqa\video_checks.py` | Picture checks (black, frozen, blur, broken frames) |
| `videoqa\faces.py` | Finds faces/mouths, guesses face gender |
| `videoqa\audio_core.py` | Sound measuring (loudness, voice pitch, where talking is) |
| `videoqa\audio_checks.py` | All sound checks + lip sync + gender + language |
| `videoqa\analyzer.py` | Runs every check and writes the log/CSV |
| `download_models.py` | Downloads the AI brains into `models\` |
| `build_exe.py` | Makes the `.exe` |
| `tests\make_test_video.py` | Makes the test video with planted mistakes |
| `tests\test_av_logic.py` | Tests lip-sync and gender logic with a pretend face |

### How to rebuild the exe after changing code
```
.venv\Scripts\python.exe download_models.py     (only the first time)
.venv\Scripts\python.exe build_exe.py
```
