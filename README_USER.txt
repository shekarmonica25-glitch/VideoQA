VideoQA - automatic video defect finder
=======================================

HOW TO USE (window)
  1. Double-click VideoQA.exe
  2. Click "Add video files..." or "Add a folder..."
  3. (Optional) choose where to save the logs. If empty, a "VideoQA_logs" folder is made next to each video.
  4. Untick any checks you don't need (unticking "Audio language change" makes it much faster)
  5. Click "Start checking"
  6. When it finishes, click "Open last log"

HOW TO USE (command line / batch)
  VideoQA.exe "D:\Videos\episode1.mp4" "D:\Videos\Season2" -o "D:\QA_logs"
  VideoQA.exe "D:\Videos" --skip language gender

WHAT YOU GET for each video
  <video>_QA_<date>.log  - easy-to-read list of every defect with its time
  <video>_QA_<date>.csv  - same list for Excel

DEFECTS IT LOOKS FOR
  MISSING_IMAGE    black / blank screen
  FROZEN_VIDEO     picture stuck for 3+ seconds
  VIDEO_BLUR       picture much less sharp than the rest of the video
  FRAME_FRAGMENT   broken frame, blocky squares, green/pink patches, half-drawn frame, decoder errors
  NO_AUDIO         no sound track, sound ends before the picture, long silent parts, mouth moving without voice
  AUDIO_LAG        voice cuts out mid-speech, clicks/pops, stutter (sound repeats itself)
  AUDIO_OVERLAP    two voices talking at the same time
  DOUBLE_BGM       two background songs at once, or the same music layered twice (echo)
  LANGUAGE_CHANGE  speech switches to a different language than the main one
  AUDIO_DELAY      voice is early/late compared to the lips (needs a visible face)
  GENDER_MISMATCH  speaking face looks male/female but the voice sounds the opposite (needs a visible face)

"(please check)" means the robot is guessing - a person should confirm it.
Confidence: 1.00 = very sure, 0.50 = unsure.

Keep the "models" and "ffmpeg" folders next to VideoQA.exe.
