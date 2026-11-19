"""VideoQA - window app (double-click) and command line.

Command line:  VideoQA.exe video1.mp4 [folder ...] [-o OUTPUT_FOLDER]
Without arguments the window opens.
"""

from __future__ import annotations

import argparse
import multiprocessing
import os
import queue
import sys
import threading
from pathlib import Path


def run_cli(argv: list[str]) -> int:
    from videoqa.analyzer import ALL_CHECKS, analyze, collect_videos, write_report

    ap = argparse.ArgumentParser(prog="VideoQA")
    ap.add_argument("inputs", nargs="+", help="video files or folders")
    ap.add_argument("-o", "--out", help="folder for the log files (default: next to each video)")
    ap.add_argument("--skip", nargs="*", default=[], choices=list(ALL_CHECKS), help="checks to skip")
    a = ap.parse_args(argv)
    checks = set(ALL_CHECKS) - set(a.skip)
    vids = collect_videos(a.inputs)
    status = 0
    for v in vids:
        out_dir = Path(a.out) if a.out else Path(v).parent / "VideoQA_logs"
        try:
            d, notes = analyze(v, checks, lambda m: print(m, flush=True))
            log, _ = write_report(v, d, notes, out_dir)
            print(f"{len(d)} defect(s) -> {log}", flush=True)
        except Exception as e:
            print(f"FAILED {v}: {e}", flush=True)
            status = 1
    return status


def run_gui() -> None:
    import tkinter as tk
    from tkinter import filedialog, messagebox, ttk

    from videoqa import __version__
    from videoqa.analyzer import ALL_CHECKS, VIDEO_EXT, analyze, collect_videos, write_report

    root = tk.Tk()
    root.title(f"VideoQA {__version__} - video defect finder")
    root.geometry("900x640")
    root.minsize(720, 520)

    files: list[str] = []
    q: queue.Queue = queue.Queue()
    out_var = tk.StringVar(value="")
    running = {"on": False}
    last_log = {"path": None}

    top = ttk.Frame(root, padding=10)
    top.pack(fill="x")
    ttk.Label(top, text="1. Choose videos", font=("Segoe UI", 10, "bold")).grid(row=0, column=0, sticky="w")
    lst = tk.Listbox(top, height=6, selectmode="extended")
    lst.grid(row=1, column=0, columnspan=4, sticky="nsew", pady=4)
    top.columnconfigure(0, weight=1)

    def refresh():
        lst.delete(0, "end")
        for f in files:
            lst.insert("end", f)

    def add_files():
        exts = " ".join(f"*{e}" for e in sorted(VIDEO_EXT))
        for f in filedialog.askopenfilenames(title="Choose videos", filetypes=[("Videos", exts), ("All files", "*.*")]):
            if f not in files:
                files.append(f)
        refresh()

    def add_folder():
        d = filedialog.askdirectory(title="Choose a folder with videos")
        if d:
            for f in collect_videos([d]):
                if f not in files:
                    files.append(f)
            refresh()

    def remove_sel():
        for i in reversed(lst.curselection()):
            files.pop(i)
        refresh()

    btns = ttk.Frame(top)
    btns.grid(row=2, column=0, sticky="w")
    ttk.Button(btns, text="Add video files...", command=add_files).pack(side="left", padx=(0, 6))
    ttk.Button(btns, text="Add a folder...", command=add_folder).pack(side="left", padx=6)
    ttk.Button(btns, text="Remove selected", command=remove_sel).pack(side="left", padx=6)

    ttk.Label(top, text="2. Where to save the log files (empty = a 'VideoQA_logs' folder next to each video)",
              font=("Segoe UI", 10, "bold")).grid(row=3, column=0, sticky="w", pady=(12, 0))
    of = ttk.Frame(top)
    of.grid(row=4, column=0, columnspan=4, sticky="ew")
    ttk.Entry(of, textvariable=out_var).pack(side="left", fill="x", expand=True)
    ttk.Button(of, text="Browse...", command=lambda: out_var.set(filedialog.askdirectory() or out_var.get())).pack(side="left", padx=6)

    ttk.Label(top, text="3. What to check", font=("Segoe UI", 10, "bold")).grid(row=5, column=0, sticky="w", pady=(12, 0))
    cf = ttk.Frame(top)
    cf.grid(row=6, column=0, columnspan=4, sticky="w")
    check_vars = {}
    for i, (k, label) in enumerate(ALL_CHECKS.items()):
        v = tk.BooleanVar(value=True)
        check_vars[k] = v
        ttk.Checkbutton(cf, text=label, variable=v).grid(row=i // 3, column=i % 3, sticky="w", padx=(0, 18))

    act = ttk.Frame(root, padding=(10, 0))
    act.pack(fill="x")
    start_btn = ttk.Button(act, text="▶  Start checking")
    start_btn.pack(side="left")
    open_btn = ttk.Button(act, text="Open last log", state="disabled",
                          command=lambda: last_log["path"] and os.startfile(last_log["path"]))
    open_btn.pack(side="left", padx=6)
    folder_btn = ttk.Button(act, text="Open log folder", state="disabled",
                            command=lambda: last_log["path"] and os.startfile(str(Path(last_log["path"]).parent)))
    folder_btn.pack(side="left")
    bar = ttk.Progressbar(act, mode="determinate")
    bar.pack(side="left", fill="x", expand=True, padx=10)

    txt = tk.Text(root, wrap="word", font=("Consolas", 9), height=14)
    txt.pack(fill="both", expand=True, padx=10, pady=10)

    def log(msg: str):
        txt.insert("end", msg + "\n")
        txt.see("end")

    def worker(vids: list[str], checks: set[str], out: str):
        total_def = 0
        for i, v in enumerate(vids):
            q.put(("log", f"\n=== ({i + 1}/{len(vids)}) {v}"))
            q.put(("bar", 100 * i / len(vids)))
            try:
                d, notes = analyze(v, checks, lambda m: q.put(("log", m)))
                out_dir = Path(out) if out else Path(v).parent / "VideoQA_logs"
                lp, cp = write_report(v, d, notes, out_dir)
                total_def += len(d)
                q.put(("log", f"Found {len(d)} defect(s). Log: {lp}"))
                for dd in d[:200]:
                    q.put(("log", f"   {dd.time_label()}  {dd.kind}  {dd.message}"))
                q.put(("last", str(lp)))
            except Exception as e:
                q.put(("log", f"FAILED: {e}"))
        q.put(("bar", 100))
        q.put(("done", f"Finished {len(vids)} video(s), {total_def} defect(s) found."))

    def start():
        if running["on"]:
            return
        if not files:
            messagebox.showinfo("VideoQA", "Please add at least one video first.")
            return
        checks = {k for k, v in check_vars.items() if v.get()}
        if not checks:
            messagebox.showinfo("VideoQA", "Please tick at least one check.")
            return
        running["on"] = True
        start_btn.config(state="disabled")
        threading.Thread(target=worker, args=(list(files), checks, out_var.get().strip()), daemon=True).start()

    start_btn.config(command=start)

    def poll():
        try:
            while True:
                kind, val = q.get_nowait()
                if kind == "log":
                    log(val)
                elif kind == "bar":
                    bar["value"] = val
                elif kind == "last":
                    last_log["path"] = val
                    open_btn.config(state="normal")
                    folder_btn.config(state="normal")
                elif kind == "done":
                    log("\n" + val)
                    running["on"] = False
                    start_btn.config(state="normal")
        except queue.Empty:
            pass
        root.after(150, poll)

    log("Welcome! Add videos, then press 'Start checking'.\n"
        "Each video gets a .log file (easy to read) and a .csv file (open in Excel) with every defect and its time.\n"
        "Items marked '(please check)' are guesses - a person should confirm them.")
    poll()
    if os.environ.get("VIDEOQA_SMOKE"):  # used by the build test: open, then close by itself
        root.after(2500, root.destroy)
    root.mainloop()


def main() -> None:
    multiprocessing.freeze_support()
    if len(sys.argv) > 1:
        # a windowed exe has no console - attach to the parent one so CLI output is visible
        if getattr(sys, "frozen", False) and os.name == "nt":
            try:
                import ctypes

                if ctypes.windll.kernel32.AttachConsole(-1):
                    sys.stdout = open("CONOUT$", "w", encoding="utf-8", errors="replace")
                    sys.stderr = sys.stdout
            except Exception:
                pass
        sys.exit(run_cli(sys.argv[1:]))
    run_gui()


if __name__ == "__main__":
    main()
