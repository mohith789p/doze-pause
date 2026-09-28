"""
Demo launcher — opens a video in VLC, then starts the drowsiness detector.
Usage:
    python demo.py                        # uses built-in test pattern
    python demo.py path/to/video.mp4      # your own video
"""

import subprocess
import sys
import time
import os
import shutil
from pathlib import Path

DEMO_VIDEO = None

SAMPLE_VIDEOS = [
    Path(r"C:\Users\Public\Videos\Sample Videos"),
    Path(r"C:\Windows\Web"),
]


def _vlc_from_registry():
    """Read VLC install path from the Windows Registry."""
    try:
        import winreg
        reg_keys = [
            # 64-bit VLC on 64-bit Windows
            (winreg.HKEY_LOCAL_MACHINE, r"SOFTWARE\VideoLAN\VLC"),
            # 32-bit VLC on 64-bit Windows (WOW6432Node)
            (winreg.HKEY_LOCAL_MACHINE, r"SOFTWARE\WOW6432Node\VideoLAN\VLC"),
            # Per-user install
            (winreg.HKEY_CURRENT_USER,  r"SOFTWARE\VideoLAN\VLC"),
        ]
        for hive, subkey in reg_keys:
            try:
                with winreg.OpenKey(hive, subkey) as key:
                    # The default value (empty string) holds the path to vlc.exe
                    vlc_path, _ = winreg.QueryValueEx(key, "")
                    if vlc_path and Path(vlc_path).exists():
                        return vlc_path
            except FileNotFoundError:
                continue
    except ImportError:
        pass  # not on Windows
    return None


def _vlc_from_env_drives():
    """Scan all available drive roots for VideoLAN\VLC\vlc.exe."""
    import string
    drives = [
        Path(f"{d}:\\")
        for d in string.ascii_uppercase
        if Path(f"{d}:\\").exists()
    ]
    search_dirs = ["Program Files", "Program Files (x86)", "Programs", "Apps", "Software"]
    for drive in drives:
        for d in search_dirs:
            candidate = drive / d / "VideoLAN" / "VLC" / "vlc.exe"
            if candidate.exists():
                return str(candidate)
    return None


def find_vlc():
    """
    Dynamically locate the VLC executable — no hardcoded paths.

    Strategy (in order):
      1. Windows Registry  — most reliable; VLC always registers itself here.
      2. System PATH       — works if VLC was added to PATH or installed via winget/choco.
      3. Drive scan        — walks all drive roots for common install folder patterns.
    """
    # 1. Registry
    vlc = _vlc_from_registry()
    if vlc:
        print(f"   Found VLC via registry: {vlc}")
        return vlc

    # 2. PATH
    vlc = shutil.which("vlc") or shutil.which("vlc.exe")
    if vlc:
        print(f"   Found VLC on PATH: {vlc}")
        return vlc

    # 3. Drive scan
    vlc = _vlc_from_env_drives()
    if vlc:
        print(f"   Found VLC via drive scan: {vlc}")
        return vlc

    return None


def launch_vlc(video_path=None):
    """Launch VLC with a video file or a test signal."""
    vlc_exe = find_vlc()
    if not vlc_exe:
        print("⚠  VLC not found. Please install VLC or open any video player manually.")
        print("   The detector will still monitor and close whatever video player is running.")
        return None

    cmd = [vlc_exe]
    if video_path and Path(video_path).exists():
        cmd.append(str(video_path))
        print(f"▶  Launching VLC with: {video_path}")
    else:
        # Use VLC's built-in colour bars / test pattern as a demo
        cmd += ["--demux", "rawvideo", "--no-video-title-show"]
        print("▶  Launching VLC (no video file given — open a video manually in VLC)")
        # Just launch VLC without a file so user can open something
        cmd = [vlc_exe]

    try:
        proc = subprocess.Popen(cmd)
        print(f"   VLC PID: {proc.pid}")
        return proc
    except Exception as e:
        print(f"⚠  Could not launch VLC: {e}")
        return None


def main():
    video_path = sys.argv[1] if len(sys.argv) > 1 else DEMO_VIDEO

    print("=" * 60)
    print("  DROWSINESS DETECTION DEMO")
    print("=" * 60)

    # Step 1: Launch VLC
    vlc_proc = launch_vlc(video_path)
    if vlc_proc:
        print("   Waiting 2s for VLC to open...\n")
        time.sleep(2)

    # Step 2: Start detector
    print("Starting drowsiness detector...\n")
    import detector
    detector.run_detector()


if __name__ == "__main__":
    main()
