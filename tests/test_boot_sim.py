#!/usr/bin/env python3
"""
================================================================================
Project     : River Kova — Household Chore Robot Control System
File        : tests/test_boot_sim.py
Purpose     : End-to-end smoke test — boots the entire system against the
              simulated body (units/kova_sim_profile.json), lets the control
              loop run for several seconds, and verifies a clean shutdown
              with no watchdog e-stop.
Author      : [Author Placeholder]
Version     : 1.0.0
Date        : 2026-06-12
License     : Proprietary — River Song AI (riversongai.com)
================================================================================
"""

import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SIM_PROFILE = REPO_ROOT / "units" / "kova_sim_profile.json"


@pytest.mark.timeout(60)
def test_full_boot_run_and_graceful_shutdown():
    """Boot the brain on the simulated body, run ~6 s, then SIGINT."""
    env = dict(os.environ, KOVA_PROFILE=str(SIM_PROFILE))

    proc = subprocess.Popen(
        [sys.executable, "-m", "core.main"],
        cwd=str(REPO_ROOT),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )

    try:
        time.sleep(6.0)
        proc.send_signal(signal.SIGINT)
        output, _ = proc.communicate(timeout=20)
    finally:
        if proc.poll() is None:
            proc.kill()
            output, _ = proc.communicate()

    assert "Boot complete" in output, f"Boot did not complete:\n{output}"
    assert "Entering main control loop" in output
    assert "WATCHDOG TIMEOUT" not in output, f"Watchdog fired on healthy sim:\n{output}"
    assert "Boot failed" not in output
    assert "Shutdown complete" in output, f"No graceful shutdown:\n{output}"
    assert proc.returncode == 0
