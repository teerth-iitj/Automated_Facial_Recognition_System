"""Dependency checker and auto-installer for the facial-recognition attendance system.

Usage:
    python req.py            # check and install missing packages
    python req.py --check    # check only, do not install

The module also exposes check_all() so run_all.py can call it programmatically.
"""

import importlib
import os
import subprocess
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# ANSI colour helpers (work on Windows 10+ and most terminals)
# ---------------------------------------------------------------------------
_RESET  = "\033[0m"
_GREEN  = "\033[92m"
_YELLOW = "\033[93m"
_RED    = "\033[91m"
_CYAN   = "\033[96m"
_BOLD   = "\033[1m"


def _c(text, *codes):
    return "".join(codes) + str(text) + _RESET


# ---------------------------------------------------------------------------
# Package table: (import_name, pip_name)
# ---------------------------------------------------------------------------
REQUIRED = [
    ("face_recognition", "face_recognition"),
    ("flask",            "flask"),
    ("flask_cors",       "flask-cors"),
    ("werkzeug",         "werkzeug"),
    ("cv2",              "opencv-python"),
    ("numpy",            "numpy"),
    ("PIL",              "Pillow"),
    ("joblib",           "joblib"),
    ("scipy",            "scipy"),
]

ENCODINGS_FILE = Path("face_encodings.pkl")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _check_python_version():
    major, minor = sys.version_info[:2]
    ok = (major, minor) >= (3, 8)
    tag = _c("✓", _GREEN) if ok else _c("✗", _RED)
    status = f"Python {major}.{minor}.{sys.version_info[2]}"
    if not ok:
        status += "  ← requires >= 3.8"
    print(f"  {tag}  {status}")
    return ok


def _try_import(import_name):
    """Return True if the module can be imported."""
    try:
        importlib.import_module(import_name)
        return True
    except ImportError:
        return False


def _pip_install(pip_name):
    """Run pip install and return (success, output)."""
    result = subprocess.run(
        [sys.executable, "-m", "pip", "install", pip_name, "--quiet"],
        capture_output=True,
        text=True,
    )
    return result.returncode == 0, result.stderr.strip()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def check_all(auto_install=True):
    """Check all requirements, optionally install missing ones.

    Returns True if everything is satisfied after the check/install pass.
    """
    print()
    print(_c(" Facial-Recognition Attendance System — Dependency Check ", _BOLD, _CYAN))
    print("─" * 60)

    # Python version
    print(_c("Python version:", _BOLD))
    py_ok = _check_python_version()
    print()

    # Packages
    print(_c("Required packages:", _BOLD))
    missing_pip = []
    results = {}

    for import_name, pip_name in REQUIRED:
        found = _try_import(import_name)
        results[pip_name] = found
        tag = _c("✓", _GREEN) if found else _c("✗", _RED)
        label = f"{pip_name:<20} (import {import_name})"
        print(f"  {tag}  {label}  {'installed' if found else _c('MISSING', _RED)}")
        if not found:
            missing_pip.append(pip_name)

    print()

    # Auto-install
    install_failed = []
    if missing_pip and auto_install:
        print(_c(f"Installing {len(missing_pip)} missing package(s)…", _YELLOW))
        for pip_name in missing_pip:
            print(f"  → pip install {pip_name} … ", end="", flush=True)
            ok, err = _pip_install(pip_name)
            if ok:
                # Verify the import now works
                import_name = next(
                    imp for imp, pkg in REQUIRED if pkg == pip_name
                )
                ok = _try_import(import_name)

            if ok:
                print(_c("done", _GREEN))
                results[pip_name] = True
            else:
                print(_c("FAILED", _RED))
                if err:
                    print(f"     {_c(err[:200], _RED)}")
                install_failed.append(pip_name)
        print()

    # face_encodings.pkl presence check
    print(_c("Gallery file:", _BOLD))
    if ENCODINGS_FILE.exists():
        size_kb = ENCODINGS_FILE.stat().st_size // 1024
        print(f"  {_c('✓', _GREEN)}  {ENCODINGS_FILE} ({size_kb} KB)")
    else:
        print(
            f"  {_c('⚠', _YELLOW)}  {ENCODINGS_FILE} not found — "
            f"run {_c('python train_model.py', _CYAN)} to build the gallery"
        )
    print()

    # Final summary
    all_ok = py_ok and not install_failed and all(results.values())
    print("─" * 60)
    if all_ok:
        print(_c("✓ All requirements satisfied", _GREEN, _BOLD))
    else:
        print(_c("✗ Some requirements could not be satisfied:", _RED, _BOLD))
        if not py_ok:
            print(f"  • Python version too old (need >= 3.8)")
        for pkg in install_failed:
            print(f"  • {pkg} failed to install — try:  pip install {pkg}")
        still_missing = [pkg for pkg, ok in results.items() if not ok]
        for pkg in still_missing:
            if pkg not in install_failed:
                print(f"  • {pkg} still not importable")
    print()

    return all_ok


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

def main():
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="Check only; do not attempt to install missing packages.",
    )
    args = parser.parse_args()

    ok = check_all(auto_install=not args.check)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
