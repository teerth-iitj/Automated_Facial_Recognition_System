"""Main orchestrator for the facial-recognition attendance system.

Usage examples:
    python run_all.py                        # start Flask server (default)
    python run_all.py --run --port 5443 --https
    python run_all.py --train
    python run_all.py --evaluate --threshold 0.45
    python run_all.py --all --port 5000
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# ANSI helpers
# ---------------------------------------------------------------------------
_RESET  = "\033[0m"
_GREEN  = "\033[92m"
_YELLOW = "\033[93m"
_RED    = "\033[91m"
_CYAN   = "\033[96m"
_BLUE   = "\033[94m"
_BOLD   = "\033[1m"


def _c(text, *codes):
    return "".join(codes) + str(text) + _RESET


def _header(emoji, title):
    bar = "─" * 60
    print()
    print(bar)
    print(f"  {emoji}  {_c(title, _BOLD, _CYAN)}")
    print(bar)


# ---------------------------------------------------------------------------
# Banner
# ---------------------------------------------------------------------------

def _print_banner(port, https_enabled):
    scheme = "https" if https_enabled else "http"
    url = f"{scheme}://localhost:{port}"
    print()
    print(_c("╔══════════════════════════════════════════════════════╗", _BLUE))
    print(_c("║   Facial-Recognition Attendance System               ║", _BLUE))
    print(_c("╚══════════════════════════════════════════════════════╝", _RESET))
    print(f"  Port  : {_c(port, _BOLD)}")
    print(f"  HTTPS : {_c('enabled' if https_enabled else 'disabled', _GREEN if https_enabled else _YELLOW)}")
    print(f"  URL   : {_c(url, _CYAN, _BOLD)}")
    print()


# ---------------------------------------------------------------------------
# Step runners
# ---------------------------------------------------------------------------

def _run_subprocess(cmd, step_name):
    """Run *cmd* as a subprocess; return True on success."""
    print(_c(f"  Running: {' '.join(str(c) for c in cmd)}", _YELLOW))
    try:
        result = subprocess.run(cmd, check=False)
        if result.returncode != 0:
            print(
                _c(
                    f"  ✗ {step_name} exited with code {result.returncode}",
                    _RED,
                    _BOLD,
                )
            )
            return False
        print(_c(f"  ✓ {step_name} completed successfully", _GREEN))
        return True
    except FileNotFoundError as exc:
        print(_c(f"  ✗ Could not start {step_name}: {exc}", _RED))
        return False


def step_train(dataset, output, workers, jitters):
    _header("🔧", "Training model…")
    cmd = [
        sys.executable, "train_model.py",
        "--dataset", dataset,
        "--output", output,
        "--workers", str(workers),
        "--jitters", str(jitters),
    ]
    return _run_subprocess(cmd, "train_model.py")


def step_evaluate(dataset, encodings, threshold):
    _header("📊", "Evaluating model…")
    cmd = [
        sys.executable, "evaluate_model.py",
        "--dataset", dataset,
        "--encodings", encodings,
        "--threshold", str(threshold),
    ]
    print(_c(f"  Running: {' '.join(str(c) for c in cmd)}", _YELLOW))
    try:
        result = subprocess.run(
            cmd,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
    except FileNotFoundError as exc:
        print(_c(f"  ✗ Could not start evaluate_model.py: {exc}", _RED))
        return False

    report_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "evaluation_split.log"
    )
    with open(report_path, "w", encoding="utf-8", newline="\n") as report_file:
        report_file.write(result.stdout)

    if result.stdout:
        print(result.stdout, end="" if result.stdout.endswith("\n") else "\n")
    print(f"  Evaluation report saved to: {report_path}")
    if result.returncode != 0:
        print(
            _c(
                f"  ✗ evaluate_model.py exited with code {result.returncode}",
                _RED,
                _BOLD,
            )
        )
        return False
    print(_c("  ✓ evaluate_model.py completed successfully", _GREEN))
    return True


def step_run(port, https_enabled, encodings):
    _header("🚀", "Starting Flask server…")
    _print_banner(port, https_enabled)
    env = os.environ.copy()
    env["FLASK_PORT"] = str(port)
    env["FLASK_ENCODINGS"] = encodings
    if https_enabled:
        env["FLASK_HTTPS"] = "1"
    else:
        env.pop("FLASK_HTTPS", None)

    cmd = [sys.executable, "app.py"]
    print(_c(f"  Running: {' '.join(cmd)}", _YELLOW))
    print(_c("  Press Ctrl+C to stop the server.\n", _YELLOW))
    try:
        proc = subprocess.run(cmd, env=env, check=False)
        if proc.returncode not in (0, -2):   # -2 = SIGINT on POSIX
            print(_c(f"  ✗ Server exited with code {proc.returncode}", _RED))
            return False
    except KeyboardInterrupt:
        # Ctrl+C propagates to child on most platforms; we just catch the
        # Python-side re-raise so we can print a tidy goodbye.
        pass
    print()
    print(_c("  Server stopped.", _YELLOW))
    return True


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # --- mode flags (at least one of these or default to --run) ---
    mode = parser.add_argument_group("Mode (default: --run)")
    mode.add_argument("--train",    action="store_true", help="Build face gallery")
    mode.add_argument("--evaluate", action="store_true", help="Evaluate the gallery on held-out photos")
    mode.add_argument("--run",      action="store_true", help="Start the Flask web server")
    mode.add_argument("--all",      action="store_true", help="Train → evaluate → run")

    # --- server options ---
    srv = parser.add_argument_group("Server options")
    srv.add_argument("--port",  type=int, default=int(os.getenv("FLASK_PORT", "5000")),
                     help="Port for the Flask server (default: 5000 or $FLASK_PORT)")
    srv.add_argument("--https", action="store_true",
                     help="Enable HTTPS (sets FLASK_HTTPS env var for app.py)")

    # --- training options ---
    trn = parser.add_argument_group("Training options")
    trn.add_argument("--dataset",  default="Dataset",           help="Dataset folder (default: Dataset)")
    trn.add_argument("--output",   default="face_encodings.pkl", help="Gallery output file")
    trn.add_argument("--workers",  type=int, default=min(4, os.cpu_count() or 1),
                     help="Parallel workers for encoding (default: min(4, cpu_count))")
    trn.add_argument("--jitters",  type=int, default=1,
                     help="Jitter count for face_encodings (default: 1)")

    # --- evaluation options ---
    evl = parser.add_argument_group("Evaluation options")
    evl.add_argument("--threshold", type=float, default=0.45,
                     help="Face distance threshold (default: 0.45)")

    # --- misc ---
    parser.add_argument("--skip-req-check", action="store_true",
                        help="Skip the dependency check at startup")

    return parser


def main():
    parser = _build_parser()
    args = parser.parse_args()

    # Default to --run when no mode flag is set
    if not (args.train or args.evaluate or args.run or args.all):
        args.run = True

    # ------------------------------------------------------------------
    # Dependency check
    # ------------------------------------------------------------------
    if not args.skip_req_check:
        try:
            import req  # noqa: PLC0415
            ok = req.check_all(auto_install=True)
            if not ok:
                print(
                    _c(
                        "\n  Dependency check failed. Fix the issues above and retry,\n"
                        "  or run with --skip-req-check to bypass.\n",
                        _RED,
                    )
                )
                sys.exit(1)
        except ImportError:
            print(
                _c(
                    "  ⚠ req.py not found — skipping dependency check.",
                    _YELLOW,
                )
            )

    # ------------------------------------------------------------------
    # Execute requested steps
    # ------------------------------------------------------------------
    success = True

    try:
        if args.all or args.train:
            ok = step_train(args.dataset, args.output, args.workers, args.jitters)
            if not ok:
                success = False
                if not args.all:
                    sys.exit(1)

        if args.all or args.evaluate:
            ok = step_evaluate(args.dataset, args.output, args.threshold)
            if not ok:
                success = False
                if not args.all:
                    sys.exit(1)

        if args.all or args.run:
            if success or args.run:   # always try to run if --run was explicit
                ok = step_run(args.port, args.https, args.output)
                if not ok:
                    success = False

    except KeyboardInterrupt:
        print()
        print(_c("\n  Interrupted by user. Goodbye! 👋", _YELLOW, _BOLD))

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
