from pathlib import Path


def run_record_path(run_directory: Path, name: str) -> Path:
    """Locate an artifact in current or pre-0.3 evaluation runs."""
    run = run_directory.expanduser().resolve()
    current = run / ".heragent" / name
    legacy = run / name
    return legacy if legacy.is_file() and not current.exists() else current
