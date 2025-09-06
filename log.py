import logging
from datetime import datetime
from pathlib import Path

LOG_FILE = Path("runs/training.log")
LOG_FILE.parent.mkdir(parents=True, exist_ok=True)


def log(message: str, prefix: str = "INFO", sep: bool = True) -> None:
    
    """Simple logger: prints to console and writes to file.

    Args:
        message: The log message.
        prefix: Short tag like INFO/WARN/ERR/etc.
        sep: If True, appends a separator line after the message.
    """
    
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{timestamp}] {prefix}: {message}"

    # Print to console
    print(line)

    # Append to file
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(line + "\n")
        if sep:
            f.write("=" * 80 + "\n")

    # Print separator to console as well
    if sep:
        print("=" * 80)