"""Same initial-stage screen, now with a generic syntax-only decoding policy."""
import argparse
from pathlib import Path
from probe_stages import main

if __name__ == "__main__":
    cli = argparse.ArgumentParser()
    cli.add_argument("--output", type=Path, required=True)
    main(cli.parse_args().output, grammar=True)
