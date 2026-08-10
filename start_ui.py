import sys
import argparse
from pathlib import Path

from PyQt5.QtWidgets import QApplication


from ui.main_window import MainWindow


def parse_args(argv):
    parser = argparse.ArgumentParser(description="Start CSP Analysis Tool UI.")
    parser.add_argument(
        "raw_folder",
        nargs="?",
        help="Folder with raw data files. Defaults to data/{project}/raw/{stage}/{session}.",
    )
    args = parser.parse_args(argv)

    if args.raw_folder:
        raw_folder = Path(args.raw_folder).expanduser()
        if not raw_folder.is_dir():
            parser.error(f"raw_folder does not exist or is not a folder: {raw_folder}")
        args.raw_folder = str(raw_folder.resolve())

    return args


if __name__ == "__main__":
    args = parse_args(sys.argv[1:])
    app = QApplication([sys.argv[0]])
    window = MainWindow(raw_data_folder=args.raw_folder)
    
    sys.exit(app.exec_())
