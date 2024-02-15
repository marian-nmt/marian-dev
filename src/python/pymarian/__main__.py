import argparse

from pymarian import __version__


def parse_args():
    parser = argparse.ArgumentParser(
        prog='pymarian',
        description="Python wrapper for Marian NMT",
        epilog='URL: https://github.com/marian-nmt/marian-dev',
    )
    parser.add_argument('--version', '-v', action='version', version=__version__)
    return parser.parse_args()


def main():
    args = parse_args()
    # prints version for -v/-version option.
    # no other options are currently supported. Space left/intended for future use.


if __name__ == '__main__':
    main()
