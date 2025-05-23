import sys
import _pymarian


def main():
    code = _pymarian.main(sys.argv[1:])
    sys.exit(code)

if __name__ == '__main__':
    main()
