"""Entry point for source execution and portable builds."""
import sys

if __name__ == "__main__":
    if "--smoke-test" in sys.argv:
        from signal_scanner.smoke import main
    else:
        from signal_scanner.gui import main
    raise SystemExit(main())
