"""Support ``python -m jaxwind`` as an exact CLI fallback."""

from .cli import main


if __name__ == "__main__":
    raise SystemExit(main())
