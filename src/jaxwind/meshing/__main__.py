"""Support ``python -m jaxwind.meshing`` without package installation."""

from .cli import main


if __name__ == "__main__":
    raise SystemExit(main())
