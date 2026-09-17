"""Read Racz et al. DXEX v5 PDA measurements without inventing missing components.

Coordinates and units come from each file header, never file ordering. LDA4 is
radial on the X traverse and tangential on the Y traverse; both are stored in
the original signed instrument frame. Event samples are not automatically
unbiased inlet number/mass flux samples. No measurement correction is applied.
"""

import io
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

COLUMNS = (
    "Row#",
    "AT [ms]",
    "TT [us]",
    "LDA1 [m/s]",
    "LDA4 [m/s]",
    "U12 [deg]",
    "U13 [deg]",
    "D [um]",
)


@dataclass(frozen=True)
class PDAHeader:
    point: int
    position_m: tuple[float, float, float]
    traverse: str
    acquisition_label: str

    @property
    def role(self):
        return (
            "source"
            if np.isclose(self.position_m[2], 0.020, atol=1e-12, rtol=0)
            else "holdout"
        )

    @property
    def transverse_component(self):
        return (
            "signed_radial_instrument"
            if self.traverse == "X"
            else "signed_tangential_instrument"
        )


@dataclass(frozen=True)
class PDAStation:
    header: PDAHeader
    # SI: row ID, arrival s, transit s, axial m/s, transverse m/s,
    # phase12 deg, phase13 deg, diameter m.
    samples: np.ndarray


def read_header(path):
    with Path(path).open(encoding="utf-8-sig") as f:
        lines = [f.readline().strip() for _ in range(6)]
    if lines[0] != "DXEX v5":
        raise ValueError("Unsupported PDA export format")
    point, *coordinates = lines[3].split(";")
    if len(coordinates) != 3:
        raise ValueError("Expected point;X;Y;Z in header")
    xyz = []
    for coordinate in coordinates:
        if not coordinate.endswith(" mm"):
            raise ValueError("Coordinates must explicitly be in mm")
        xyz.append(float(coordinate[:-3].replace(",", ".")) * 1e-3)
    match = re.fullmatch(r"([XY])_(20|40|60)mm(?:_center|_2)?", lines[4])
    if match is None or not np.isclose(
        xyz[2], int(match[2]) * 1e-3, atol=1e-12, rtol=0
    ):
        raise ValueError("Traverse label and axial coordinate disagree")
    if xyz[1 if match[1] == "X" else 0] != 0:
        raise ValueError("Point is outside its declared traverse")
    columns = tuple(s.strip('"') for s in lines[5].split("\t"))
    if columns != COLUMNS:
        raise ValueError(f"Unexpected measurement columns: {columns}")
    return PDAHeader(int(point), tuple(xyz), match[1], lines[4])


def read_station(path, *, source_only=False):
    header = read_header(path)
    if source_only and header.role != "source":
        raise ValueError("Held-out station cannot be used as an inlet")
    # Reject unexpected columns; preserve outliers/invalid physical values for
    # an explicit audit instead of silently deleting them from the experiment.
    with Path(path).open(encoding="utf-8-sig") as f:
        for _ in range(6):
            next(f)
        samples = np.loadtxt(io.StringIO(f.read().replace(",", ".")), ndmin=2)
    if samples.shape[1] != 8 or len(samples) == 0:
        raise ValueError("Expected nonempty eight-column droplet records")
    samples[:, 1] *= 1e-3
    samples[:, 2] *= 1e-6
    samples[:, 7] *= 1e-6
    return PDAStation(header, samples)


def source_files(directory):
    """Return only the 20 mm files; the centre repeats remain separate."""
    return [
        p
        for p in sorted(Path(directory).glob("*.0*.txt"))
        if read_header(p).role == "source"
    ]
