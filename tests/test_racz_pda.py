"""Guard units, velocity orientation and the measured source/holdout split."""

import numpy as np
import pytest

from tools.racz_pda import read_header, read_station, source_files


def write_file(path, *, traverse="X", z=20, x=-2, y=0, records=None):
    text = (
        "DXEX v5\nsource.lda\n5:32:17 AM\n"
        f"1;{x},00 mm;{y},00 mm;{z},00 mm\n{traverse}_{z}mm\n"
        '"Row#"\t"AT [ms]"\t"TT [us]"\t"LDA1 [m/s]"\t"LDA4 [m/s]"\t"U12 [deg]"\t"U13 [deg]"\t"D [um]"\t\n'
    )
    text += (
        records if records is not None else "1\t10,0\t2,0\t20,0\t-3,0\t1,0\t2,0\t5,0\n"
    )
    path.write_text(text)
    return path


def test_units_and_joint_velocity_are_preserved(tmp_path):
    station = read_station(write_file(tmp_path / "source.txt"), source_only=True)
    assert station.header.position_m == (-0.002, 0.0, 0.02)
    np.testing.assert_allclose(station.samples[0], [1, 0.01, 2e-6, 20, -3, 1, 2, 5e-6])
    assert station.header.transverse_component == "signed_radial_instrument"


def test_y_traverse_is_tangential_and_centre_repeats_remain_separate(tmp_path):
    write_file(tmp_path / "source.000001.txt", x=0)
    p = write_file(tmp_path / "source.000002.txt", x=0, traverse="Y")
    assert read_header(p).transverse_component == "signed_tangential_instrument"
    assert len(source_files(tmp_path)) == 2


def test_source_loader_rejects_holdout_before_reading_records(tmp_path):
    p = write_file(
        tmp_path / "source.000001.txt", z=40, records="malformed downstream records\n"
    )
    assert source_files(tmp_path) == []
    with pytest.raises(ValueError, match="Held-out"):
        read_station(p, source_only=True)


def test_bad_header_is_rejected_instead_of_guessing_coordinates(tmp_path):
    p = write_file(tmp_path / "source.txt")
    p.write_text(p.read_text().replace("X_20mm", "X_60mm"))
    with pytest.raises(ValueError, match="disagree"):
        read_header(p)


def test_unphysical_events_are_retained_for_explicit_audit(tmp_path):
    p = write_file(tmp_path / "source.txt", records="1\t10\t0\t20\t-3\t1\t2\t-5\n")
    station = read_station(p)
    assert station.samples[0, 2] == 0
    assert station.samples[0, 7] < 0


@pytest.mark.parametrize("suffix", ["_center", "_2"])
def test_acquisition_segment_labels_do_not_override_coordinates(tmp_path, suffix):
    p = write_file(tmp_path / "source.txt", x=-1)
    p.write_text(p.read_text().replace("X_20mm", "X_20mm" + suffix))
    h = read_header(p)
    assert h.position_m == (-0.001, 0, 0.020)
    assert h.acquisition_label == "X_20mm" + suffix
    assert h.role == "source"
