"""fibsem.codec on classes of its own, so each rule is checked apart from the
structures that will use it."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import List, Optional

import numpy as np
import pytest
from pydantic import ValidationError

from fibsem.codec import Array, NamedEnum, Tagged, Timestamp, decode, encode


class Beam(NamedEnum, Enum):
    ELECTRON = 1
    ION = 2


class Transform(NamedEnum, Enum):
    NONE = None
    FLIP_X = "flip-x"


@dataclass
class Inner:
    x: float = 0.0
    beam: Beam = Beam.ELECTRON


@dataclass
class Outer:
    name: str = ""
    inner: Inner = field(default_factory=Inner)
    count: int = 0
    flag: bool = False
    transform: Transform = Transform.NONE
    when: Optional[Timestamp] = None


@dataclass
class Shape(Tagged, ABC):
    rotation: float = 0.0

    @abstractmethod
    def area(self) -> float: ...


@dataclass
class Square(Shape):
    side: float = 1.0

    def area(self) -> float:
        return self.side**2


@dataclass
class Disc(Shape):
    radius: float = 1.0

    def area(self) -> float:
        return np.pi * self.radius**2


@dataclass
class Drawing:
    shapes: List[Shape] = field(default_factory=list)


@dataclass
class Picture:
    pixels: Array


def test_round_trip_by_field_names():
    value = Outer("a", Inner(2.0, Beam.ION), 3, True)
    data = encode(value)
    assert data["inner"] == {"x": 2.0, "beam": "ION"}
    assert decode(Outer, data) == value


def test_a_missing_key_takes_the_default():
    assert decode(Outer, {"name": "a"}) == Outer("a")


def test_unknown_keys_are_ignored_unless_strict():
    data = {"name": "a", "inner": {"x": 1.0, "zoom": 2}, "colour": "red"}
    assert decode(Outer, data) == Outer("a", Inner(1.0))
    with pytest.raises(ValidationError) as caught:
        decode(Outer, data, strict=True)
    paths = {".".join(map(str, error["loc"])) for error in caught.value.errors()}
    assert paths == {"inner.zoom", "colour"}


def test_numbers_sent_as_text_read_as_numbers():
    value = decode(Outer, {"count": "4.0", "flag": "true", "inner": {"x": "2.5"}})
    assert (value.count, value.flag, value.inner.x) == (4, True, 2.5)


def test_numpy_scalars_encode_as_plain_values():
    data = encode(
        Outer(count=np.int64(3), flag=np.bool_(True), inner=Inner(np.float32(1.5)))
    )
    assert data["count"] == 3 and type(data["count"]) is int
    assert data["flag"] is True
    assert data["inner"]["x"] == 1.5 and type(data["inner"]["x"]) is float


@pytest.mark.parametrize("sent", ["ION", "ion", 2, "2", Beam.ION])
def test_an_enum_reads_by_name_or_value(sent):
    assert decode(Inner, {"beam": sent}).beam is Beam.ION


def test_an_enum_whose_value_is_none_reads_from_null():
    assert decode(Outer, {"transform": None}).transform is Transform.NONE
    assert decode(Outer, {"transform": "flip-x"}).transform is Transform.FLIP_X
    assert encode(Outer(transform=Transform.FLIP_X))["transform"] == "FLIP_X"


def test_an_unknown_enum_name_is_refused_with_the_choices():
    with pytest.raises(ValidationError, match="ELECTRON, ION"):
        decode(Inner, {"beam": "PLASMA"})


def test_a_tagged_base_writes_and_reads_the_subclass():
    drawing = Drawing([Square(10.0, side=2.0), Disc(radius=3.0)])
    data = encode(drawing)
    assert data["shapes"] == [
        {"type": "Square", "rotation": 10.0, "side": 2.0},
        {"type": "Disc", "rotation": 0.0, "radius": 3.0},
    ]
    assert decode(Drawing, data) == drawing


def test_a_subclass_named_directly_has_no_type_key():
    assert encode(Square(side=2.0)) == {"rotation": 0.0, "side": 2.0}
    assert decode(Square, {"side": 2.0}) == Square(side=2.0)


def test_a_tagged_base_refuses_a_missing_or_unknown_type():
    with pytest.raises(ValidationError, match="'type' key"):
        decode(Drawing, {"shapes": [{"side": 1.0}]})
    with pytest.raises(ValidationError, match="no Shape named 'Hexagon'"):
        decode(Drawing, {"shapes": [{"type": "Hexagon"}]})


def test_strict_reaches_inside_a_tagged_subclass():
    data = {"shapes": [{"type": "Square", "side": 1.0, "colour": "red"}]}
    assert decode(Drawing, data) == Drawing([Square(side=1.0)])
    with pytest.raises(ValidationError, match="colour"):
        decode(Drawing, data, strict=True)


def test_an_array_is_written_as_lists():
    data = encode(Picture(np.eye(2)))
    assert data == {"pixels": [[1.0, 0.0], [0.0, 1.0]]}
    back = decode(Picture, data)
    np.testing.assert_array_equal(back.pixels, np.eye(2))


def test_a_timestamp_keeps_its_offset():
    when = datetime(2026, 9, 13, 21, 19, 40, tzinfo=timezone(timedelta(hours=-6)))
    data = encode(Outer(when=when))
    assert data["when"] == "2026-09-13T21:19:40-06:00"
    assert decode(Outer, data).when == when
    assert decode(Outer, data).when.utcoffset() == timedelta(hours=-6)


def test_a_timestamp_reads_the_older_forms():
    assert decode(Outer, {"when": 0.0}).when.timestamp() == 0.0
    legacy = decode(Outer, {"when": "07/16/2026 11:07:27"}).when
    assert legacy == datetime(2026, 7, 16, 11, 7, 27)
    assert legacy.tzinfo is None  # a naive time stays naive
    assert decode(Outer, {"when": None}).when is None
