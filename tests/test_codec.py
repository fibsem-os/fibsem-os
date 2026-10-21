"""fibsem.codec on classes of its own, so each rule is checked apart from the
structures that will use it."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Annotated, ClassVar, List, Optional

import jsonschema
import numpy as np
import pytest
from pydantic import TypeAdapter, ValidationError

from fibsem.codec import (
    Array,
    NamedEnum,
    NotSaved,
    Tagged,
    Timestamp,
    decode,
    encode,
    saved,
)


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


@saved(
    keys={"settings": "image"},
    reads={"current": "beam_current"},
    absent={"rotation": 0.0},
)
@dataclass
class Saved:
    settings: Inner = field(default_factory=Inner)
    current: Optional[float] = None
    rotation: Optional[float] = None


@dataclass
class HoldsSaved:
    saved: Saved = field(default_factory=Saved)


def test_saved_writes_and_reads_a_field_under_its_key():
    value = Saved(Inner(2.0), current=1e-9, rotation=3.0)
    data = encode(value)
    assert data == {
        "image": {"x": 2.0, "beam": "ELECTRON"},
        "current": 1e-9,
        "rotation": 3.0,
    }
    assert decode(Saved, data) == value


def test_saved_reads_another_spelling_and_an_absent_value():
    value = decode(Saved, {"beam_current": 1e-9})
    assert value.current == 1e-9
    assert value.rotation == 0.0  # absent, not the dataclass default
    assert Saved().rotation is None


def test_saved_rules_hold_inside_another_structure():
    data = encode(HoldsSaved(Saved(Inner(1.0))))
    assert "image" in data["saved"]
    assert decode(HoldsSaved, {"saved": {"beam_current": 2.0}}).saved == Saved(
        current=2.0, rotation=0.0
    )


@saved(required=("x", "y"))
@dataclass
class Required:
    x: float = 0.0
    y: float = 0.0


def test_saved_refuses_a_required_key_that_is_absent():
    assert decode(Required, {"x": 1.0, "y": 2.0}) == Required(1.0, 2.0)
    with pytest.raises(ValidationError, match="missing y"):
        decode(Required, {"x": 1.0})
    assert Required() == Required(0.0, 0.0)  # a new instance still has defaults


@dataclass
class Strategy(Tagged, ABC):
    TYPE_KEY: ClassVar[str] = "name"
    SAVED_UNDER: ClassVar[str] = "config"


@dataclass
class Fast(Strategy):
    speed: float = 1.0


@dataclass
class Plan:
    drawing: Drawing = field(default_factory=Drawing)
    strategy: Strategy = field(default_factory=Fast)
    picture: Optional[Picture] = None
    when: Optional[Timestamp] = None
    scratch: Annotated[Optional[str], NotSaved] = None


def _schema(type_, mode):
    return TypeAdapter(type_).json_schema(mode=mode)


@pytest.mark.parametrize("mode", ["validation", "serialization"])
def test_a_tagged_base_has_one_schema_variant_per_subclass(mode):
    schema = _schema(Drawing, mode)
    variants = schema["$defs"]["Shape"]["oneOf"]
    assert [v["properties"]["type"] for v in variants] == [
        {"const": "Square"},
        {"const": "Disc"},
    ]
    assert all(v["required"][0] == "type" for v in variants)


@pytest.mark.parametrize("mode", ["validation", "serialization"])
def test_what_encode_writes_matches_the_schema(mode):
    plan = Plan(
        Drawing([Square(side=2.0), Disc()]),
        Fast(3.0),
        Picture(np.eye(2)),
        datetime(2026, 10, 21, tzinfo=timezone.utc),
        scratch="not saved",
    )
    data = encode(plan)
    jsonschema.validate(data, _schema(Plan, mode))
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(
            {"drawing": {"shapes": [{"type": "Hexagon"}]}}, _schema(Plan, mode)
        )


def test_a_tagged_base_saved_under_a_key_nests_the_fields_there():
    variant = _schema(Strategy, "serialization")["oneOf"][0]
    assert variant["properties"]["name"] == {"const": "Fast"}
    assert "config" in variant["properties"]


def test_an_array_and_a_timestamp_have_json_types():
    props = _schema(Plan, "serialization")["$defs"]["Picture"]["properties"]
    assert props["pixels"]["type"] == "array"
    when = _schema(Plan, "serialization")["properties"]["when"]["anyOf"][0]
    assert when["type"] == "string"


def test_a_field_not_saved_is_left_out_of_the_written_schema():
    assert "scratch" not in _schema(Plan, "serialization")["properties"]
