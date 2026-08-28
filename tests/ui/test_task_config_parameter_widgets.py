"""Tests for the generic task-parameters form's field dispatch.

The form builds an editor per dataclass field from the field's type. Its old
fallback for a type it had no editor for was a plain QLineEdit holding
``str(value)`` -- on the assumption that an unknown type round-trips through its
own text. Almost nothing does: for a nested dataclass the box shows a ``repr``
that cannot be parsed back, and the form then wrote that string onto the config,
replacing the object. It needed no typing -- a QLineEdit emits ``editingFinished``
on focus loss -- so tabbing past the field was enough, and the damage only
surfaced later, when the task ran or the experiment was saved.

Built from a synthetic config rather than a real task's: the point is the type
dispatch, and a config written here states exactly which types are under test
instead of depending on which fields a production task happens to carry.
"""

import pytest

pytest.importorskip("PyQt5")  # CI installs .[test] only; the UI extra is deliberate

from dataclasses import dataclass, field  # noqa: E402
from enum import Enum  # noqa: E402
from typing import ClassVar, List, Literal, Optional  # noqa: E402

from PyQt5.QtCore import Qt  # noqa: E402
from PyQt5.QtWidgets import (  # noqa: E402
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QLineEdit,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from fibsem.applications.autolamella.structures import (
    AutoLamellaTaskConfig,  # noqa: E402
)
from fibsem.applications.autolamella.ui.autolamella_task_config_widget import (  # noqa: E402
    TASK_FORM_DEFAULTS,
    AutoLamellaTaskConfigWidget,
    AutoLamellaTaskParametersConfigWidget,
)
from fibsem.ui.widgets.custom_widgets import (  # noqa: E402
    IntegerValueSpinBox,
    ValueComboBox,
    ValueSpinBox,
)


def _control(form, field_name):
    """The Control adapter the shared builder produced for *field_name*.

    The form used to hold a ParameterWidget per field; it now holds a row whose
    `control` carries the widget and its read/write adapters (FIB-526).
    """
    return next(row.control for row in form._rows if row.field == field_name)


class Flavour(Enum):
    SWEET = "sweet"
    SOUR = "sour"


@dataclass
class NestedSettings:
    """Stands in for ZParameters/ChannelSettings -- a dataclass the form cannot edit."""

    depth: float = 1.5
    label: str = "nested"


@dataclass
class SampleTaskConfig(AutoLamellaTaskConfig):
    """One field per dispatch branch the form has to get right."""

    task_type: ClassVar[str] = "SAMPLE_TASK"
    display_name: ClassVar[str] = "Sample Task"

    a_bool: bool = True
    an_int: int = 3
    a_float: float = 2.5
    a_string: str = "hello"
    an_enum: Flavour = Flavour.SWEET
    a_scalar_list: List[str] = field(default_factory=lambda: ["x", "y"])
    a_literal: Literal["cells", "hpf", "other"] = "hpf"
    an_optional_float: Optional[float] = 1.0
    with_items: str = field(default="b", metadata={"items": ["a", "b", "c"]})
    # The two that used to be silently destroyed.
    nested: NestedSettings = field(default_factory=NestedSettings)
    nested_list: List[NestedSettings] = field(default_factory=lambda: [NestedSettings()])


# Both forms in the module share one dispatch; run every test against both so a
# fix landing in only one of them fails here rather than in a user's protocol.
FORMS = [AutoLamellaTaskParametersConfigWidget, AutoLamellaTaskConfigWidget]
FORM_IDS = ["parameters_form", "config_form"]


def _build(form_cls, config):
    widget = form_cls(task_config=config)
    return widget


@pytest.mark.parametrize("form_cls", FORMS, ids=FORM_IDS)
@pytest.mark.parametrize(
    "field_name,expected_widget",
    [
        ("a_bool", QCheckBox),
        ("an_int", IntegerValueSpinBox),
        ("a_float", ValueSpinBox),
        ("a_string", QLineEdit),
        ("an_enum", ValueComboBox),
        ("a_scalar_list", QLineEdit),
        ("an_optional_float", ValueSpinBox),
        ("with_items", ValueComboBox),
    ],
)
def test_editable_field_types_keep_their_editors(
    qapp, form_cls, field_name, expected_widget
):
    """Every type the form genuinely supports is untouched by the read-only fallback."""
    form = _build(form_cls, SampleTaskConfig())

    control = _control(form, field_name)

    assert isinstance(control.widget, expected_widget)
    assert control.editable


@pytest.mark.parametrize("form_cls", FORMS, ids=FORM_IDS)
@pytest.mark.parametrize("field_name", ["nested", "nested_list"])
def test_a_field_the_form_cannot_edit_is_shown_read_only(qapp, form_cls, field_name):
    """A dataclass field (or a list of them) is displayed, but not as an editor.

    Displayed rather than dropped: these are real settings an operator wants to
    read off the form, even when they have to be changed elsewhere.
    """
    form = _build(form_cls, SampleTaskConfig())

    param_widget = _control(form, field_name)
    widget = param_widget.widget

    assert not param_widget.editable
    assert not param_widget.editable
    assert isinstance(widget, QLineEdit)
    assert widget.isReadOnly() and not widget.isEnabled()
    assert widget.toolTip()  # says where the field can be edited instead


@pytest.mark.parametrize("form_cls", FORMS, ids=FORM_IDS)
def test_tabbing_past_an_uneditable_field_leaves_the_value_alone(
    qapp, form_cls, monkeypatch
):
    """The original bug: focus in, focus out, and the dataclass became a string.

    No typing and no editing -- QLineEdit emits editingFinished on focus loss, the
    form read the box's text back and assigned it to the field. The task then
    failed at run time and the experiment could not be saved.
    """
    config = SampleTaskConfig()
    form = _build(form_cls, config)

    # A host with a second focusable widget, so focus can actually leave the field.
    host = QWidget()
    layout = QVBoxLayout(host)
    layout.addWidget(form)
    elsewhere = QLineEdit()
    layout.addWidget(elsewhere)
    host.show()

    nested_box = _control(form, "nested").widget
    nested_box.setFocus(Qt.MouseFocusReason)
    qapp.processEvents()
    elsewhere.setFocus(Qt.MouseFocusReason)
    qapp.processEvents()

    assert isinstance(config.nested, NestedSettings)
    assert config.nested.depth == 1.5

    host.close()


@pytest.mark.parametrize("form_cls", FORMS, ids=FORM_IDS)
def test_the_write_back_refuses_an_uneditable_field(qapp, form_cls):
    """Even called directly, the change handler will not write a read-only field.

    The disabled box cannot take focus, so this path is unreachable through the
    UI today -- it is guarded anyway because the failure it prevents is silent
    data loss, and a later change to how the form is built should not be able to
    reintroduce it.

    Asserting the *signal* rather than only the value: ReadOnlyParameterWidget
    hands back the value it was given, so an unguarded handler would assign the
    same object and leave the config looking untouched. What gives it away is the
    change being announced -- which marks the protocol dirty and saves it -- for
    an edit that never happened.
    """
    config = SampleTaskConfig()
    form = _build(form_cls, config)

    announced = []
    for signal_name in ("parameter_changed", "config_changed"):
        signal = getattr(form, signal_name, None)
        if signal is not None:
            signal.connect(lambda *args, name=signal_name: announced.append(name))

    form._on_parameter_changed("nested")
    form._on_parameter_changed("nested_list")

    assert announced == []
    assert isinstance(config.nested, NestedSettings)
    assert isinstance(config.nested_list, list)
    assert all(isinstance(item, NestedSettings) for item in config.nested_list)


@pytest.mark.parametrize("form_cls", FORMS, ids=FORM_IDS)
def test_an_editable_field_still_writes_through(qapp, form_cls):
    """The guard blocks only the fields it should -- editing is otherwise unchanged."""
    config = SampleTaskConfig()
    form = _build(form_cls, config)

    _control(form, "a_float").widget.setValue(9.0)
    form._on_parameter_changed("a_float")

    assert config.a_float == 9.0


@pytest.mark.parametrize("form_cls", FORMS, ids=FORM_IDS)
def test_a_literal_field_offers_its_allowed_values(qapp, form_cls):
    """Literal[...] is a fixed set of values written in the type itself.

    It used to fall through to the unknown-type fallback and render as free text,
    so an operator could type a value the field does not accept and nothing
    objected. It is the same thing the `items` metadata key expresses, so it gets
    the same combo box.
    """
    config = SampleTaskConfig()
    form = _build(form_cls, config)

    param_widget = _control(form, "a_literal")
    combo = param_widget.widget

    assert isinstance(combo, ValueComboBox)
    assert [combo.itemData(i) for i in range(combo.count())] == ["cells", "hpf", "other"]
    assert combo.currentData() == "hpf"

    combo.setCurrentIndex(0)
    form._on_parameter_changed("a_literal")
    assert config.a_literal == "cells"


@pytest.mark.parametrize("form_cls", FORMS, ids=FORM_IDS)
def test_no_signal_is_connected_to_an_uneditable_field(qapp, form_cls):
    """The guard is applied at connection time too, not only in the handler."""
    config = SampleTaskConfig()
    form = _build(form_cls, config)

    nested_box = _control(form, "nested").widget
    editable_box = _control(form, "a_string").widget

    assert nested_box.receivers(nested_box.editingFinished) == 0
    assert editable_box.receivers(editable_box.editingFinished) == 1


@pytest.mark.parametrize("form_cls", FORMS, ids=FORM_IDS)
def test_every_widget_type_the_form_builds_is_connectable(qapp, form_cls):
    """Guards the list in _connect_widget_signals against a new editor type.

    An editable widget whose Qt class is not one of the four the connector knows
    about is connected to nothing: it looks editable, accepts input and never
    reaches the config. Cheaper to assert here than to notice in the app.
    """
    form = _build(form_cls, SampleTaskConfig())
    known = (QCheckBox, QSpinBox, QDoubleSpinBox, QLineEdit, QComboBox)

    unconnectable = [
        name
        for name, param_widget in ((r.field, r.control) for r in form._rows)
        if param_widget.editable and not isinstance(param_widget.widget, known)
    ]

    assert unconnectable == []


# --- the vocabulary the task form could not read before (FIB-526) -------------


@dataclass
class OtherTaskConfig(AutoLamellaTaskConfig):
    """A second task to switch to, so a rebuild is exercised."""

    task_type: ClassVar[str] = "OTHER_TEST"
    display_name: ClassVar[str] = "Other Test"

    something_else: float = 2.0


@dataclass
class VocabularyTaskConfig(AutoLamellaTaskConfig):
    """A task config declaring the keys the form used to ignore.

    Every one of these was already valid metadata that a pattern or strategy
    honoured. The task form read four keys and silently dropped the rest, so
    declaring them here did nothing at all.
    """

    task_type: ClassVar[str] = "VOCABULARY_TEST"
    display_name: ClassVar[str] = "Vocabulary Test"

    bounded: float = field(
        default=5.0,
        metadata={"minimum": -20.0, "maximum": 20.0, "step": 0.5, "decimals": 4},
    )
    bounded_int: int = field(default=3, metadata={"minimum": 1, "maximum": 9})
    tucked_away: bool = field(default=True, metadata={"advanced": True})
    not_shown: bool = field(default=True, metadata={"hidden": True})
    labelled: float = field(default=1.0, metadata={"label": "Custom Label"})
    maybe_text: Optional[str] = None


@pytest.mark.parametrize("form_cls", FORMS, ids=FORM_IDS)
def test_declared_bounds_reach_the_spinbox(qapp, form_cls):
    """The headline: an int field used to get the full 32-bit range regardless.

    Nothing could tell the form otherwise, so a field with a real range had no
    way to say so and an operator could type any integer at all.
    """
    form = _build(form_cls, VocabularyTaskConfig())

    spin = _control(form, "bounded").widget
    assert (spin.minimum(), spin.maximum()) == (-20.0, 20.0)
    assert spin.singleStep() == 0.5
    assert spin.decimals() == 4

    int_spin = _control(form, "bounded_int").widget
    assert (int_spin.minimum(), int_spin.maximum()) == (1, 9)


@pytest.mark.parametrize("form_cls", FORMS, ids=FORM_IDS)
def test_an_undeclared_field_keeps_the_form_defaults(qapp, form_cls):
    """Moving onto the shared builder must not silently retune existing forms.

    `ValueSpinBox` would otherwise supply (0.0, 1e6) -- a floor of zero on every
    float field that never declared a minimum, which clamps without erroring.
    """
    form = _build(form_cls, VocabularyTaskConfig())

    spin = _control(form, "labelled").widget
    assert (spin.minimum(), spin.maximum()) == TASK_FORM_DEFAULTS.float_range
    assert spin.decimals() == TASK_FORM_DEFAULTS.decimals


@pytest.mark.parametrize("form_cls", FORMS, ids=FORM_IDS)
def test_a_hidden_field_gets_no_row(qapp, form_cls):
    form = _build(form_cls, VocabularyTaskConfig())

    assert [row.field for row in form._rows].count("not_shown") == 0


@pytest.mark.parametrize("form_cls", FORMS, ids=FORM_IDS)
def test_an_advanced_field_is_built_but_hidden_until_asked_for(qapp, form_cls):
    """Advanced is a disclosure level, not a deletion -- the row still exists."""
    form = _build(form_cls, VocabularyTaskConfig())

    row = next(r for r in form._rows if r.field == "tucked_away")
    assert row.advanced
    assert not row.control.widget.isVisibleTo(form)

    form.set_advanced_visible(True)
    assert row.control.widget.isVisibleTo(form)


@pytest.mark.parametrize("form_cls", FORMS, ids=FORM_IDS)
def test_a_declared_label_is_used(qapp, form_cls):
    """One of the two containers already did this and the other did not."""
    form = _build(form_cls, VocabularyTaskConfig())

    row = next(r for r in form._rows if r.field == "labelled")
    assert row.label.text() == "Custom Label"


@pytest.mark.parametrize("form_cls", FORMS, ids=FORM_IDS)
def test_an_unset_optional_string_reads_back_as_none(qapp, form_cls):
    """It used to render the literal text "None" and read that straight back.

    AcquireReferenceImage.filename is Optional[str] and defaults to None,
    meaning "generate one". The box showed "None", and a focus-out was enough to
    write that four-character string onto the config as the filename.
    """
    form = _build(form_cls, VocabularyTaskConfig())

    control = _control(form, "maybe_text")
    assert control.widget.text() == ""
    assert control.read() is None


# --- the write-back path, driven through the signals a user actually triggers --


@pytest.mark.parametrize("form_cls", FORMS, ids=FORM_IDS)
@pytest.mark.parametrize(
    "field_name,drive,expected",
    [
        ("a_float", lambda w: w.setValue(9.0), 9.0),
        ("an_int", lambda w: w.setValue(7), 7),
        ("a_bool", lambda w: w.setChecked(False), False),
        ("with_items", lambda w: w.setCurrentIndex(1), "b"),
        ("a_literal", lambda w: w.setCurrentIndex(0), "cells"),
    ],
)
def test_editing_a_control_reaches_the_config(qapp, form_cls, field_name, drive, expected):
    """Drive the widget's own signal, not the handler.

    The tests around this one called `form._on_parameter_changed(name)` directly,
    which exercises the write but not the wiring -- and the wiring is where this
    broke. The slot was `lambda name=row.field: ...`, which accepts one
    positional argument, so PyQt handed it the signal's payload: `valueChanged`'s
    float, `toggled`'s bool, `currentIndexChanged`'s int. `name` became the value,
    the row lookup found nothing, and the edit was dropped in silence.

    Every control whose signal carries a payload was affected; only
    `editingFinished` (the text fields) has none, which is why the form looked
    like it half worked.
    """
    config = SampleTaskConfig()
    form = _build(form_cls, config)

    drive(_control(form, field_name).widget)

    assert getattr(config, field_name) == expected


@pytest.mark.parametrize("form_cls", FORMS, ids=FORM_IDS)
def test_the_change_signal_carries_the_field_name_and_value(qapp, form_cls):
    """The protocol editor writes the value into the protocol from these two.

    A payload-shadowed slot never got as far as emitting, so nothing downstream
    saw the edit and nothing was saved.
    """
    form = _build(form_cls, SampleTaskConfig())
    seen = []
    signal = getattr(form, "parameter_changed", None)
    if signal is None:
        pytest.skip("this container reports whole-config changes instead")
    signal.connect(lambda name, value: seen.append((name, value)))

    _control(form, "a_float").widget.setValue(3.5)

    assert seen == [("a_float", 3.5)]


@pytest.mark.parametrize("form_cls", FORMS, ids=FORM_IDS)
def test_an_edit_survives_switching_task_and_coming_back(qapp, form_cls):
    """The reported symptom, end to end.

    Switching tasks rebuilds the form from the config, so an edit that never
    reached the config reappears as the old value -- which is what made this look
    like a rebuild bug rather than a write-back one.
    """
    config = SampleTaskConfig()
    other = OtherTaskConfig()
    form = _build(form_cls, config)

    _control(form, "a_float").widget.setValue(9.0)

    form.set_task_config(other)
    form.set_task_config(config)

    assert config.a_float == 9.0
    assert _control(form, "a_float").widget.value() == 9.0
