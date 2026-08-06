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

from fibsem.applications.autolamella.structures import AutoLamellaTaskConfig  # noqa: E402
from fibsem.ui.widgets.autolamella_task_config_widget import (  # noqa: E402
    AutoLamellaTaskConfigWidget,
    AutoLamellaTaskParametersConfigWidget,
    BoolParameterWidget,
    ComboParameterWidget,
    EnumParameterWidget,
    FloatParameterWidget,
    IntParameterWidget,
    ListParameterWidget,
    ReadOnlyParameterWidget,
    StringParameterWidget,
)


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
        ("a_bool", BoolParameterWidget),
        ("an_int", IntParameterWidget),
        ("a_float", FloatParameterWidget),
        ("a_string", StringParameterWidget),
        ("an_enum", EnumParameterWidget),
        ("a_scalar_list", ListParameterWidget),
        ("an_optional_float", FloatParameterWidget),
        ("with_items", ComboParameterWidget),
    ],
)
def test_editable_field_types_keep_their_editors(
    qapp, form_cls, field_name, expected_widget
):
    """Every type the form genuinely supports is untouched by the read-only fallback."""
    form = _build(form_cls, SampleTaskConfig())

    param_widget = form.parameter_widgets[field_name]

    assert isinstance(param_widget, expected_widget)
    assert param_widget.editable


@pytest.mark.parametrize("form_cls", FORMS, ids=FORM_IDS)
@pytest.mark.parametrize("field_name", ["nested", "nested_list"])
def test_a_field_the_form_cannot_edit_is_shown_read_only(qapp, form_cls, field_name):
    """A dataclass field (or a list of them) is displayed, but not as an editor.

    Displayed rather than dropped: these are real settings an operator wants to
    read off the form, even when they have to be changed elsewhere.
    """
    form = _build(form_cls, SampleTaskConfig())

    param_widget = form.parameter_widgets[field_name]
    widget = param_widget.widget

    assert isinstance(param_widget, ReadOnlyParameterWidget)
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

    nested_box = form.parameter_widgets["nested"].widget
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

    form.parameter_widgets["a_float"].widget.setValue(9.0)
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

    param_widget = form.parameter_widgets["a_literal"]
    combo = param_widget.widget

    assert isinstance(param_widget, ComboParameterWidget)
    assert isinstance(combo, QComboBox)
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

    nested_box = form.parameter_widgets["nested"].widget
    editable_box = form.parameter_widgets["a_string"].widget

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
        for name, param_widget in form.parameter_widgets.items()
        if param_widget.editable and not isinstance(param_widget.widget, known)
    ]

    assert unconnectable == []
