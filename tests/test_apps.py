"""The app registry: built-in apps, and apps other packages register."""

import importlib.metadata

import pytest

from fibsem.tools import apps as registry
from fibsem.tools.apps import AUTOLAMELLA, BUILTIN_APPS, GATEWAY, App


@pytest.mark.parametrize("app", BUILTIN_APPS.values(), ids=BUILTIN_APPS.keys())
def test_builtin_apps_ship_their_drawings(app):
    assert app.icon.is_file()


def test_the_gateway_shortcut_has_fibsemos_windows_icon():
    assert GATEWAY.command == "fibsem-gateway-open"
    assert GATEWAY.windows_icon is not None
    assert GATEWAY.windows_icon.stem == "fibsemos"
    assert "gateway" not in registry.apps()  # a shortcut, not a launcher app


def _registered(monkeypatch, *entry_points):
    # Real EntryPoint objects; only the environment's list of them is stood in for.
    monkeypatch.setattr(
        registry.importlib.metadata,
        "entry_points",
        lambda group: [ep for ep in entry_points if ep.group == group],
    )


def _entry_point(name, value):
    return importlib.metadata.EntryPoint(name, value, registry.ENTRY_POINT_GROUP)


def test_a_registered_app_is_added_after_the_builtins(monkeypatch):
    _registered(monkeypatch, _entry_point("extra", "fibsem.tools.apps:AUTOLAMELLA"))

    found = registry.apps()

    assert list(found) == [*BUILTIN_APPS, "extra"]
    assert found["extra"] is AUTOLAMELLA


def test_broken_or_wrong_entry_points_are_skipped(monkeypatch):
    _registered(
        monkeypatch,
        _entry_point("missing", "no_such_module_anywhere:APP"),
        _entry_point("not-an-app", "fibsem.tools.apps:ENTRY_POINT_GROUP"),
        _entry_point("autolamella", "fibsem.tools.apps:FIBSEM_UI"),
    )

    found = registry.apps()

    assert found == BUILTIN_APPS  # nothing added, and no built-in replaced
    assert all(isinstance(app, App) for app in found.values())
