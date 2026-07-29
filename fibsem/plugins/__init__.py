"""Entry point plugin loading and introspection.

Deliberately empty.

``loader`` is imported by ``fibsem.milling.patterning`` while
``fibsem.milling.base`` is only partially initialised (see the note in
``loader.py``), so anything this package's ``__init__`` imported would be
dragged into that window. ``report`` in particular imports all three
registries and would deadlock the import graph outright.

Import the submodules directly::

    from fibsem.plugins.loader import load_entry_point_group   # early, safe
    from fibsem.plugins.report import collect_extensions       # late, needs the registries
"""
