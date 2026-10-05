"""Stage and view geometry, as pure functions: no microscope, no I/O.

Where an image displacement lands on the stage, and where a stage position shows up in
a view. Everything here takes the instrument geometry and the stage pose as arguments,
so the same answer serves a live move and a saved image.

``movement`` turns image displacements into stage movements. The projection it builds on
is still in ``fibsem.transformations``, and the views in ``fibsem.projection``; both are
to move here, with their old modules kept as re-exports.
"""
