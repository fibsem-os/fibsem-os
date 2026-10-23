"""Stage and view geometry, as pure functions: no microscope, no I/O.

Where an image displacement lands on the stage, and where a stage position shows up in
a view. Everything here takes the instrument geometry and the stage pose as arguments,
so the same answer serves a live move and a saved image.

``frames`` is the stage as rotations in the chamber, the model the readers are moving
onto. ``movement`` holds the view projection (an image displacement to a stage
movement and back) and the stage movements built on it, and ``projection`` the views a
canvas draws stage positions in (``StageProjection``). ``orientations`` names the pose a
position is in, the milling angle a tilt gives, and carries a position from one pose to
another. ``similarity`` fits the 2D turn, scale and offset between point pairs, behind
three-point alignment.
"""
