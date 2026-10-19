"""Readers for older file shapes: one module per kind of file.

Each takes the data as an older version wrote it and returns it in the current
shape, for `fibsem.codec.decode` to read. A file is saved back in the current shape,
so an old shape is read here once, the next time its file is saved.

Imported where the file is read, not by `fibsem.codec`: these know the structures,
and the structures import the codec.
"""
