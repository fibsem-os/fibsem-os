import os


def remove_suffix(text: str, suffix: str) -> str:
    """Return ``text`` with a trailing ``suffix`` removed, if it is there.

    ``str.removesuffix`` does exactly this, but it landed in Python 3.9 and this
    package still supports 3.8 (``requires-python = ">=3.8"``, and the CI matrix
    runs it). The failure is a runtime ``AttributeError`` rather than an import
    error, so nothing catches it until a 3.8 user hits the line -- see the
    fluorescence save path in ``fibsem.fm.acquisition.acquire_image``, where the
    exception was swallowed by the surrounding ``except`` and lost the image.
    """
    if suffix and text.endswith(suffix):
        return text[: -len(suffix)]
    return text


def _get_extension(filename: str) -> str:
    if filename.endswith(
        ".ome.tiff"
    ):  # special case for OME-TIFF files (double extension)
        return ".ome.tiff"
    else:
        return os.path.splitext(filename)[1]


def _get_basename(filename: str) -> str:
    return remove_suffix(filename, _get_extension(filename))


def _get_basename_and_extension(filename: str) -> tuple:
    return _get_basename(filename), _get_extension(filename)


def get_unique_filename(filename):
    if not os.path.exists(filename):
        return filename

    basename, ext = _get_basename_and_extension(filename)
    idx = 1
    while os.path.exists(filename):
        filename = f"{basename}-{idx}{ext}"
        idx += 1

    return filename
