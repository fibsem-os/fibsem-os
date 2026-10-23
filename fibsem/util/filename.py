import os


def _get_extension(filename: str) -> str:
    if filename.endswith(
        ".ome.tiff"
    ):  # special case for OME-TIFF files (double extension)
        return ".ome.tiff"
    else:
        return os.path.splitext(filename)[1]


def _get_basename(filename: str) -> str:
    return filename.removesuffix(_get_extension(filename))


def _get_basename_and_extension(filename: str) -> tuple:
    return _get_basename(filename), _get_extension(filename)
