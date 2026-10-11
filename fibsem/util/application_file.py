"""Matching a requested patterning application file against the available ones."""

from __future__ import annotations

from difflib import get_close_matches
from typing import List


def match_application_file(
    application_file: str, application_files: List[str], strict: bool = True
) -> str:
    """The application file to use for `application_file`, from those available.

    Application files are a ThermoFisher patterning setting, which the Demo simulates
    too. With `strict`, a name that is not available raises; otherwise the closest
    available name is used.

    Raises:
        ValueError: If no available application file matches.
    """
    if application_file not in application_files:
        if strict:
            raise ValueError(
                f"Application file {application_file} not available. Available files: {application_files}"
            )
        closest_match = get_close_matches(application_file, application_files, n=1)
        if not closest_match:
            raise ValueError(
                f"Application file {application_file} not available. Available files: {application_files}"
            )
        application_file = str(closest_match[0])

    return application_file
