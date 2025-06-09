from typing import Optional
from abc import ABC, abstractmethod
import numpy as np
from typing import List, Tuple, Dict
import matplotlib.pyplot as plt
from pprint import pprint

from fibsem import utils
from dataclasses import dataclass, field

@dataclass
class ChannelSettings:
    name: str
    excitation_wavelength: float
    emission_wavelength: Optional[float]
    power: float
    exposure_time: float
    binning: int = 1

    def to_dict(self):
        return {
            "name": self.name,
            "excitation_wavelength": self.excitation_wavelength,
            "emission_wavelength": self.emission_wavelength,
            "power": self.power,
            "exposure_time": self.exposure_time,
            "binning": self.binning
        }

    @classmethod
    def from_dict(cls, ddict: dict) -> "ChannelSettings":
        return cls(
            name=ddict["name"],
            excitation_wavelength=ddict["excitation_wavelength"],
            emission_wavelength=ddict.get("emission_wavelength"),
            power=ddict["power"],
            exposure_time=ddict["exposure_time"],
            binning=ddict.get("binning", 1)
        )


@dataclass
class ZParameters:
    zmin: float = -10e-6
    zmax: float = 10e-6
    zstep: float = 1e-6

    def to_dict(self) -> dict:
        return {
            "zmin": self.zmin,
            "zmax": self.zmax,
            "zstep": self.zstep
        }

    @classmethod
    def from_dict(cls, ddict: dict) -> "ZParameters":
        return cls(
            zmin=ddict["zmin"],
            zmax=ddict["zmax"],
            zstep=ddict["zstep"]
        )
    
    def generate_positions(self, z_init: float) -> List[float]:
        """Generate a list of z positions based on the current z init and relative z parameters.
        Args:
            z_init (float): The initial z position from which to calculate the relative positions.
        Returns:
            List[float]: A list of z positions starting from z_init + zmin to z_init + zmax, with steps of zstep."""

        zmin = self.zmin
        zmax = self.zmax
        zstep = self.zstep

        if zstep <= 0:
            raise ValueError("zstep must be a positive value.")

        # Generate z positions
        z_positions = []
        current_z = z_init + zmin
        while current_z <= z_init + zmax:
            z_positions.append(current_z)
            current_z += zstep

        return z_positions

@dataclass
class FluorescenceImage:
    data: np.ndarray
    metadata: Dict[str, any] = field(default_factory=dict)

    def save(self, filename: str):
        """Save the image data and metadata to a file."""
        pass
        # utils.save_image(self.data, filename, metadata=self.metadata)

        # TODO: convert metadata to valid OME format

    @classmethod
    def load(cls, filename: str) -> "FluorescenceImage":
        """Load an image from a file."""
        from tifffile import imread
        data = imread(filename)
        metadata = {}
        return cls(
            data=data,
            metadata=metadata
        )