import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import tifffile as tff
from ome_types import from_xml, to_xml
from ome_types.model import (
    OME as OMEMetadata,
    Binning,
    Channel,
    Channel_ContrastMethod,
    Channel_IlluminationType,
    DetectorSettings,
    FilterSet,
    FilterSetRef,
    Image,
    Instrument,
    LightEmittingDiode,
    LightSourceSettings,
    Objective,
    ObjectiveSettings,
    Pixels,
    Pixels_DimensionOrder,
    PixelType,
    Plane,
    TiffData,
    UnitsLength,
)
from ome_types.model import Detector as OME_Detector

BINNING_MAP = {
    1: Binning.ONEBYONE,
    2: Binning.TWOBYTWO,
    4: Binning.FOURBYFOUR,
    8: Binning.EIGHTBYEIGHT
}

@dataclass
class ChannelSettings:
    name: str
    excitation_wavelength: float
    emission_wavelength: Optional[float]
    power: float
    exposure_time: float

    def to_dict(self):
        return {
            "name": self.name,
            "excitation_wavelength": self.excitation_wavelength,
            "emission_wavelength": self.emission_wavelength,
            "power": self.power,
            "exposure_time": self.exposure_time,
        }

    @classmethod
    def from_dict(cls, ddict: dict) -> "ChannelSettings":
        return cls(
            name=ddict["name"],
            excitation_wavelength=ddict["excitation_wavelength"],
            emission_wavelength=ddict.get("emission_wavelength"),
            power=ddict["power"],
            exposure_time=ddict["exposure_time"],
        )


@dataclass
class ZParameters:
    zmin: float = -10e-6
    zmax: float = 10e-6
    zstep: float = 1e-6

    def to_dict(self) -> dict:
        return {"zmin": self.zmin, "zmax": self.zmax, "zstep": self.zstep}

    @classmethod
    def from_dict(cls, ddict: dict) -> "ZParameters":
        return cls(zmin=ddict["zmin"], zmax=ddict["zmax"], zstep=ddict["zstep"])

    def generate_positions(self, z_init: float) -> List[float]:
        """Generate a list of z positions based on the current z init and relative z parameters.
        Args:
            z_init (float): The initial z position from which to calculate the relative positions.
        Returns:
            List[float]: A list of z positions starting from z_init + zmin to z_init + zmax, with steps of zstep."""

        if self.zstep <= 0:
            raise ValueError("zstep must be a positive value.")

        # Generate z positions
        z_positions = np.arange(
            start=z_init + self.zmin, stop=z_init + self.zmax, step=self.zstep
        )

        logging.debug(f"Generated z positions: {z_positions}")
        return z_positions.tolist()


# QUERY: add AcquisitionSettings class to handle acquisition settings?

@dataclass
class FluorescenceImage:
    data: np.ndarray # TCZYX format (Time, Channels, Z, Y, X)
    metadata: List[Dict[str, any]] = field(default_factory=list)

    def save(self, filename: str):
        """
        Save a FMImage to a TIFF file with OME metadata.
        
        Args:
            filename (str): The filename to save the image to.
        """
        ome_md = self.get_ome_metadata()
        ome_xml = ome_md.to_xml()
        
        # Validate OME XML
        assert tff.OmeXml.validate(ome_xml), "OME XML is not valid"
        
        # Reshape image to 5D for tifffile (CZYX -> TCZYX)
        if self.data.ndim != 5:
            nc, nz, ny, nx = self.data.shape  # C, Z, Y, X
            tifffile_image = self.data.reshape(1, nc, nz, ny, nx)
        else:
            tifffile_image = self.data

        # TODO: add overwrite protection to prevent overwriting existing files
        with tff.TiffWriter(filename) as tif:
            tif.write(data=tifffile_image, contiguous=True)
            tif.overwrite_description(ome_xml)

        return filename

    def get_ome_metadata(self) -> OMEMetadata:
        """Generate OME metadata for the FluorescenceImage."""

        ifd = 0
        planes: List[Plane] = []
        tiff_data_blocks: List[TiffData] = []

        channels_md: List[Channel] = []
        detectors: List[OME_Detector] = []
        light_sources: List[LightEmittingDiode] = []

        nc, nz, ny, nx = self.data.shape  # C, Z, Y, X
        dtype = self.data.dtype.name

        for ch_idx in range(nc):
            print(f"Image {ch_idx} Metadata:")
            md = self.metadata[ch_idx]
            id_str = f"{ch_idx+1:02d}"


            # TODO: stage-position

            light_source_md = md["light_source"]
            filter_md = md["filter_set"]
            camera_md = md["camera"]

            light_source_power = light_source_md["power"]

            camera_binning = BINNING_MAP.get(camera_md["binning"], None)
            camera_gain = camera_md["gain"]
            camera_offset = camera_md["offset"]

            # Convert to nm
            excitation_wavelength = int(filter_md["excitation_wavelength"] * 1e9)
            emission_wavelength = filter_md["emission_wavelength"]
            # convert to nm if not None
            if emission_wavelength is not None:
                emission_wavelength = int(emission_wavelength * 1e9)

            light_source = LightEmittingDiode(
                id=f"LightSource:{id_str}", power=light_source_power
            )

            detector = OME_Detector(
                id=f"Detector:{id_str}",
                gain=camera_gain,
                offset=camera_offset,
            )
            detector_settings = DetectorSettings(
                id=f"Detector:{id_str}",
                gain=camera_gain,
                offset=camera_offset,
                binning=camera_binning,
            )

            detectors.append(detector)
            light_sources.append(light_source)

            ch_md = Channel(
                name=f"Channel:{id_str}",
                id=f"Channel:{id_str}",
                excitation_wavelength=excitation_wavelength,
                excitation_wavelength_unit=UnitsLength.NANOMETER,
                emission_wavelength=emission_wavelength,
                emission_wavelength_unit=UnitsLength.NANOMETER,
                detector_settings=detector_settings,
                illumination_type=Channel_IlluminationType.EPIFLUORESCENCE,
                contrast_method=Channel_ContrastMethod.FLUORESCENCE,
            )
            channels_md.append(ch_md)

            pixel_size_x = camera_md["pixel_size"][0]
            pixel_size_y = camera_md["pixel_size"][1]
            if nz > 1:
                z_positions = md["objective-positions"]
                pixel_size_z = z_positions[1] - z_positions[0]
            else:
                z_positions = [md["objective"]["position"]]
                pixel_size_z = None  # No z dimension, set to 0
            exposure_time = camera_md["exposure_time"]

            for z_idx in range(nz):
                plane = Plane(
                    the_c=ch_idx,
                    the_t=0,
                    the_z=z_idx,
                    exposure_time=exposure_time,
                    position_x=None,  # TODO: implement stage position
                    position_y=None,  # NOTE: implement stage position
                    position_z=z_positions[z_idx],
                    position_x_unit=UnitsLength.METER,
                    position_y_unit=UnitsLength.METER,
                    position_z_unit=UnitsLength.METER,
                )
                tiff_data = TiffData(
                    ifd=ifd, first_c=ch_idx, first_z=z_idx, plane_count=1
                )

                planes.append(plane)
                tiff_data_blocks.append(tiff_data)

                ifd += 1

        # TODO: manufacturer, model, serial number, etc.
        inst = Instrument(
            id="Instrument:01", detectors=detectors, light_emitting_diodes=light_sources
        )
        image_md = Image(
            id="Image:01",
            name="Image",
            description="Image Description",
            acquisition_date=datetime.now(),
            pixels=Pixels(
                id="Pixels:01",
                channels=channels_md,
                dimension_order=Pixels_DimensionOrder.XYCZT,
                size_x=nx,
                size_y=ny,
                size_z=nz,
                size_c=nc,
                size_t=1,
                physical_size_x=pixel_size_x,
                physical_size_y=pixel_size_y,
                physical_size_z=pixel_size_z,
                physical_size_x_unit=UnitsLength.METER,
                physical_size_y_unit=UnitsLength.METER,
                physical_size_z_unit=UnitsLength.METER,
                type=dtype,
                planes=planes,
                tiff_data_blocks=tiff_data_blocks,
            ),
        )

        ome_md = OMEMetadata(images=[image_md], instruments=[inst])

        return ome_md


    @classmethod
    def load(cls, filename: str) -> "FluorescenceImage":
        """Load an image from a file."""
        from tifffile import imread

        data = imread(filename)
        metadata = {}
        return cls(data=data, metadata=metadata) # TODO: load metadata from OME XML

    @staticmethod
    def create_z_stack(ch_images: List['FluorescenceImage']) -> 'FluorescenceImage':
        """Create a z-stack from a list of FluorescenceImage objects."""

        # Ensure all images have the same shape
        shapes = [img.data.shape for img in ch_images]
        if not all(shape == shapes[0] for shape in shapes):
            raise ValueError("All images must have the same shape for z-stacking.")
        
        arrs = np.stack([img.data for img in ch_images], axis=0)
        md = ch_images[0].metadata.copy()
        md['objective-positions'] = [img.metadata['objective']['position'] for img in ch_images]

        # this should be all the same metadata, except for z position

        return FluorescenceImage(data=arrs, metadata=md)

    @staticmethod
    def create_multi_channel_image(
        images: List['FluorescenceImage'],
    ) -> 'FluorescenceImage':
        """Create a multi-channel image from a list of FluorescenceImage objects."""
        
        # Ensure all images have the same shape
        shapes = [img.data.shape for img in images]
        if not all(shape == shapes[0] for shape in shapes):
            raise ValueError("All images must have the same shape for multi-channel stacking.")


        if len(images) == 0:
            raise ValueError("No images provided for multi-channel stacking.")

        img_data= []
        for img in images:
            # If the image is 2D, add a singleton z dimension
            if img.data.ndim == 2:
                img.data = img.data[np.newaxis, :, :]
            img_data.append(img.data)

        # Stack images along the channel dimension (CZYX)
        stacked_data = np.stack(img_data, axis=0)  # Shape: (C, Z, Y, X)

        # Combine metadata from all images #TODO: handle metadata more clearly
        mds = [img.metadata for img in images]

        return FluorescenceImage(data=stacked_data, metadata=mds)


# TODO: create FMImageMetadata class to handle metadata more clearly rather than using dicts
# split into core microscope metadata, and channel metadata
# e.g. microscope-state?