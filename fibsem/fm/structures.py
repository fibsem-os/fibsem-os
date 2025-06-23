import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import List, Optional, Tuple

import numpy as np
import tifffile as tff
from ome_types.model import (
    OME as OMEMetadata,
    Binning,
    Channel,
    Channel_ContrastMethod,
    Channel_IlluminationType,
    Detector as OME_Detector,
    DetectorSettings,
    Image,
    Instrument,
    LightEmittingDiode,
    LightSourceSettings,
    MapAnnotation,
    Objective,
    Pixels,
    Pixels_DimensionOrder,
    Plane,
    StructuredAnnotations,
    TiffData,
    UnitsLength,
)

from fibsem.structures import FibsemStagePosition

BINNING_MAP = {
    1: Binning.ONEBYONE,
    2: Binning.TWOBYTWO,
    4: Binning.FOURBYFOUR,
    8: Binning.EIGHTBYEIGHT
}

@dataclass
class ChannelSettings:
    name: str
    excitation_wavelength: float  # nm
    emission_wavelength: Optional[float]  # nm
    power: float
    exposure_time: float

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, ddict: dict) -> "ChannelSettings":
        return cls(**ddict)


@dataclass
class ZParameters:
    zmin: float = -10e-6
    zmax: float = 10e-6
    zstep: float = 1e-6

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, ddict: dict) -> "ZParameters":
        return cls(**ddict)

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
            start=z_init + self.zmin,
            stop=z_init + self.zmax,
            step=self.zstep
        )

        logging.debug(f"Generated z positions: {z_positions}")
        return z_positions.tolist()


# QUERY: add AcquisitionSettings class to handle acquisition settings?

@dataclass
class FluorescenceImage:
    data: np.ndarray # TCZYX format (Time, Channels, Z, Y, X)
    metadata: 'FluorescenceImageMetadata' = None

    def save(self, filename: str) -> str:
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
        objectives: List[Objective] = []

        nc, nz, ny, nx = self.data.shape  # C, Z, Y, X
        dtype = self.data.dtype.name

        # Use structured metadata directly
        for ch_idx, channel in enumerate(self.metadata.channels):
            id_str = f"{ch_idx+1:02d}"

            # Extract channel metadata
            light_source_power = channel.power
            camera_binning = BINNING_MAP.get(channel.binning, None)
            camera_gain = channel.gain
            camera_offset = channel.offset

            # wavelengths are in nm
            excitation_wavelength = int(channel.excitation_wavelength)
            emission_wavelength = channel.emission_wavelength
            if emission_wavelength is not None:
                emission_wavelength = int(emission_wavelength)

            # Create OME components
            light_source = LightEmittingDiode(
                id=f"LightSource:{id_str}", power=light_source_power
            )

            light_source_settings = LightSourceSettings(
                id=f"LightSource:{id_str}",
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

            # Create objective if magnification or NA are available
            objective_id = f"Objective:{id_str}"
            if channel.objective_magnification is not None or channel.objective_numerical_aperture is not None:
                objective = Objective(
                    id=objective_id,
                    nominal_magnification=channel.objective_magnification,
                    lens_na=channel.objective_numerical_aperture,
                )
                objectives.append(objective)

            ch_md = Channel(
                name=channel.name,
                id=f"Channel:{id_str}",
                excitation_wavelength=excitation_wavelength,
                excitation_wavelength_unit=UnitsLength.NANOMETER,
                emission_wavelength=emission_wavelength,
                emission_wavelength_unit=UnitsLength.NANOMETER,
                detector_settings=detector_settings,
                illumination_type=Channel_IlluminationType.EPIFLUORESCENCE,
                contrast_method=Channel_ContrastMethod.FLUORESCENCE,
                light_source_settings=light_source_settings,
            )
            channels_md.append(ch_md)

            # Get pixel sizes and z-positions from metadata
            pixel_size_x = self.metadata.pixel_size_x
            pixel_size_y = self.metadata.pixel_size_y
            
            if nz > 1 and self.metadata.z_positions:
                z_positions = self.metadata.z_positions
                pixel_size_z = self.metadata.pixel_size_z
            else:
                z_positions = [channel.objective_position]
                pixel_size_z = None

            exposure_time = channel.exposure_time

            # Create planes for this channel
            for z_idx in range(nz):
                # Get stage position if available
                stage_x = self.metadata.stage_position.x if self.metadata.stage_position else None
                stage_y = self.metadata.stage_position.y if self.metadata.stage_position else None
                
                plane = Plane(
                    the_c=ch_idx,
                    the_t=0,
                    the_z=z_idx,
                    exposure_time=exposure_time,
                    position_x=stage_x,
                    position_y=stage_y,
                    position_z=z_positions[z_idx] if z_idx < len(z_positions) else z_positions[0],
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

        # TODO: use metadata.system_info for manufacturer, model, serial number, etc.
        inst = Instrument(
            id="Instrument:01", 
            detectors=detectors, 
            light_emitting_diodes=light_sources,
            objectives=objectives if objectives else None
        )
        
        # Parse acquisition date
        try:
            acquisition_date = datetime.fromisoformat(self.metadata.acquisition_date)
        except (ValueError, AttributeError):
            acquisition_date = datetime.now()

        image_md = Image(
            id="Image:01",
            name="Fluorescence Image",
            description="Fluorescence microscopy image",
            acquisition_date=acquisition_date,
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

        # Create structured annotations with full metadata
        metadata_annotation = MapAnnotation(
            id="Annotation:01",
            value={"FluorescenceImageMetadata": json.dumps(self.metadata.to_dict())}
        )
        
        structured_annotations = StructuredAnnotations(
            map_annotations=[metadata_annotation]
        )

        ome_md = OMEMetadata(
            images=[image_md], 
            instruments=[inst],
            structured_annotations=structured_annotations
        )

        return ome_md

    # NOTES: objective, detector should be common across all channels... not per channel

    @classmethod
    def load(cls, filename: str) -> "FluorescenceImage":
        """Load an image from a file with metadata recovery from structured annotations."""
        from tifffile import imread
        
        # Load image data
        data = imread(filename)
        
        # Handle fallback reshaping for non-OME files
        if data.ndim == 2:
            # Simple 2D image -> CZYX (single channel, single Z)
            data = data[np.newaxis, np.newaxis, :, :]
        elif data.ndim == 3:
            # 3D image -> CZYX (single channel, multi-Z)
            data = data[np.newaxis, :, :, :]
        
        # Try to load metadata from structured annotations
        try:
            import ome_types
            ome = ome_types.from_tiff(filename)
            
            # Look for FluorescenceImageMetadata in structured annotations
            if (ome.structured_annotations and 
                ome.structured_annotations.map_annotations):
                for annotation in ome.structured_annotations.map_annotations:
                    if (annotation.value and 
                        "FluorescenceImageMetadata" in annotation.value):
                        # Found our custom metadata
                        metadata_json = annotation.value["FluorescenceImageMetadata"]
                        metadata_dict = json.loads(metadata_json)
                        metadata = FluorescenceImageMetadata.from_dict(metadata_dict)
                        
                        # Reshape data to CZYX based on metadata
                        nc = len(metadata.channels)
                        nz = len(metadata.z_positions) if metadata.z_positions else 1
                        
                        if data.ndim == 4:  # Reshape from loaded format to CZYX
                            if data.shape[0] == nc and data.shape[1] == nz:
                                # Data is already CZYX
                                pass
                            elif data.shape[0] == nz and data.shape[1] == nc:
                                # Data is ZCYX, transpose to CZYX
                                data = data.transpose(1, 0, 2, 3)
                        elif data.ndim == 3:
                            if nc > 1 and nz == 1:
                                # Multi-channel, single Z: CYX -> CZYX
                                data = data[:, np.newaxis, :, :]
                            else:
                                # Single channel, multi-Z: ZYX -> CZYX
                                data = data[np.newaxis, :, :, :]
                        elif data.ndim == 2:
                            # Single channel, single Z: YX -> CZYX
                            data = data[np.newaxis, np.newaxis, :, :]
                        
                        return cls(data=data, metadata=metadata)
                        
        except Exception as e:
            logging.warning(f"Failed to load structured annotations: {e}")
        
        # Fallback to basic metadata
        metadata = cls._create_basic_metadata(data.shape)
        return cls(data=data, metadata=metadata)
    
    
    @classmethod
    def _create_basic_metadata(cls, data_shape: tuple) -> "FluorescenceImageMetadata":
        """Create basic metadata when no structured annotations are available."""
        # Handle different data shapes: (Y, X), (Z, Y, X), (C, Z, Y, X)
        if len(data_shape) == 2:
            ny, nx = data_shape
            nc = 1
        elif len(data_shape) == 3:
            _, ny, nx = data_shape
            nc = 1
        elif len(data_shape) == 4:
            nc, _, ny, nx = data_shape
        else:
            raise ValueError(f"Unsupported data shape: {data_shape}")
        
        channels = []
        for i in range(nc):
            channel = FluorescenceChannelMetadata(
                name=f"Channel_{i+1:02d}",
                excitation_wavelength=488.0,
                power=1.0,
                exposure_time=0.1,
                gain=1.0,
                offset=0.0
            )
            channels.append(channel)
        
        return FluorescenceImageMetadata(
            acquisition_date=datetime.now().isoformat(),
            pixel_size_x=1e-6,  # default 1 micron
            pixel_size_y=1e-6,  # default 1 micron
            resolution=(nx, ny),
            channels=channels
        )

    @staticmethod
    def create_z_stack(ch_images: List['FluorescenceImage']) -> 'FluorescenceImage':
        """Create a z-stack from a list of FluorescenceImage objects."""

        # Ensure all images have the same shape
        shapes = [img.data.shape for img in ch_images]
        if not all(shape == shapes[0] for shape in shapes):
            raise ValueError("All images must have the same shape for z-stacking.")
        
        # Stack images along Z axis, squeezing singleton dimensions if needed
        data_list = []
        for img in ch_images:
            # If data is (1, Y, X), squeeze to (Y, X) for stacking
            if img.data.ndim == 3 and img.data.shape[0] == 1:
                data_list.append(img.data.squeeze(0))
            else:
                data_list.append(img.data)
        arrs = np.stack(data_list, axis=0)
        
        # Create z-stack metadata from first image, adding z-positions
        first_metadata = ch_images[0].metadata
        z_positions = [img.metadata.channels[0].objective_position for img in ch_images]
        
        # Create new metadata with z-positions
        md = FluorescenceImageMetadata(
            acquisition_date=first_metadata.acquisition_date,
            pixel_size_x=first_metadata.pixel_size_x,
            pixel_size_y=first_metadata.pixel_size_y,
            pixel_size_z=first_metadata.pixel_size_z,
            resolution=first_metadata.resolution,
            channels=first_metadata.channels,
            z_positions=z_positions,
            stage_position=first_metadata.stage_position,
            system_info=first_metadata.system_info
        )

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

        # Combine metadata from all images into FluorescenceImageMetadata
        # Extract common metadata from first image
        first_img = images[0]
        
        # Combine channels from all images
        channels = []
        for img in images:
            channels.extend(img.metadata.channels)
        
        mds = FluorescenceImageMetadata(
            acquisition_date=first_img.metadata.acquisition_date,
            pixel_size_x=first_img.metadata.pixel_size_x,
            pixel_size_y=first_img.metadata.pixel_size_y,
            pixel_size_z=first_img.metadata.pixel_size_z,
            resolution=first_img.metadata.resolution,
            channels=channels,
            z_positions=first_img.metadata.z_positions,
            stage_position=first_img.metadata.stage_position,
            system_info=first_img.metadata.system_info
        )

        return FluorescenceImage(data=stacked_data, metadata=mds)


# TODO: create FMImageMetadata class to handle metadata more clearly rather than using dicts
# split into core microscope metadata, and channel metadata
# e.g. microscope-state?

@dataclass 
class FluorescenceChannelMetadata:
    """Metadata for a single fluorescence channel."""
    
    # Channel identification
    name: str
    
    # Optical parameters
    excitation_wavelength: float  # nm
    
    # Light source
    power: float  # light source power
    
    # Camera settings
    exposure_time: float  # seconds
    gain: float  # camera gain
    offset: float  # camera offset
    
    # Fields with defaults must come after required fields
    emission_wavelength: Optional[float] = None  # nm, None for reflection
    binning: int = 1  # binning factor (1, 2, 4, 8)
    objective_position: float = 0.0  # z-axis position in meters
    objective_magnification: Optional[float] = None
    objective_numerical_aperture: Optional[float] = None
    
    def to_dict(self) -> dict:
        """Convert to dictionary format expected by get_ome_metadata."""
        return {
            "channel": {
                "name": self.name
            },
            "filter_set": {
                "excitation_wavelength": self.excitation_wavelength,
                "emission_wavelength": self.emission_wavelength
            },
            "light_source": {
                "power": self.power
            },
            "camera": {
                "exposure_time": self.exposure_time,
                "gain": self.gain,
                "offset": self.offset,
                "binning": self.binning
            },
            "objective": {
                "position": self.objective_position,
                "magnification": self.objective_magnification,
                "numerical_aperture": self.objective_numerical_aperture
            }
        }
    
    @staticmethod
    def from_dict(data: dict) -> "FluorescenceChannelMetadata":
        """Create from dictionary format used by get_ome_metadata."""
        return FluorescenceChannelMetadata(
            name=data["channel"]["name"],
            excitation_wavelength=data["filter_set"]["excitation_wavelength"],
            emission_wavelength=data["filter_set"].get("emission_wavelength"),
            power=data["light_source"]["power"],
            exposure_time=data["camera"]["exposure_time"],
            gain=data["camera"]["gain"],
            offset=data["camera"]["offset"],
            binning=data["camera"].get("binning", 1),
            objective_position=data["objective"]["position"],
            objective_magnification=data["objective"].get("magnification"),
            objective_numerical_aperture=data["objective"].get("numerical_aperture")
        )


@dataclass
class FluorescenceImageMetadata:
    """Complete metadata for fluorescence microscopy images."""
    
    # Image acquisition settings
    acquisition_date: str  # ISO timestamp string
    pixel_size_x: float  # meters, after binning
    pixel_size_y: float  # meters, after binning
    pixel_size_z: Optional[float] = None  # meters, for z-stacks
    resolution: Tuple[int, int] = (1024, 1024)  # (width, height) after binning
    
    # Channel metadata - list of individual channel settings
    channels: List[FluorescenceChannelMetadata] = field(default_factory=list)
    
    # Z-stack parameters (for multi-plane acquisitions)
    z_positions: Optional[List[float]] = None  # meters, objective positions
    
    # Stage position (optional, for correlative imaging)
    stage_position: Optional[FibsemStagePosition] = None
    
    # System information (optional)
    system_info: Optional[dict] = None
    
    def __post_init__(self):
        """Validate metadata consistency."""
        if not self.channels:
            raise ValueError("At least one channel must be specified")
            
        # Validate binning factors
        valid_binning = {1, 2, 4, 8}
        for channel in self.channels:
            if channel.binning not in valid_binning:
                raise ValueError(f"Invalid binning factor {channel.binning}, must be one of {valid_binning}")
        
        # Validate z-stack consistency
        if self.z_positions is not None:
            if len(self.z_positions) < 2:
                raise ValueError("Z-stack must have at least 2 positions")
            if self.pixel_size_z is None:
                # Auto-calculate z pixel size from positions
                self.pixel_size_z = abs(self.z_positions[1] - self.z_positions[0])   
    
    def add_channel(self, channel: FluorescenceChannelMetadata) -> None:
        """Add a new channel to the metadata."""
        self.channels.append(channel)
    
    def get_channel_count(self) -> int:
        """Get the number of channels."""
        return len(self.channels)
    
    def get_z_count(self) -> int:
        """Get the number of z-planes."""
        return len(self.z_positions) if self.z_positions else 1
    
    def to_dict(self) -> dict:
        """Convert to dictionary format for UI/viewer compatibility."""
        return {
            "acquisition_date": self.acquisition_date,
            "pixel_size_x": self.pixel_size_x,
            "pixel_size_y": self.pixel_size_y,
            "pixel_size_z": self.pixel_size_z,
            "resolution": self.resolution,
            "channel_count": self.get_channel_count(),
            "z_count": self.get_z_count(),
            "z_positions": self.z_positions,
            "stage_position": self.stage_position.to_dict() if self.stage_position else None,
            "system_info": self.system_info,
            "channels": [
                {
                    "name": ch.name,
                    "excitation_wavelength": ch.excitation_wavelength,
                    "emission_wavelength": ch.emission_wavelength,
                    "power": ch.power,
                    "exposure_time": ch.exposure_time,
                    "gain": ch.gain,
                    "offset": ch.offset,
                    "binning": ch.binning,
                    "objective_position": ch.objective_position,
                    "objective_magnification": ch.objective_magnification,
                    "objective_numerical_aperture": ch.objective_numerical_aperture
                } for ch in self.channels
            ]
        }
    
    @classmethod
    def from_dict(cls, metadata_dict: dict) -> "FluorescenceImageMetadata":
        """Create FluorescenceImageMetadata from dictionary."""
        # Reconstruct channels
        channels = []
        for ch_dict in metadata_dict.get("channels", []):
            channel = FluorescenceChannelMetadata(
                name=ch_dict["name"],
                excitation_wavelength=ch_dict["excitation_wavelength"],
                emission_wavelength=ch_dict.get("emission_wavelength"),
                power=ch_dict["power"],
                exposure_time=ch_dict["exposure_time"],
                gain=ch_dict["gain"],
                offset=ch_dict["offset"],
                binning=ch_dict.get("binning", 1),
                objective_position=ch_dict.get("objective_position", 0.0),
                objective_magnification=ch_dict.get("objective_magnification"),
                objective_numerical_aperture=ch_dict.get("objective_numerical_aperture")
            )
            channels.append(channel)
        
        # Reconstruct stage position if present
        stage_position = None
        if metadata_dict.get("stage_position"):
            stage_dict = metadata_dict["stage_position"]
            stage_position = FibsemStagePosition(
                x=stage_dict.get("x", 0),
                y=stage_dict.get("y", 0),
                z=stage_dict.get("z", 0),
                r=stage_dict.get("r", 0),
                t=stage_dict.get("t", 0),
                coordinate_system=stage_dict.get("coordinate_system", "RAW")
            )
        
        return cls(
            acquisition_date=metadata_dict["acquisition_date"],
            pixel_size_x=metadata_dict["pixel_size_x"],
            pixel_size_y=metadata_dict["pixel_size_y"],
            pixel_size_z=metadata_dict.get("pixel_size_z"),
            resolution=tuple(metadata_dict["resolution"]),
            channels=channels,
            z_positions=metadata_dict.get("z_positions"),
            stage_position=stage_position,
            system_info=metadata_dict.get("system_info")
        )
