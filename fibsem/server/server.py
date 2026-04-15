"""
FibsemServer: wraps any FibsemMicroscope instance and exposes it over HTTP via FastAPI.

Usage:
    from fibsem.server.server import FibsemServer
    server = FibsemServer.from_session(manufacturer="Demo", ip_address="localhost", port=8001)
    server.run()

Or as a script:
    python -m fibsem.server.server --manufacturer Demo --host 0.0.0.0 --port 8001
"""

import io

import tifffile as tff
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response

from fibsem import utils
from fibsem.microscope import FibsemMicroscope
from fibsem.server.models import (
    AcquireImageRequest,
    AvailableValuesRequest,
    BeamSettingsRequest,
    BeamSystemSettingsRequest,
    BeamTypeRequest,
    DetectorSettingsRequest,
    DrawPatternsRequest,
    FinishMillingRequest,
    FlatToBeamRequest,
    FloatBeamRequest,
    ImageSettingsRequest,
    IsCloseToMillingAngleRequest,
    MillingAngleFromPositionRequest,
    MillingAngleRequest,
    MicroscopeStateRequest,
    MillingSettingsRequest,
    MoveToMillingAngleRequest,
    PointBeamRequest,
    ProjectStableMoveRequest,
    ResolutionBeamRequest,
    RunMillingRequest,
    StableMoveRequest,
    StagePositionRequest,
    StagePositionResponse,
    StringBeamRequest,
    VerticalMoveRequest,
)
from fibsem.structures import (
    BeamSettings,
    BeamSystemSettings,
    BeamType,
    FibsemBitmapSettings,
    FibsemCircleSettings,
    FibsemDetectorSettings,
    FibsemLineSettings,
    FibsemMillingSettings,
    FibsemPatternSettings,
    FibsemPolygonSettings,
    FibsemRectangleSettings,
    FibsemStagePosition,
    ImageSettings,
    MicroscopeState,
    Point,
)

_PATTERN_CLASSES = {
    "Rectangle": FibsemRectangleSettings,
    "Line": FibsemLineSettings,
    "Circle": FibsemCircleSettings,
    "Bitmap": FibsemBitmapSettings,
    "Polygon": FibsemPolygonSettings,
}


def _pattern_from_dict(d: dict) -> FibsemPatternSettings:
    type_name = d.get("type")
    if type_name not in _PATTERN_CLASSES:
        raise ValueError(f"Unknown pattern type: {type_name!r}. Available: {list(_PATTERN_CLASSES)}")
    return _PATTERN_CLASSES[type_name].from_dict(d)


def _image_response(image) -> Response:
    buf = io.BytesIO()
    metadata = image.metadata.to_dict() if image.metadata is not None else None
    tff.imwrite(buf, image.data, metadata=metadata)
    return Response(content=buf.getvalue(), media_type="image/tiff")


def _beam_type(value: str) -> BeamType:
    try:
        return BeamType[value.upper()]
    except KeyError:
        raise HTTPException(status_code=422, detail=f"Unknown beam_type: {value!r}. Use 'ELECTRON' or 'ION'.")


class FibsemServer:
    def __init__(self, microscope: FibsemMicroscope, host: str = "0.0.0.0", port: int = 8001):
        self.microscope = microscope
        self.host = host
        self.port = port
        self.app = self._build_app()

    def _build_app(self) -> FastAPI:
        app = FastAPI(title="FibsemMicroscope Server")
        microscope = self.microscope

        # --- Health / System ---

        @app.get("/health")
        def health():
            return {"status": "ok", "manufacturer": type(microscope).__name__}

        @app.get("/system")
        def get_system():
            return {
                "system": microscope.system.to_dict(),
                "stage_is_compustage": microscope.stage_is_compustage,
            }

        # --- Image acquisition ---

        @app.post("/acquire_image")
        def acquire_image(body: AcquireImageRequest) -> Response:
            bt = _beam_type(body.beam_type)
            image_settings = ImageSettings.from_dict(body.image_settings) if body.image_settings else None
            return _image_response(microscope.acquire_image(image_settings=image_settings, beam_type=bt))

        @app.post("/last_image")
        def last_image(body: BeamTypeRequest) -> Response:
            return _image_response(microscope.last_image(beam_type=_beam_type(body.beam_type)))

        @app.post("/acquire_chamber_image")
        def acquire_chamber_image() -> Response:
            return _image_response(microscope.acquire_chamber_image())

        @app.post("/autocontrast")
        def autocontrast(body: BeamTypeRequest):
            microscope.autocontrast(beam_type=_beam_type(body.beam_type))
            return {"status": "ok"}

        @app.post("/auto_focus")
        def auto_focus(body: BeamTypeRequest):
            microscope.auto_focus(beam_type=_beam_type(body.beam_type))
            return {"status": "ok"}

        # --- Stage movement ---

        @app.get("/stage_position")
        def get_stage_position():
            return {"position": microscope.get_stage_position().to_dict()}

        @app.get("/stage_orientation")
        def get_stage_orientation():
            return {"orientation": microscope.get_stage_orientation()}

        @app.post("/move_stage_absolute", response_model=StagePositionResponse)
        def move_stage_absolute(body: StagePositionRequest):
            result = microscope.move_stage_absolute(FibsemStagePosition.from_dict(body.position))
            return StagePositionResponse(position=result.to_dict())

        @app.post("/move_stage_relative", response_model=StagePositionResponse)
        def move_stage_relative(body: StagePositionRequest):
            result = microscope.move_stage_relative(FibsemStagePosition.from_dict(body.position))
            return StagePositionResponse(position=result.to_dict())

        @app.post("/stable_move", response_model=StagePositionResponse)
        def stable_move(body: StableMoveRequest):
            result = microscope.stable_move(dx=body.dx, dy=body.dy, beam_type=_beam_type(body.beam_type))
            return StagePositionResponse(position=result.to_dict())

        @app.post("/project_stable_move", response_model=StagePositionResponse)
        def project_stable_move(body: ProjectStableMoveRequest):
            base_position = FibsemStagePosition.from_dict(body.base_position)
            result = microscope.project_stable_move(
                dx=body.dx, dy=body.dy,
                beam_type=_beam_type(body.beam_type),
                base_position=base_position,
            )
            return StagePositionResponse(position=result.to_dict())

        @app.post("/vertical_move", response_model=StagePositionResponse)
        def vertical_move(body: VerticalMoveRequest):
            result = microscope.vertical_move(dy=body.dy, dx=body.dx, static_wd=body.static_wd)
            return StagePositionResponse(position=result.to_dict())

        @app.post("/safe_absolute_stage_movement")
        def safe_absolute_stage_movement(body: StagePositionRequest):
            microscope.safe_absolute_stage_movement(FibsemStagePosition.from_dict(body.position))
            return {"status": "ok"}

        @app.post("/move_flat_to_beam")
        def move_flat_to_beam(body: FlatToBeamRequest):
            microscope.move_flat_to_beam(beam_type=_beam_type(body.beam_type))
            return {"status": "ok"}

        # --- Microscope state ---

        @app.get("/microscope_state")
        def get_microscope_state():
            return {"microscope_state": microscope.get_microscope_state().to_dict()}

        @app.post("/microscope_state")
        def set_microscope_state(body: MicroscopeStateRequest):
            microscope.set_microscope_state(MicroscopeState.from_dict(body.microscope_state))
            return {"status": "ok"}

        # --- Imaging settings ---

        @app.post("/imaging_settings/get")
        def get_imaging_settings(body: BeamTypeRequest):
            return {"image_settings": microscope.get_imaging_settings(_beam_type(body.beam_type)).to_dict()}

        @app.post("/imaging_settings/set")
        def set_imaging_settings(body: ImageSettingsRequest):
            microscope.set_imaging_settings(ImageSettings.from_dict(body.image_settings))
            return {"status": "ok"}

        # --- Beam settings ---

        @app.post("/beam_settings/get")
        def get_beam_settings(body: BeamTypeRequest):
            return {"beam_settings": microscope.get_beam_settings(_beam_type(body.beam_type)).to_dict()}

        @app.post("/beam_settings/set")
        def set_beam_settings(body: BeamSettingsRequest):
            microscope.set_beam_settings(BeamSettings.from_dict(body.beam_settings))
            return {"status": "ok"}

        @app.post("/beam_system_settings/get")
        def get_beam_system_settings(body: BeamTypeRequest):
            return {"beam_system_settings": microscope.get_beam_system_settings(_beam_type(body.beam_type)).to_dict()}

        @app.post("/beam_system_settings/set")
        def set_beam_system_settings(body: BeamSystemSettingsRequest):
            microscope.set_beam_system_settings(BeamSystemSettings.from_dict(body.beam_system_settings))
            return {"status": "ok"}

        # --- Detector settings ---

        @app.post("/detector_settings/get")
        def get_detector_settings(body: BeamTypeRequest):
            return {"detector_settings": microscope.get_detector_settings(_beam_type(body.beam_type)).to_dict()}

        @app.post("/detector_settings/set")
        def set_detector_settings(body: DetectorSettingsRequest):
            microscope.set_detector_settings(
                FibsemDetectorSettings.from_dict(body.detector_settings),
                beam_type=_beam_type(body.beam_type),
            )
            return {"status": "ok"}

        # --- Individual beam getters / setters ---

        @app.post("/beam_current/get")
        def get_beam_current(body: BeamTypeRequest):
            return {"value": microscope.get_beam_current(_beam_type(body.beam_type))}

        @app.post("/beam_current/set")
        def set_beam_current(body: FloatBeamRequest):
            return {"value": microscope.set_beam_current(body.value, _beam_type(body.beam_type))}

        @app.post("/beam_voltage/get")
        def get_beam_voltage(body: BeamTypeRequest):
            return {"value": microscope.get_beam_voltage(_beam_type(body.beam_type))}

        @app.post("/beam_voltage/set")
        def set_beam_voltage(body: FloatBeamRequest):
            return {"value": microscope.set_beam_voltage(body.value, _beam_type(body.beam_type))}

        @app.post("/field_of_view/get")
        def get_field_of_view(body: BeamTypeRequest):
            return {"value": microscope.get_field_of_view(_beam_type(body.beam_type))}

        @app.post("/field_of_view/set")
        def set_field_of_view(body: FloatBeamRequest):
            return {"value": microscope.set_field_of_view(body.value, _beam_type(body.beam_type))}

        @app.post("/working_distance/get")
        def get_working_distance(body: BeamTypeRequest):
            return {"value": microscope.get_working_distance(_beam_type(body.beam_type))}

        @app.post("/working_distance/set")
        def set_working_distance(body: FloatBeamRequest):
            return {"value": microscope.set_working_distance(body.value, _beam_type(body.beam_type))}

        @app.post("/dwell_time/get")
        def get_dwell_time(body: BeamTypeRequest):
            return {"value": microscope.get_dwell_time(_beam_type(body.beam_type))}

        @app.post("/dwell_time/set")
        def set_dwell_time(body: FloatBeamRequest):
            return {"value": microscope.set_dwell_time(body.value, _beam_type(body.beam_type))}

        @app.post("/resolution/get")
        def get_resolution(body: BeamTypeRequest):
            return {"value": list(microscope.get_resolution(_beam_type(body.beam_type)))}

        @app.post("/resolution/set")
        def set_resolution(body: ResolutionBeamRequest):
            return {"value": list(microscope.set_resolution(body.value, _beam_type(body.beam_type)))}

        @app.post("/scan_rotation/get")
        def get_scan_rotation(body: BeamTypeRequest):
            return {"value": microscope.get_scan_rotation(_beam_type(body.beam_type))}

        @app.post("/scan_rotation/set")
        def set_scan_rotation(body: FloatBeamRequest):
            return {"value": microscope.set_scan_rotation(body.value, _beam_type(body.beam_type))}

        @app.post("/stigmation/get")
        def get_stigmation(body: BeamTypeRequest):
            return {"value": microscope.get_stigmation(_beam_type(body.beam_type)).to_dict()}

        @app.post("/stigmation/set")
        def set_stigmation(body: PointBeamRequest):
            result = microscope.set_stigmation(Point.from_dict(body.value), _beam_type(body.beam_type))
            return {"value": result.to_dict()}

        @app.post("/beam_shift/get")
        def get_beam_shift(body: BeamTypeRequest):
            return {"value": microscope.get_beam_shift(_beam_type(body.beam_type)).to_dict()}

        @app.post("/beam_shift/set")
        def set_beam_shift(body: PointBeamRequest):
            result = microscope.set_beam_shift(Point.from_dict(body.value), _beam_type(body.beam_type))
            return {"value": result.to_dict()}

        # --- Detector individual getters / setters ---

        @app.post("/detector_type/get")
        def get_detector_type(body: BeamTypeRequest):
            return {"value": microscope.get_detector_type(_beam_type(body.beam_type))}

        @app.post("/detector_type/set")
        def set_detector_type(body: StringBeamRequest):
            return {"value": microscope.set_detector_type(body.value, _beam_type(body.beam_type))}

        @app.post("/detector_mode/get")
        def get_detector_mode(body: BeamTypeRequest):
            return {"value": microscope.get_detector_mode(_beam_type(body.beam_type))}

        @app.post("/detector_mode/set")
        def set_detector_mode(body: StringBeamRequest):
            return {"value": microscope.set_detector_mode(body.value, _beam_type(body.beam_type))}

        @app.post("/detector_contrast/get")
        def get_detector_contrast(body: BeamTypeRequest):
            return {"value": microscope.get_detector_contrast(_beam_type(body.beam_type))}

        @app.post("/detector_contrast/set")
        def set_detector_contrast(body: FloatBeamRequest):
            return {"value": microscope.set_detector_contrast(body.value, _beam_type(body.beam_type))}

        @app.post("/detector_brightness/get")
        def get_detector_brightness(body: BeamTypeRequest):
            return {"value": microscope.get_detector_brightness(_beam_type(body.beam_type))}

        @app.post("/detector_brightness/set")
        def set_detector_brightness(body: FloatBeamRequest):
            return {"value": microscope.set_detector_brightness(body.value, _beam_type(body.beam_type))}

        # --- Available values ---

        @app.post("/available_values")
        def get_available_values(body: AvailableValuesRequest):
            beam_type = _beam_type(body.beam_type) if body.beam_type else None
            return {"values": microscope.get_available_values(body.key, beam_type=beam_type)}

        # --- Milling angle ---

        @app.get("/milling_angle")
        def get_milling_angle():
            return {"milling_angle": microscope.get_current_milling_angle()}

        @app.post("/milling_angle/from_position")
        def get_milling_angle_from_position(body: MillingAngleFromPositionRequest):
            position = FibsemStagePosition.from_dict(body.stage_position) if body.stage_position else None
            return {"milling_angle": microscope.get_current_milling_angle(stage_position=position)}

        @app.post("/milling_angle/set")
        def set_milling_angle(body: MillingAngleRequest):
            microscope.set_milling_angle(body.milling_angle)
            return {"status": "ok"}

        @app.post("/milling_angle/move")
        def move_to_milling_angle(body: MoveToMillingAngleRequest):
            success = microscope.move_to_milling_angle(body.milling_angle, rotation=body.rotation)
            return {"success": success, "milling_angle": microscope.get_current_milling_angle()}

        @app.post("/milling_angle/is_close")
        def is_close_to_milling_angle(body: IsCloseToMillingAngleRequest):
            return {"is_close": microscope.is_close_to_milling_angle(body.milling_angle, atol=body.atol)}

        # --- Milling ---

        @app.post("/setup_milling")
        def setup_milling(body: MillingSettingsRequest):
            microscope.setup_milling(mill_settings=FibsemMillingSettings.from_dict(body.mill_settings))
            return {"status": "ok"}

        @app.post("/draw_patterns")
        def draw_patterns(body: DrawPatternsRequest):
            try:
                patterns = [_pattern_from_dict(p) for p in body.patterns]
            except (KeyError, ValueError) as e:
                raise HTTPException(status_code=422, detail=str(e))
            microscope.draw_patterns(patterns)
            return {"status": "ok"}

        @app.post("/run_milling")
        def run_milling(body: RunMillingRequest):
            microscope.run_milling(
                milling_current=body.milling_current,
                milling_voltage=body.milling_voltage,
                asynch=body.asynch,
            )
            return {"status": "ok"}

        @app.post("/start_milling")
        def start_milling():
            microscope.start_milling()
            return {"status": "ok"}

        @app.post("/stop_milling")
        def stop_milling():
            microscope.stop_milling()
            return {"status": "ok"}

        @app.post("/pause_milling")
        def pause_milling():
            microscope.pause_milling()
            return {"status": "ok"}

        @app.post("/resume_milling")
        def resume_milling():
            microscope.resume_milling()
            return {"status": "ok"}

        @app.post("/finish_milling")
        def finish_milling(body: FinishMillingRequest):
            microscope.finish_milling(
                imaging_current=body.imaging_current,
                imaging_voltage=body.imaging_voltage,
            )
            return {"status": "ok"}

        @app.post("/clear_patterns")
        def clear_patterns():
            microscope.clear_patterns()
            return {"status": "ok"}

        @app.get("/milling_state")
        def get_milling_state():
            return {"state": microscope.get_milling_state().name}

        @app.get("/estimate_milling_time")
        def estimate_milling_time():
            return {"seconds": microscope.estimate_milling_time()}

        return app

    def run(self):
        uvicorn.run(self.app, host=self.host, port=self.port)

    @classmethod
    def from_session(
        cls,
        manufacturer: str,
        ip_address: str,
        host: str = "0.0.0.0",
        port: int = 8001,
    ) -> "FibsemServer":
        microscope, _ = utils.setup_session(manufacturer=manufacturer, ip_address=ip_address)
        return cls(microscope, host=host, port=port)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Start a FibsemMicroscope HTTP server")
    parser.add_argument("--manufacturer", default="Demo", help="Microscope manufacturer (default: Demo)")
    parser.add_argument("--ip-address", default="localhost", help="Microscope IP address")
    parser.add_argument("--host", default="0.0.0.0", help="Server host (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8001, help="Server port (default: 8001)")
    args = parser.parse_args()

    server = FibsemServer.from_session(
        manufacturer=args.manufacturer,
        ip_address=args.ip_address,
        host=args.host,
        port=args.port,
    )
    server.run()
