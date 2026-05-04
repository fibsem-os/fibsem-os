"""
FibsemServer: wraps any FibsemMicroscope instance and exposes it over HTTP via FastAPI.

Usage:
    from fibsem.server.server import FibsemServer
    server = FibsemServer.from_session(manufacturer="Demo", ip_address="localhost", port=8001)
    server.run()

Or as a script:
    python -m fibsem.server.server --manufacturer Demo --host 0.0.0.0 --port 8001
"""

import asyncio
import base64
import io
import math
from datetime import datetime
from typing import Optional, Set

import tifffile as tff
import uvicorn
from fastapi import FastAPI, HTTPException, WebSocket
from fastapi.responses import HTMLResponse, Response
from fastapi.websockets import WebSocketDisconnect
from pydantic import BaseModel

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


def _image_to_b64_jpeg(image, max_width: int = 1536) -> Optional[str]:
    try:
        import numpy as np
        from PIL import Image as PILImage
        data = image.data.copy()
        if data.dtype != np.uint8:
            d_min, d_max = float(data.min()), float(data.max())
            if d_max > d_min:
                data = ((data.astype(np.float32) - d_min) / (d_max - d_min) * 255).astype(np.uint8)
            else:
                data = np.zeros_like(data, dtype=np.uint8)
        pil = PILImage.fromarray(data, mode="L" if data.ndim == 2 else "RGB")
        if pil.width > max_width:
            pil = pil.resize((max_width, int(pil.height * max_width / pil.width)), PILImage.Resampling.LANCZOS)
        buf = io.BytesIO()
        pil.save(buf, format="JPEG", quality=80)
        return base64.b64encode(buf.getvalue()).decode()
    except Exception:
        return None


def _stage_to_dash(pos) -> dict:
    return {
        "x_mm": pos.x * 1e3 if pos.x is not None else None,
        "y_mm": pos.y * 1e3 if pos.y is not None else None,
        "z_mm": pos.z * 1e3 if pos.z is not None else None,
        "tilt_deg": math.degrees(pos.t) if pos.t is not None else None,
    }


def _pattern_to_dash(p: dict) -> Optional[dict]:
    t = p.get("type", "")
    if t == "Rectangle":
        return {"type": "rect",
                "width_um": p.get("width", 0) * 1e6, "height_um": p.get("height", 0) * 1e6,
                "centre_x_um": p.get("centre_x", 0) * 1e6, "centre_y_um": p.get("centre_y", 0) * 1e6,
                "rotation_deg": math.degrees(p.get("rotation", 0))}
    if t == "Line":
        return {"type": "line",
                "start_x_um": p.get("start_x", 0) * 1e6, "start_y_um": p.get("start_y", 0) * 1e6,
                "end_x_um": p.get("end_x", 0) * 1e6, "end_y_um": p.get("end_y", 0) * 1e6}
    if t == "Circle":
        return {"type": "circle",
                "centre_x_um": p.get("centre_x", 0) * 1e6, "centre_y_um": p.get("centre_y", 0) * 1e6,
                "radius_um": p.get("radius", 0) * 1e6}
    return None


_DASHBOARD_HTML = """<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>fibsemOS Live View</title>
<style>
*{box-sizing:border-box;margin:0;padding:0}
body{background:#1e1e1e;color:#ccc;font-family:monospace;font-size:12px;display:flex;flex-direction:column;height:100vh;overflow:hidden}
#header{background:#252526;padding:6px 12px;display:flex;justify-content:space-between;align-items:center;border-bottom:1px solid #3c3c3c;flex-shrink:0}
#header h1{font-size:13px;color:#ddd;font-weight:normal}
#badge{padding:2px 8px;border-radius:3px;font-size:11px;background:#333;color:#aaa}
#badge.RUNNING{background:#7a5c00;color:#ffd}
#badge.IDLE{background:#1a3a1a;color:#8c8}
#images{display:grid;grid-template-columns:1fr 1fr;gap:4px;padding:4px;flex:1;min-height:0}
.panel{background:#252526;border:1px solid #3c3c3c;border-radius:3px;display:flex;flex-direction:column;min-height:0}
.panel-label{padding:3px 8px;font-size:10px;color:#666;background:#2a2a2a;border-bottom:1px solid #3c3c3c;flex-shrink:0}
.img-wrap{position:relative;flex:1;display:flex;align-items:center;justify-content:center;overflow:hidden;background:#111}
.img-wrap img{width:100%;height:100%;object-fit:contain;display:block}
.placeholder{color:#444;font-size:11px;position:absolute;pointer-events:none}
.img-svg{position:absolute;top:0;left:0;width:100%;height:100%;pointer-events:none}
#fib-wrap{cursor:default}
#fib-wrap.waiting{cursor:crosshair}
#click-banner{display:none;padding:4px 8px;background:#003a5c;color:#6cc;font-size:11px;border-bottom:1px solid #007acc;flex-shrink:0}
#click-banner.on{display:block}
#approval-banner{display:none;padding:6px 12px;background:#1a2e1a;border-bottom:2px solid #4caf50;flex-shrink:0;align-items:center;gap:12px}
#approval-banner.on{display:flex}
#approval-prompt{flex:1;font-size:11px;color:#8fc}
#approval-params{font-size:10px;color:#666;font-family:monospace;white-space:pre;flex:2}
.btn-approve{background:#2d5a2d;color:#8fc;border:1px solid #4caf50;padding:3px 14px;cursor:pointer;font-size:11px;border-radius:2px}
.btn-approve:hover{background:#3d6a3d}
.btn-cancel{background:#5a2d2d;color:#fc8c8c;border:1px solid #cf4f4f;padding:3px 14px;cursor:pointer;font-size:11px;border-radius:2px;margin-left:6px}
.btn-cancel:hover{background:#6a3d3d}
#footer{background:#252526;padding:5px 12px;border-top:1px solid #3c3c3c;display:flex;justify-content:space-between;font-size:11px;color:#666;flex-shrink:0}
</style>
</head>
<body>
<div id="header">
  <h1>fibsemOS Live View</h1>
  <span id="badge">IDLE</span>
</div>
<div id="approval-banner">
  <span id="approval-prompt">Approve milling?</span>
  <span id="approval-params"></span>
  <button class="btn-approve" onclick="handleApprove()">&#10003; Approve</button>
  <button class="btn-cancel" onclick="handleCancel()">&#10007; Cancel</button>
</div>
<div id="images">
  <div class="panel">
    <div class="panel-label">SEM (electron)</div>
    <div class="img-wrap">
      <span class="placeholder" id="sem-ph">No image acquired</span>
      <img id="sem-img" style="display:none">
      <svg id="sem-svg" class="img-svg" preserveAspectRatio="none"></svg>
    </div>
  </div>
  <div class="panel">
    <div class="panel-label">FIB (ion)</div>
    <div id="click-banner"><span id="click-prompt">Click to select a point</span></div>
    <div class="img-wrap" id="fib-wrap" onclick="handleClick(event)">
      <span class="placeholder" id="fib-ph">No image acquired</span>
      <img id="fib-img" style="display:none">
      <svg id="fib-svg" class="img-svg" preserveAspectRatio="none"></svg>
    </div>
  </div>
</div>
<div id="footer">
  <span id="stage-info">Stage: —</span>
  <span id="ts">Connecting…</span>
</div>
<script>
let st={};
const WS=(location.protocol==='https:'?'wss:':'ws:')+'//'+location.host+'/ws';
let ws;
function connect(){
  ws=new WebSocket(WS);
  ws.onmessage=function(e){try{render(JSON.parse(e.data));}catch(ex){}};
  ws.onclose=function(){document.getElementById('ts').textContent='⚠ Disconnected — reconnecting…';setTimeout(connect,2000);};
  ws.onerror=function(){ws.close();};
}
function svgEl(tag,attrs){const e=document.createElementNS('http://www.w3.org/2000/svg',tag);Object.entries(attrs).forEach(([k,v])=>e.setAttribute(k,v));return e;}
function niceBarUm(hfw){const raw=hfw/5;const s=[1,2,5,10,20,50,100,200,500,1000];return s.reduce((p,c)=>Math.abs(c-raw)<Math.abs(p-raw)?c:p);}
function drawCrosshair(svg,cx,cy){
  const arm=1.5,gap=0.4,lw=0.25,col='#ffe066';
  [[cx-arm,cy,cx-gap,cy],[cx+gap,cy,cx+arm,cy],[cx,cy-arm,cx,cy-gap],[cx,cy+gap,cx,cy+arm]].forEach(
    ([x1,y1,x2,y2])=>svg.appendChild(svgEl('line',{x1,y1,x2,y2,stroke:col,'stroke-width':lw})));
}
function drawScaleBar(svg,hfw,h){
  const sc=100/hfw,barUm=niceBarUm(hfw),bw=barUm*sc,x=3,y=h-2,col='#ffe066';
  svg.appendChild(svgEl('line',{x1:x,y1:y,x2:x+bw,y2:y,stroke:col,'stroke-width':0.4}));
  svg.appendChild(svgEl('line',{x1:x,y1:y-0.5,x2:x,y2:y+0.5,stroke:col,'stroke-width':0.3}));
  svg.appendChild(svgEl('line',{x1:x+bw,y1:y-0.5,x2:x+bw,y2:y+0.5,stroke:col,'stroke-width':0.3}));
  const t=svgEl('text',{x:x+bw/2,y:y-1,'text-anchor':'middle','font-size':1.8,fill:col});
  t.textContent=barUm>=1000?(barUm/1000)+' mm':barUm+' µm';svg.appendChild(t);
}
function render(s){
  st=s;
  const si=document.getElementById('sem-img');
  if(s.sem_jpeg){si.src='data:image/jpeg;base64,'+s.sem_jpeg;si.style.display='';document.getElementById('sem-ph').style.display='none';}
  const fi=document.getElementById('fib-img');
  if(s.fib_jpeg){fi.src='data:image/jpeg;base64,'+s.fib_jpeg;fi.style.display='';document.getElementById('fib-ph').style.display='none';}
  // SEM overlay: crosshair + scale bar
  const semSvg=document.getElementById('sem-svg');
  const shfw=s.sem_hfw_um||150,sasp=s.sem_aspect||0.75,sh=100*sasp;
  semSvg.setAttribute('viewBox','0 0 100 '+sh.toFixed(2));
  semSvg.innerHTML='';
  if(s.sem_jpeg){drawCrosshair(semSvg,50,sh/2);drawScaleBar(semSvg,shfw,sh);}
  // FIB overlay: crosshair + scale bar + patterns
  const svg=document.getElementById('fib-svg');
  const hfw=s.hfw_um||150,asp=s.fib_aspect||0.667,h=100*asp;
  svg.setAttribute('viewBox','0 0 100 '+h.toFixed(2));
  svg.innerHTML='';
  const cx=50,cy=h/2,sc=100/hfw;
  if(s.fib_jpeg){drawCrosshair(svg,cx,cy);drawScaleBar(svg,hfw,h);}
  (s.patterns||[]).forEach(p=>{
    const col='#ffe066',fill='rgba(255,220,0,0.15)';
    if(p.type==='rect'){
      const w=p.width_um*sc,rh=p.height_um*sc,px=cx+p.centre_x_um*sc,py=cy+p.centre_y_um*sc;
      const r=svgEl('rect',{x:px-w/2,y:py-rh/2,width:w,height:rh,fill,stroke:col,'stroke-width':0.4});
      if(p.rotation_deg)r.setAttribute('transform','rotate('+p.rotation_deg+','+px+','+py+')');
      svg.appendChild(r);
      const lbl=svgEl('text',{x:px,y:py-rh/2-0.8,'text-anchor':'middle','font-size':1.8,fill:col});
      lbl.textContent=p.width_um.toFixed(1)+'×'+p.height_um.toFixed(1)+' µm';svg.appendChild(lbl);
    }else if(p.type==='line'){
      const x1=cx+p.start_x_um*sc,y1=cy+p.start_y_um*sc,x2=cx+p.end_x_um*sc,y2=cy+p.end_y_um*sc;
      svg.appendChild(svgEl('line',{x1,y1,x2,y2,stroke:col,'stroke-width':0.4}));
      const len=Math.hypot(p.end_x_um-p.start_x_um,p.end_y_um-p.start_y_um);
      const lbl=svgEl('text',{x:(x1+x2)/2,y:(y1+y2)/2-0.8,'text-anchor':'middle','font-size':1.8,fill:col});
      lbl.textContent=len.toFixed(1)+' µm';svg.appendChild(lbl);
    }else if(p.type==='circle'){
      const pcx=cx+p.centre_x_um*sc,pcy=cy+p.centre_y_um*sc,r=p.radius_um*sc;
      svg.appendChild(svgEl('circle',{cx:pcx,cy:pcy,r,fill:'rgba(255,220,0,0.1)',stroke:col,'stroke-width':0.4}));
      const lbl=svgEl('text',{x:pcx,y:pcy-r-0.8,'text-anchor':'middle','font-size':1.8,fill:col});
      lbl.textContent='r='+p.radius_um.toFixed(1)+' µm';svg.appendChild(lbl);
    }
  });
  const badge=document.getElementById('badge');
  badge.textContent=s.milling_state||'IDLE';badge.className=s.milling_state||'IDLE';
  const fw=document.getElementById('fib-wrap');
  const banner=document.getElementById('click-banner');
  document.getElementById('click-prompt').textContent=s.click_prompt||'Click to select a point';
  if(s.waiting_for_click){fw.classList.add('waiting');banner.classList.add('on');}
  else{fw.classList.remove('waiting');banner.classList.remove('on');}
  const ab=document.getElementById('approval-banner');
  document.getElementById('approval-prompt').textContent=s.approval_prompt||'Approve milling?';
  document.getElementById('approval-params').textContent=s.approval_params||'';
  if(s.waiting_for_approval){ab.classList.add('on');}else{ab.classList.remove('on');}
  const sg=s.stage||{};
  const parts=[];
  if(sg.x_mm!=null)parts.push('x='+sg.x_mm.toFixed(3));
  if(sg.y_mm!=null)parts.push('y='+sg.y_mm.toFixed(3));
  if(sg.z_mm!=null)parts.push('z='+sg.z_mm.toFixed(3));
  if(sg.tilt_deg!=null)parts.push('t='+sg.tilt_deg.toFixed(1)+'°');
  document.getElementById('stage-info').textContent=parts.length?'Stage: '+parts.join('  '):'Stage: —';
  document.getElementById('ts').textContent=s.last_updated?'Updated: '+s.last_updated:'';
}
function handleApprove(){fetch('/dashboard/approval_response',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({approved:true})});}
function handleCancel(){fetch('/dashboard/approval_response',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({approved:false})});}
function handleClick(e){
  if(!st.waiting_for_click)return;
  const img=document.getElementById('fib-img');
  if(!img||img.style.display==='none')return;
  const r=img.getBoundingClientRect();
  const xf=(e.clientX-r.left)/r.width,yf=(e.clientY-r.top)/r.height;
  fetch('/dashboard/click',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({x_frac:xf,y_frac:yf})});
}
connect();
</script>
</body>
</html>"""


class _ClickBody(BaseModel):
    x_frac: float
    y_frac: float


class _RequestClickBody(BaseModel):
    prompt: str = "Click to select a point"


class _ApprovalRequestBody(BaseModel):
    prompt: str = "Approve milling?"
    params: str = ""


class _ApprovalResponseBody(BaseModel):
    approved: bool


class FibsemServer:
    def __init__(self, microscope: FibsemMicroscope, host: str = "0.0.0.0", port: int = 8001):
        self.microscope = microscope
        self.host = host
        self.port = port
        self._dash_state: dict = {
            "sem_jpeg": None, "fib_jpeg": None,
            "sem_hfw_um": 150.0, "sem_aspect": 0.75,
            "hfw_um": 150.0, "fib_aspect": 0.667,
            "patterns": [], "stage": {},
            "milling_state": "IDLE",
            "waiting_for_click": False,
            "click_prompt": "Click to select a point",
            "waiting_for_approval": False,
            "approval_prompt": "Approve milling?",
            "approval_params": "",
            "last_updated": "",
        }
        self._dash_version: int = 0
        self._ws_connections: Set[WebSocket] = set()
        self._click_queue: Optional[asyncio.Queue] = None
        self._approval_queue: Optional[asyncio.Queue] = None
        self.app = self._build_app()

    def _update_dash(self, **kwargs) -> None:
        self._dash_state.update(kwargs)
        self._dash_state["last_updated"] = datetime.now().strftime("%H:%M:%S")
        self._dash_version += 1

    def _build_app(self) -> FastAPI:
        app = FastAPI(title="FibsemMicroscope Server")
        microscope = self.microscope

        # --- Dashboard startup + routes ---

        @app.on_event("startup")
        async def _startup():
            self._click_queue = asyncio.Queue()
            self._approval_queue = asyncio.Queue()

        @app.get("/dashboard", response_class=HTMLResponse)
        def dashboard_page():
            return _DASHBOARD_HTML

        @app.websocket("/ws")
        async def ws_endpoint(websocket: WebSocket):
            await websocket.accept()
            self._ws_connections.add(websocket)
            last_ver = -1
            try:
                while True:
                    await asyncio.sleep(0.5)
                    if self._dash_version != last_ver:
                        await websocket.send_json(self._dash_state)
                        last_ver = self._dash_version
            except (WebSocketDisconnect, Exception):
                self._ws_connections.discard(websocket)

        @app.post("/dashboard/request_click")
        async def request_click(body: _RequestClickBody):
            self._update_dash(waiting_for_click=True, click_prompt=body.prompt)
            return {"status": "ok"}

        @app.post("/dashboard/click")
        async def handle_click(body: _ClickBody):
            self._dash_state["waiting_for_click"] = False
            self._dash_version += 1
            if self._click_queue is not None:
                await self._click_queue.put({"x_frac": body.x_frac, "y_frac": body.y_frac})
            return {"status": "ok"}

        @app.get("/dashboard/wait_for_click")
        async def wait_for_click(timeout: int = 60):
            if self._click_queue is None:
                raise HTTPException(status_code=503, detail="Server not ready")
            try:
                result = await asyncio.wait_for(self._click_queue.get(), timeout=float(timeout))
                self._update_dash(waiting_for_click=False)
                return result
            except asyncio.TimeoutError:
                self._update_dash(waiting_for_click=False)
                raise HTTPException(status_code=408, detail="Timeout waiting for click")

        @app.post("/dashboard/request_approval")
        async def request_approval(body: _ApprovalRequestBody):
            self._update_dash(
                waiting_for_approval=True,
                approval_prompt=body.prompt,
                approval_params=body.params,
            )
            return {"status": "ok"}

        @app.post("/dashboard/approval_response")
        async def approval_response(body: _ApprovalResponseBody):
            self._update_dash(waiting_for_approval=False)
            if self._approval_queue is not None:
                await self._approval_queue.put({"approved": body.approved})
            return {"status": "ok"}

        @app.get("/dashboard/wait_for_approval")
        async def wait_for_approval(timeout: int = 120):
            if self._approval_queue is None:
                raise HTTPException(status_code=503, detail="Server not ready")
            try:
                result = await asyncio.wait_for(self._approval_queue.get(), timeout=float(timeout))
                self._update_dash(waiting_for_approval=False)
                return result
            except asyncio.TimeoutError:
                self._update_dash(waiting_for_approval=False)
                raise HTTPException(status_code=408, detail="Timeout waiting for approval")

        @app.get("/dashboard/info")
        def dashboard_info():
            return {
                "hfw_um": self._dash_state["hfw_um"],
                "fib_aspect": self._dash_state["fib_aspect"],
                "stage": self._dash_state["stage"],
                "milling_state": self._dash_state["milling_state"],
                "patterns": self._dash_state["patterns"],
                "last_updated": self._dash_state["last_updated"],
            }

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
            image = microscope.acquire_image(image_settings=image_settings, beam_type=bt)
            b64 = _image_to_b64_jpeg(image)
            if b64 is not None:
                h, w = image.data.shape[:2]
                # Prefer image_settings.beam_type: FibsemClient always sends body.beam_type=ELECTRON
                # as the default, so body.beam_type alone is unreliable when image_settings is present.
                dash_bt = (image_settings.beam_type if image_settings and image_settings.beam_type else bt)
                key = "fib_jpeg" if dash_bt == BeamType.ION else "sem_jpeg"
                fallback_hfw_key = "hfw_um" if dash_bt == BeamType.ION else "sem_hfw_um"
                hfw = self._dash_state[fallback_hfw_key]
                if image.metadata and image.metadata.image_settings and image.metadata.image_settings.hfw:
                    hfw = image.metadata.image_settings.hfw * 1e6
                if dash_bt == BeamType.ION:
                    update = {key: b64, "hfw_um": hfw, "fib_aspect": h / w}
                else:
                    update = {key: b64, "sem_hfw_um": hfw, "sem_aspect": h / w}
                self._update_dash(**update)
            return _image_response(image)

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
            self._update_dash(stage=_stage_to_dash(result))
            return StagePositionResponse(position=result.to_dict())

        @app.post("/move_stage_relative", response_model=StagePositionResponse)
        def move_stage_relative(body: StagePositionRequest):
            result = microscope.move_stage_relative(FibsemStagePosition.from_dict(body.position))
            self._update_dash(stage=_stage_to_dash(result))
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
            pos = microscope.get_stage_position()
            self._update_dash(stage=_stage_to_dash(pos))
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
            pos = microscope.get_stage_position()
            self._update_dash(stage=_stage_to_dash(pos))
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
            dash_patterns = self._dash_state["patterns"][:]
            for p in body.patterns:
                d = _pattern_to_dash(p)
                if d is not None:
                    dash_patterns.append(d)
            self._update_dash(patterns=dash_patterns)
            return {"status": "ok"}

        @app.post("/run_milling")
        def run_milling(body: RunMillingRequest):
            self._update_dash(milling_state="RUNNING")
            microscope.run_milling(
                milling_current=body.milling_current,
                milling_voltage=body.milling_voltage,
                asynch=body.asynch,
            )
            if not body.asynch:
                self._update_dash(milling_state="IDLE")
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
            self._update_dash(patterns=[], milling_state="IDLE")
            return {"status": "ok"}

        @app.post("/clear_patterns")
        def clear_patterns():
            microscope.clear_patterns()
            self._update_dash(patterns=[])
            return {"status": "ok"}

        @app.get("/milling_state")
        def get_milling_state():
            return {"state": microscope.get_milling_state().name}

        @app.get("/estimate_milling_time")
        def estimate_milling_time():
            return {"seconds": microscope.estimate_milling_time()}

        @app.post("/link_stage")
        def link_stage():
            microscope.link_stage()
            return {"status": "ok"}

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
