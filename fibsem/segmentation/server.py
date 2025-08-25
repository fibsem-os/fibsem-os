#!/usr/bin/env python3
"""
FastAPI server for remote segmentation model inference.
"""

import base64
import io
import logging
import psutil
import time
from typing import Dict, Optional, Any, List

import numpy as np
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from pydantic import BaseModel, Field

from pathlib import Path
from fibsem.segmentation.model import load_model
from fibsem.config import DEFAULT_CHECKPOINT

# Global model cache and stats
model_cache: Dict[str, Any] = {}
server_stats = {
    "start_time": time.time(),
    "requests_processed": 0,
    "total_processing_time": 0.0,
    "errors": 0
}

# Configuration
MAX_IMAGE_SIZE = 4096  # Maximum image dimension


class SegmentationRequest(BaseModel):
    """Request schema for segmentation"""
    image: str = Field(..., description="Base64 encoded image data")
    return_rgb: bool = Field(default=True, description="Return RGB mask")


class SegmentationResponse(BaseModel):
    """Response schema for segmentation"""
    success: bool
    mask: Optional[str] = Field(None, description="Base64 encoded mask data")
    processing_time_ms: float
    error: Optional[str] = None


class SystemInfo(BaseModel):
    """System information schema"""
    cpu_percent: float
    memory_percent: float
    available_memory_gb: float
    disk_usage_percent: float


class GPUInfo(BaseModel):
    """GPU information schema"""
    available: bool
    device_count: int = 0
    devices: List[Dict[str, Any]] = []


class ModelStatus(BaseModel):
    """Model status schema"""
    loaded: bool
    checkpoint: Optional[str] = None
    load_time: Optional[float] = None
    backend: Optional[str] = None


class ServerStats(BaseModel):
    """Server statistics schema"""
    uptime_seconds: float
    requests_processed: int
    total_processing_time: float
    average_processing_time: float
    errors: int
    requests_per_second: float


class HealthResponse(BaseModel):
    """Comprehensive health check response schema"""
    status: str
    timestamp: float
    system: SystemInfo
    gpu: GPUInfo
    model: ModelStatus
    stats: ServerStats


# Initialize FastAPI app
app = FastAPI(
    title="FibSEM Segmentation Server",
    description="Remote inference server for FibSEM segmentation models",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def encode_array_to_base64(arr: np.ndarray, is_mask: bool = False) -> str:
    """Encode numpy array to base64 string"""
    if arr.dtype != np.uint8:
        if is_mask:
            # For class masks, don't normalize - just convert to uint8
            arr = arr.astype(np.uint8)
        elif arr.max() <= 1.0:
            arr = (arr * 255).astype(np.uint8)
        else:
            arr = arr.astype(np.uint8)
    
    img = Image.fromarray(arr)
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode()


def decode_base64_to_array(data: str) -> np.ndarray:
    """Decode base64 string to numpy array"""
    try:
        image_data = base64.b64decode(data)
        img = Image.open(io.BytesIO(image_data))
        arr = np.array(img)

        logging.info(arr.shape)
        
        # Ensure we have a 2D grayscale image
        if arr.ndim == 3 and arr.shape[2] == 1:
            arr = arr.squeeze(2)  # Remove single channel dimension
        elif arr.ndim == 3 and arr.shape[2] == 3:
            # Convert RGB to grayscale
            arr = np.dot(arr[...,:3], [0.2989, 0.5870, 0.1140]).astype(arr.dtype)
        
        return arr
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image data: {str(e)}")


def validate_image(img: np.ndarray) -> None:
    """Validate image dimensions and format"""
    if img.ndim not in [2, 3]:
        raise HTTPException(status_code=400, detail="Image must be 2D or 3D")
    
    if max(img.shape[:2]) > MAX_IMAGE_SIZE:
        raise HTTPException(
            status_code=400, 
            detail=f"Image too large. Max dimension: {MAX_IMAGE_SIZE}"
        )


def load_model_if_needed() -> Any:
    """Load default model into cache if not already loaded"""
    if "model" in model_cache:
        return model_cache["model"]
    
    try:
        start_time = time.time()
        model = load_model(checkpoint=DEFAULT_CHECKPOINT)
        load_time = time.time() - start_time
        
        model_cache["model"] = model
        logging.info(f"Loaded model {DEFAULT_CHECKPOINT} in {load_time:.2f}s")
        return model
        
    except Exception as e:
        logging.error(f"Failed to load model {DEFAULT_CHECKPOINT}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to load model: {str(e)}")


def get_gpu_info() -> GPUInfo:
    """Get GPU information"""
    try:
        import torch
        if torch.cuda.is_available():
            device_count = torch.cuda.device_count()
            devices = []
            for i in range(device_count):
                props = torch.cuda.get_device_properties(i)
                devices.append({
                    "id": i,
                    "name": props.name,
                    "memory_total_mb": int(props.total_memory),  # Already in MB
                    "memory_allocated_mb": torch.cuda.memory_allocated(i) // (1024 * 1024),
                })
            return GPUInfo(available=True, device_count=device_count, devices=devices)
        else:
            return GPUInfo(available=False)
    except ImportError:
        return GPUInfo(available=False)


def get_system_info() -> SystemInfo:
    """Get system resource information"""
    memory = psutil.virtual_memory()
    disk = psutil.disk_usage('/')
    
    return SystemInfo(
        cpu_percent=psutil.cpu_percent(interval=0.1),
        memory_percent=memory.percent,
        available_memory_gb=memory.available / (1024**3),
        disk_usage_percent=disk.percent
    )


def get_model_status() -> ModelStatus:
    """Get model loading status"""
    if "model" in model_cache:
        model = model_cache["model"]
        return ModelStatus(
            loaded=True,
            checkpoint=DEFAULT_CHECKPOINT,
            load_time=getattr(model, 'load_time', None),
            backend="pytorch"  # Could be extracted from model metadata
        )
    else:
        return ModelStatus(loaded=False)


def get_server_stats() -> ServerStats:
    """Get server statistics"""
    current_time = time.time()
    uptime = current_time - server_stats["start_time"]
    
    avg_processing_time = (
        server_stats["total_processing_time"] / server_stats["requests_processed"]
        if server_stats["requests_processed"] > 0 else 0.0
    )
    
    requests_per_second = server_stats["requests_processed"] / uptime if uptime > 0 else 0.0
    
    return ServerStats(
        uptime_seconds=uptime,
        requests_processed=server_stats["requests_processed"],
        total_processing_time=server_stats["total_processing_time"],
        average_processing_time=avg_processing_time,
        errors=server_stats["errors"],
        requests_per_second=requests_per_second
    )


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Comprehensive health check endpoint"""
    try:
        system_info = get_system_info()
        gpu_info = get_gpu_info()
        model_status = get_model_status()
        stats = get_server_stats()
        # Determine overall status
        status = "healthy"
        if system_info.memory_percent > 90:
            status = "warning"
        if not gpu_info.available and model_status.loaded:
            status = "warning"
        if system_info.cpu_percent > 95:
            status = "degraded"
            
        return HealthResponse(
            status=status,
            timestamp=time.time(),
            system=system_info,
            gpu=gpu_info,
            model=model_status,
            stats=stats
        )
        
    except Exception as e:
        logging.error(f"Health check failed: {e}")
        # Return minimal health response
        return HealthResponse(
            status="error",
            timestamp=time.time(),
            system=SystemInfo(cpu_percent=0, memory_percent=0, available_memory_gb=0, disk_usage_percent=0),
            gpu=GPUInfo(available=False),
            model=ModelStatus(loaded=False),
            stats=ServerStats(uptime_seconds=0, requests_processed=0, total_processing_time=0, 
                            average_processing_time=0, errors=0, requests_per_second=0)
        )


@app.post("/segment", response_model=SegmentationResponse)
async def segment_image(request: SegmentationRequest):
    """Perform segmentation on a single image"""
    start_time = time.time()

    try:
        # Decode and validate image
        img = decode_base64_to_array(request.image)
        logging.info(f"Decoded image shape: {img.shape}, dtype: {img.dtype}")
        validate_image(img)
        
        # Load model if needed
        model = load_model_if_needed()
        
        # Perform inference
        logging.info(f"Performing inference on image shape: {img.shape}")
        masks = model.inference(img, rgb=request.return_rgb)
        
        # Handle batch dimension for non-RGB masks
        if not request.return_rgb and isinstance(masks, np.ndarray) and masks.ndim > 2:
            masks = masks[0]  # Extract first element from batch

        # Encode output and prepare response
        is_class_mask = not request.return_rgb
        encoded_mask = encode_array_to_base64(masks, is_mask=is_class_mask)
        
        processing_time = (time.time() - start_time) * 1000
        
        # Update server statistics
        server_stats["requests_processed"] += 1
        server_stats["total_processing_time"] += processing_time / 1000  # Convert to seconds
        
        response = SegmentationResponse(
            success=True,
            mask=encoded_mask,
            processing_time_ms=processing_time
        )
        
        return response
        
    except HTTPException:
        # Update error count
        server_stats["errors"] += 1
        raise
    except Exception as e:
        # Update error count
        server_stats["errors"] += 1
        processing_time = (time.time() - start_time) * 1000
        
        logging.error(f"Segmentation error: {str(e)}")
        return SegmentationResponse(
            success=False,
            mask=None,
            processing_time_ms=processing_time,
            error=str(e)
        )


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    # Run server
    uvicorn.run(
        "server:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info"
    )