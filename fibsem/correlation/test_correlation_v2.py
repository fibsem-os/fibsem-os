"""Test script for run_correlation_from_data.

Loads real test images and coordinates, runs the correlation, and prints the result.

Usage
-----
    python fibsem/correlation/test_correlation_v2.py
"""
import os
import pandas as pd

from fibsem.correlation.structures import (
    Coordinate,
    CorrelationInputData,
    PointType,
    PointXYZ,
)
from fibsem.correlation.correlation_v2 import run_correlation_from_data
from fibsem.structures import FibsemImage
from fibsem.fm.structures import FluorescenceImage

DEV_PATH = "/home/patrick/github/fibsem/fibsem/applications/test-data"
DEV_FIB_IMAGE = "ref_ReferenceImage-Spot-Burn-Fiducial-10-36-30_res_02_ib.tif"
DEV_FM_IMAGE = "zstack-Feature-1-Active-002.ome.tiff"
DEV_CSV = "data2.csv"

_TYPE_MAP = {
    "FIB": PointType.FIB,
    "FM": PointType.FM,
    "POI": PointType.POI,
    "Surface": PointType.SURFACE,
}


def load_coordinates(csv_path: str) -> list[Coordinate]:
    df = pd.read_csv(csv_path)
    coords = []
    for _, row in df.iterrows():
        pt = _TYPE_MAP.get(row["type"])
        if pt is None:
            continue
        coords.append(Coordinate(PointXYZ(row["x"], row["y"], row["z"]), pt))
    return coords


def main():
    # Load images
    fib_image = FibsemImage.load(os.path.join(DEV_PATH, DEV_FIB_IMAGE))
    fm_image = FluorescenceImage.load(os.path.join(DEV_PATH, DEV_FM_IMAGE))

    print(f"FIB image shape:  {fib_image.data.shape}")
    print(f"FIB pixel size:   {fib_image.metadata.pixel_size.x:.3e} m")
    print(f"FM  image shape:  {fm_image.data.shape}")

    # Parse coordinates
    all_coords = load_coordinates(os.path.join(DEV_PATH, DEV_CSV))
    fib = [c for c in all_coords if c.point_type == PointType.FIB]
    fm  = [c for c in all_coords if c.point_type == PointType.FM]
    poi = [c for c in all_coords if c.point_type == PointType.POI]
    surface = next((c for c in all_coords if c.point_type == PointType.SURFACE), None)

    print(f"\nCoordinates: {len(fib)} FIB, {len(fm)} FM, {len(poi)} POI, surface={'yes' if surface else 'no'}")

    data = CorrelationInputData(
        fib_image=fib_image,
        fm_image=fm_image,
        fib_coordinates=fib,
        fm_coordinates=fm,
        poi_coordinates=poi,
        surface_coordinate=surface,
    )

    data.save(os.path.join(DEV_PATH, "correlation_input_data.json"))


    print(f"\nimage_props will be: fib_shape={data.fib_image_shape}, pixel_size_um={data.fib_image_pixel_size * 1e6:.4f}, fm_shape={data.fm_image_shape}")

    # Run correlation
    print("\nRunning correlation...")
    result = run_correlation_from_data(data)

    print(f"\n--- CorrelationResult ---")
    print(f"scale:               {result.scale:.6f}")
    print(f"rms_error:           {result.rms_error:.4f}")
    print(f"mean_absolute_error: {[f'{v:.4f}' for v in result.mean_absolute_error]}")
    print(f"rotation_eulers:     {[f'{v:.2f}' for v in result.rotation_eulers]}")

    print(f"\nReprojected 3D markers into 2D ({len(result.reprojected_3d)} markers):")
    for i, pt in enumerate(result.reprojected_3d):
        print(f"  marker {i+1}: ({pt.x:.2f}, {pt.y:.2f}, {pt.z:.2f})")

    print("\nDelta 2D per marker (reprojection error, pixels):")
    for i, d in enumerate(result.delta_2d):
        print(f"  marker {i+1}: dx={d.x:.3f}  dy={d.y:.3f}")

    print(f"\nPOI ({len(result.poi)} points):")
    for i, p in enumerate(result.poi):
        print(f"  POI {i+1}: image_px=({p.image_px.x:.1f}, {p.image_px.y:.1f})  px_m=({p.px_m.x:.3e}, {p.px_m.y:.3e} m)")


if __name__ == "__main__":
    main()
