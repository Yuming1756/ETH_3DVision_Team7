# ======================================================================
# graph.py section: 02_geometry.py
# Camera intrinsics/extrinsics, undistortion, depth<->color alignment, mask/depth lifting
#
# NOTE: not a standalone module. Loaded by graph.py into a shared
# namespace via exec(); names from earlier sections are in scope here.
# ======================================================================
from __future__ import annotations
def quaternion_to_R(x: float, y: float, z: float, w: float) -> np.ndarray:
    """Convert a quaternion into a 3x3 rotation matrix."""
    xx, yy, zz = x * x, y * y, z * z
    return np.array(
        [
            [1 - 2 * (yy + zz), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (xx + zz), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (xx + yy)],
        ],
        dtype=np.float64,
    )


def load_extrinsic(json_path: str | Path) -> np.ndarray:
    """Load a camera pose transform from disk."""
    with open(json_path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    transform = payload["data"]["transforms"][0]
    translation = transform["translation"]
    rotation = transform["rotation"]
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = quaternion_to_R(rotation["x"], rotation["y"], rotation["z"], rotation["w"])
    matrix[:3, 3] = [translation.get("x", 0.0), translation.get("y", 0.0), translation.get("z", 0.0)]
    return matrix


def matrix_to_rt(matrix: np.ndarray) -> Tuple[Rotation, np.ndarray]:
    """Convert a 4x4 rigid transform into rotation and translation."""
    matrix = np.asarray(matrix, dtype=np.float64)
    return Rotation.from_matrix(matrix[:3, :3]), matrix[:3, 3].astype(np.float64)


def rt_to_matrix(rotation: Rotation, translation: np.ndarray) -> np.ndarray:
    """Convert rotation and translation into a 4x4 rigid transform."""
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = rotation.as_matrix()
    matrix[:3, 3] = np.asarray(translation, dtype=np.float64)
    return matrix


def load_intrinsic(json_path: str | Path) -> Tuple[np.ndarray, np.ndarray, int, int]:
    """Load camera intrinsics and distortion parameters from disk."""
    with open(json_path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    info = payload["data"]
    K = np.array(info["K"], dtype=np.float64).reshape(3, 3)
    D = np.array(info["D"], dtype=np.float64)
    return K, D, int(info["width"]), int(info["height"])


def load_static_transform_components(json_path: str | Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load camera-to-depth, depth-to-color, and camera-to-color static transforms."""
    with open(json_path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    transforms = {(item["parent_frame_id"], item["child_frame_id"]): item for item in payload["data"]["transforms"]}

    def _make_T(item: dict) -> np.ndarray:
        translation = item.get("translation", {})
        rotation = item.get("rotation", {})
        matrix = np.eye(4, dtype=np.float64)
        matrix[:3, :3] = quaternion_to_R(
            rotation.get("x", 0.0),
            rotation.get("y", 0.0),
            rotation.get("z", 0.0),
            rotation.get("w", 1.0),
        )
        matrix[:3, 3] = [translation.get("x", 0.0), translation.get("y", 0.0), translation.get("z", 0.0)]
        return matrix

    T_camera_depth = _make_T(transforms[("camera", "depth")])
    T_depth_color = _make_T(transforms[("depth", "color")])
    T_camera_color = T_camera_depth @ T_depth_color
    return T_camera_depth, T_depth_color, T_camera_color


def load_static_transforms(json_path: str | Path) -> np.ndarray:
    """Backward-compatible wrapper returning the camera-to-color transform."""
    return load_static_transform_components(json_path)[2]


def undistort_rgb(rgb: np.ndarray, K: np.ndarray, D: np.ndarray) -> np.ndarray:
    """Undistort an RGB frame into the color camera space."""
    if rgb.size == 0:
        return rgb.copy()
    return cv2.undistort(rgb, K.astype(np.float64), D.astype(np.float64))


def undistort_metric_image(image: np.ndarray, K: np.ndarray, D: np.ndarray, interpolation: int = cv2.INTER_NEAREST) -> np.ndarray:
    """Undistort a float image such as depth into the same pinhole image space as the RGB frames."""
    if image.size == 0:
        return image.copy()
    if D.size == 0 or not np.any(np.abs(D) > 1e-12):
        return np.asarray(image, copy=True)
    image_h, image_w = image.shape[:2]
    map_x, map_y = cv2.initUndistortRectifyMap(
        K.astype(np.float64),
        D.astype(np.float64),
        None,
        K.astype(np.float64),
        (image_w, image_h),
        cv2.CV_32FC1,
    )
    return cv2.remap(
        np.asarray(image, dtype=np.float32),
        map_x,
        map_y,
        interpolation=interpolation,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )


def distorted_pixels_to_undistorted(pixels: np.ndarray, K: np.ndarray, D: np.ndarray) -> np.ndarray:
    """Map raw distorted color-image pixels into the undistorted pinhole image coordinates."""
    pixels = np.asarray(pixels, dtype=np.float32)
    if pixels.size == 0:
        return np.zeros((0, 2), dtype=np.float32)
    if D.size == 0 or not np.any(np.abs(D) > 1e-12):
        return pixels.astype(np.float32, copy=True)
    undistorted = cv2.undistortPoints(
        pixels.reshape(-1, 1, 2).astype(np.float64),
        K.astype(np.float64),
        D.astype(np.float64),
        P=K.astype(np.float64),
    )
    return undistorted.reshape(-1, 2).astype(np.float32)


def align_depth_to_color(
    sensor_raw: np.ndarray,
    K_depth: np.ndarray,
    T_depth_color: np.ndarray,
    K_color: np.ndarray,
    color_w: int,
    color_h: int,
    depth_scale: Optional[float] = None,
    depth_max: float = DEPTH_MAX,
) -> np.ndarray:
    """Project the raw depth sensor image into the color camera image grid."""
    depth_scale = resolve_depth_scale(depth_scale)
    depth_m = sensor_raw.astype(np.float32) / depth_scale
    valid = (depth_m > 1e-4) & (depth_m < depth_max)
    if not np.any(valid):
        return np.zeros((color_h, color_w), dtype=np.float32)

    ys, xs = np.where(valid)
    z = depth_m[ys, xs]
    fx_d, fy_d, cx_d, cy_d = K_depth[0, 0], K_depth[1, 1], K_depth[0, 2], K_depth[1, 2]
    x_d = (xs.astype(np.float32) - cx_d) * z / fx_d
    y_d = (ys.astype(np.float32) - cy_d) * z / fy_d
    pts_depth = np.stack([x_d, y_d, z], axis=1)
    pts_depth_h = np.concatenate([pts_depth, np.ones((len(pts_depth), 1), dtype=np.float32)], axis=1)
    # tf_static stores depth_T_color (the color frame expressed in depth).
    # We need color_T_depth to project native depth pixels into the
    # undistorted color/SAM image grid.
    T_color_depth = np.asarray(np.linalg.inv(T_depth_color), dtype=np.float32)
    pts_color = (T_color_depth @ pts_depth_h.T).T[:, :3]

    valid_color = pts_color[:, 2] > 1e-4
    pts_color = pts_color[valid_color]
    if len(pts_color) == 0:
        return np.zeros((color_h, color_w), dtype=np.float32)

    fx_c, fy_c, cx_c, cy_c = K_color[0, 0], K_color[1, 1], K_color[0, 2], K_color[1, 2]
    us = np.rint((pts_color[:, 0] * fx_c / pts_color[:, 2]) + cx_c).astype(np.int32)
    vs = np.rint((pts_color[:, 1] * fy_c / pts_color[:, 2]) + cy_c).astype(np.int32)
    inside = (us >= 0) & (us < color_w) & (vs >= 0) & (vs < color_h)
    if not np.any(inside):
        return np.zeros((color_h, color_w), dtype=np.float32)

    us = us[inside]
    vs = vs[inside]
    zs = pts_color[:, 2][inside].astype(np.float32)
    flat = vs * color_w + us
    order = np.argsort(zs)
    flat_sorted = flat[order]
    z_sorted = zs[order]
    unique_flat, first_idx = np.unique(flat_sorted, return_index=True)
    aligned = np.zeros(color_h * color_w, dtype=np.float32)
    aligned[unique_flat] = z_sorted[first_idx]
    return aligned.reshape(color_h, color_w)


def prepare_sensor_depth_for_lifting(
    sensor_raw: np.ndarray,
    K_depth: np.ndarray,
    T_depth_color: np.ndarray,
    K_color: np.ndarray,
    D_color: np.ndarray,
    color_w: int,
    color_h: int,
    depth_scale: Optional[float] = None,
) -> np.ndarray:
    """Prepare raw sensor depth for lifting in the same undistorted image space as RGB frames."""
    aligned_depth = align_depth_to_color(
        sensor_raw,
        K_depth,
        T_depth_color,
        K_color,
        color_w,
        color_h,
        depth_scale=depth_scale,
    )
    # align_depth_to_color already projects into the same undistorted pinhole
    # color grid used by undistort_rgb(...), so do not undistort it again.
    return aligned_depth


def build_depth_backprojection_grid(K_depth: np.ndarray, depth_h: int, depth_w: int) -> np.ndarray:
    """Precompute depth-camera rays for native-depth backprojection."""
    xs, ys = np.meshgrid(
        np.arange(depth_w, dtype=np.float32) + 0.5,
        np.arange(depth_h, dtype=np.float32) + 0.5,
        indexing="xy",
    )
    fx, fy, cx, cy = K_depth[0, 0], K_depth[1, 1], K_depth[0, 2], K_depth[1, 2]
    return np.stack(
        [
            (xs - cx) / fx,
            (ys - cy) / fy,
            np.ones_like(xs, dtype=np.float32),
        ],
        axis=-1,
    ).astype(np.float32)


def colorize_depth(depth_m: np.ndarray, depth_max: float = DEPTH_MAX) -> np.ndarray:
    """Convert a metric depth map into a color image for quick visual debugging."""
    valid = depth_m > 1e-4
    if not np.any(valid):
        return np.zeros((*depth_m.shape, 3), dtype=np.uint8)
    vmax = min(depth_max, float(np.percentile(depth_m[valid], 95)))
    if vmax <= 1e-4:
        vmax = depth_max
    normalized = np.clip(depth_m / max(vmax, 1e-4), 0.0, 1.0)
    colored = cv2.applyColorMap((normalized * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    colored[~valid] = 0
    return cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)


def overlay_depth_points(
    rgb: np.ndarray,
    depth_m: np.ndarray,
    stride: int = 20,
    radius: int = 1,
) -> np.ndarray:
    """Overlay sparse valid depth samples on top of an RGB image."""
    overlay = rgb.copy()
    sampled = np.zeros_like(depth_m, dtype=bool)
    sampled[::stride, ::stride] = True
    valid = (depth_m > 1e-4) & sampled
    if not np.any(valid):
        return overlay
    colors = colorize_depth(depth_m)
    ys, xs = np.where(valid)
    for y, x in zip(ys.tolist(), xs.tolist()):
        cv2.circle(overlay, (int(x), int(y)), radius, tuple(int(v) for v in colors[y, x]), thickness=-1)
    return overlay


def rgb_canny_edges(rgb: np.ndarray) -> np.ndarray:
    """Compute a simple Canny edge map for one RGB image."""
    gray = cv2.cvtColor(np.asarray(rgb, dtype=np.uint8), cv2.COLOR_RGB2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    return cv2.Canny(gray, 80, 160)


def depth_canny_edges(depth_m: np.ndarray, depth_max: float = DEPTH_MAX) -> np.ndarray:
    """Compute a simple Canny edge map for one metric depth image."""
    valid = np.asarray(depth_m, dtype=np.float32) > 1e-4
    if not np.any(valid):
        return np.zeros(depth_m.shape[:2], dtype=np.uint8)
    vmax = min(depth_max, float(np.percentile(depth_m[valid], 95)))
    if vmax <= 1e-4:
        vmax = depth_max
    normalized = np.clip(np.asarray(depth_m, dtype=np.float32) / max(vmax, 1e-4), 0.0, 1.0)
    depth_u8 = (normalized * 255).astype(np.uint8)
    depth_u8[~valid] = 0
    depth_u8 = cv2.GaussianBlur(depth_u8, (5, 5), 0)
    return cv2.Canny(depth_u8, 40, 120)


def overlay_rgb_depth_edges(rgb: np.ndarray, rgb_edges: np.ndarray, depth_edges: np.ndarray) -> np.ndarray:
    """Overlay RGB and depth edges to visually check alignment/leakage."""
    canvas = np.asarray(rgb, dtype=np.uint8).copy()
    rgb_mask = np.asarray(rgb_edges, dtype=np.uint8) > 0
    depth_mask = np.asarray(depth_edges, dtype=np.uint8) > 0
    overlap = rgb_mask & depth_mask
    rgb_only = rgb_mask & ~depth_mask
    depth_only = depth_mask & ~rgb_mask
    canvas[rgb_only] = np.array([255, 80, 80], dtype=np.uint8)
    canvas[depth_only] = np.array([80, 255, 255], dtype=np.uint8)
    canvas[overlap] = np.array([255, 255, 255], dtype=np.uint8)
    return canvas


def lift_mask_to_world(
    mask: np.ndarray,
    depth_m: np.ndarray,
    rgb: np.ndarray,
    K: np.ndarray,
    T_wc: np.ndarray,
    T_cam_color: np.ndarray,
    depth_max: float = DEPTH_MAX,
) -> Tuple[np.ndarray, np.ndarray]:
    """Back-project masked color pixels into world-space points and colors."""
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.float32)

    z = depth_m[ys, xs].astype(np.float32)
    valid = (z > 1e-4) & (z < depth_max)
    xs = xs[valid].astype(np.float32)
    ys = ys[valid].astype(np.float32)
    z = z[valid]
    if len(z) == 0:
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.float32)

    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    pts_c = np.stack([(xs - cx) * z / fx, (ys - cy) * z / fy, z], axis=1)
    pts_h = np.concatenate([pts_c, np.ones((len(pts_c), 1), dtype=np.float32)], axis=1)
    world_from_color = T_wc @ T_cam_color
    pts_w = (world_from_color @ pts_h.T).T[:, :3].astype(np.float32)

    xi = np.clip(xs.astype(np.int32), 0, rgb.shape[1] - 1)
    yi = np.clip(ys.astype(np.int32), 0, rgb.shape[0] - 1)
    cols = rgb[yi, xi].astype(np.float32) / 255.0
    return pts_w, cols


def lift_depth_frame_to_world_with_rgb(
    depth_raw: np.ndarray,
    rgb: np.ndarray,
    depth_grid: np.ndarray,
    K_color: np.ndarray,
    T_wc: np.ndarray,
    T_cam_depth: np.ndarray,
    T_cam_color: np.ndarray,
    D_color: Optional[np.ndarray] = None,
    depth_mask: Optional[np.ndarray] = None,
    exclusion_mask: Optional[np.ndarray] = None,
    aligned_color_depth: Optional[np.ndarray] = None,
    pixel_stride: int = INACTIVE_DEPTH_PIXEL_STRIDE,
    depth_scale: Optional[float] = None,
    depth_max: float = DEPTH_MAX,
) -> Tuple[np.ndarray, np.ndarray]:
    """Lift one depth frame in the native depth camera frame and color it from RGB.

    If ``aligned_color_depth`` is provided, it is treated as a refined color-camera
    depth source (for example DA3 depth). We still keep the native depth-camera
    lifting path and only replace the per-ray metric depth.
    """
    depth_scale = resolve_depth_scale(depth_scale)
    depth_m = depth_raw.astype(np.float32) / depth_scale
    if pixel_stride > 1:
        depth_m = depth_m[::pixel_stride, ::pixel_stride]
        rays = depth_grid[::pixel_stride, ::pixel_stride]
        depth_mask_view = np.asarray(depth_mask, dtype=bool)[::pixel_stride, ::pixel_stride] if depth_mask is not None else None
    else:
        rays = depth_grid
        depth_mask_view = np.asarray(depth_mask, dtype=bool) if depth_mask is not None else None

    valid_depth = (depth_m > 1e-4) & (depth_m < depth_max)
    if depth_mask_view is not None:
        valid_depth &= depth_mask_view
    if not np.any(valid_depth):
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.float32)

    valid_indices = np.where(valid_depth)
    rays_valid = rays[valid_indices]
    depth_values = depth_m[valid_indices].astype(np.float32)

    T_depth_color = np.asarray(np.linalg.inv(T_cam_color) @ T_cam_depth, dtype=np.float32)
    R_depth_color = T_depth_color[:3, :3]
    t_depth_color = T_depth_color[:3, 3]
    pts_depth_sensor = rays_valid * depth_values[:, None]
    pts_color_sensor = (R_depth_color @ pts_depth_sensor.T).T + t_depth_color[None, :]
    valid_color_sensor = pts_color_sensor[:, 2] > 1e-4
    if not np.any(valid_color_sensor):
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.float32)

    rays_valid = rays_valid[valid_color_sensor]
    depth_values = depth_values[valid_color_sensor]
    pts_color_sensor = pts_color_sensor[valid_color_sensor]

    pixels_h = (np.asarray(K_color, dtype=np.float32) @ pts_color_sensor.T).T
    pixels_raw = np.floor(pixels_h[:, :2] / pixels_h[:, 2:3]).astype(np.float32)
    if D_color is not None:
        pixels_color = np.floor(distorted_pixels_to_undistorted(pixels_raw, K_color, np.asarray(D_color, dtype=np.float64))).astype(np.int32)
    else:
        pixels_color = pixels_raw.astype(np.int32)
    image_h, image_w = rgb.shape[:2]
    inside = (
        (pixels_color[:, 0] >= 0)
        & (pixels_color[:, 0] < image_w)
        & (pixels_color[:, 1] >= 0)
        & (pixels_color[:, 1] < image_h)
    )
    if not np.any(inside):
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.float32)

    rays_valid = rays_valid[inside]
    depth_values = depth_values[inside]
    pts_color_sensor = pts_color_sensor[inside]
    pixels_color = pixels_color[inside]

    # Multiple native-depth pixels can project to the same undistorted color
    # pixel. Keep the closest color-camera sample before testing SAM masks;
    # otherwise background/table depth can leak through thin object masks.
    flat_pixels = pixels_color[:, 1] * image_w + pixels_color[:, 0]
    order = np.argsort(pts_color_sensor[:, 2])
    _unique_flat, first_sorted_indices = np.unique(flat_pixels[order], return_index=True)
    visible_keep = np.sort(order[first_sorted_indices])
    rays_valid = rays_valid[visible_keep]
    depth_values = depth_values[visible_keep]
    pixels_color = pixels_color[visible_keep]

    if aligned_color_depth is not None:
        aligned_depth = np.asarray(aligned_color_depth, dtype=np.float32)
        sampled_color_depth = aligned_depth[pixels_color[:, 1], pixels_color[:, 0]]
        valid_aligned = (sampled_color_depth > 1e-4) & (sampled_color_depth < depth_max)
        if np.any(valid_aligned):
            denom = (rays_valid @ R_depth_color[2, :].astype(np.float32))
            safe = np.abs(denom) > 1e-6
            recover = valid_aligned & safe
            if np.any(recover):
                recovered_depth = (sampled_color_depth[recover] - float(t_depth_color[2])) / denom[recover]
                positive = recovered_depth > 1e-4
                if np.any(positive):
                    recover_indices = np.where(recover)[0][positive]
                    depth_values[recover_indices] = recovered_depth[positive].astype(np.float32)

    pts_depth = rays_valid * depth_values[:, None]
    world_from_depth = np.asarray(T_wc @ T_cam_depth, dtype=np.float32)
    pts_world = apply_homogeneous_transform(pts_depth, world_from_depth)
    if len(pts_world) == 0:
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.float32)

    if exclusion_mask is not None:
        exclusion_values = np.asarray(exclusion_mask, dtype=bool)[pixels_color[:, 1], pixels_color[:, 0]]
        keep = ~exclusion_values
        if not np.any(keep):
            return np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.float32)
        pts_world = pts_world[keep]
        pixels_color = pixels_color[keep]

    colors = rgb[pixels_color[:, 1], pixels_color[:, 0]].astype(np.float32) / 255.0
    return pts_world.astype(np.float32), colors.astype(np.float32)


