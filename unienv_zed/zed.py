from __future__ import annotations

from enum import Enum
from typing import Any, Dict, Optional, TypeVar

import numpy as np
import pyzed.sl as sl

from unienv_interface.world import RealWorld, World, WorldNode
from unienv_interface.backends import ComputeBackend
from unienv_interface.backends.numpy import (
    NumpyArrayType,
    NumpyComputeBackend,
    NumpyDeviceType,
    NumpyDtypeType,
    NumpyRNGType,
)
from unienv_interface.space import BoxSpace, DictSpace

__all__ = [
    "ZedCameraNode",
    "ZedCameraSensorNode",
    "ZedResolution",
    "ZedDepthMode",
    "ZedFlipMode",
    "ZedUnit",
    "list_connected_zed_cameras",
]


class ZedResolution(str, Enum):
    AUTO = "AUTO"
    HD2K = "HD2K"
    HD1200 = "HD1200"
    HD1080 = "HD1080"
    HD720 = "HD720"
    SVGA = "SVGA"
    VGA = "VGA"


class ZedDepthMode(str, Enum):
    NONE = "NONE"
    PERFORMANCE = "PERFORMANCE"
    QUALITY = "QUALITY"
    ULTRA = "ULTRA"
    NEURAL_LIGHT = "NEURAL_LIGHT"
    NEURAL = "NEURAL"
    NEURAL_PLUS = "NEURAL_PLUS"


class ZedFlipMode(str, Enum):
    OFF = "OFF"
    ON = "ON"
    AUTO = "AUTO"


class ZedUnit(str, Enum):
    METER = "METER"
    CENTIMETER = "CENTIMETER"
    MILLIMETER = "MILLIMETER"
    INCH = "INCH"
    FOOT = "FOOT"


EnumT = TypeVar("EnumT", bound=Enum)


def _coerce_enum(value: str | EnumT, enum_cls: type[EnumT], name: str) -> EnumT:
    if isinstance(value, enum_cls):
        return value
    if isinstance(value, str):
        try:
            return enum_cls[value.upper()]
        except KeyError as exc:
            valid = ", ".join(member.name for member in enum_cls)
            raise ValueError(f"Unsupported {name}: {value!r}. Expected one of: {valid}") from exc
    raise TypeError(f"{name} must be a str or {enum_cls.__name__}, got {type(value).__name__}")


def _resolve_sl_enum(namespace: Any, value: str | Enum, enum_cls: type[EnumT], name: str) -> Any:
    member = _coerce_enum(value, enum_cls, name)
    if not hasattr(namespace, member.name):
        valid = [member_name for member_name in enum_cls.__members__ if hasattr(namespace, member_name)]
        raise ValueError(
            f"Installed ZED SDK does not support {name}={member.name!r}. "
            f"Available SDK values: {valid}"
        )
    return getattr(namespace, member.name)


def _default_supported_depth_mode() -> ZedDepthMode:
    for candidate in (
        ZedDepthMode.NEURAL,
        ZedDepthMode.NEURAL_LIGHT,
        ZedDepthMode.ULTRA,
        ZedDepthMode.QUALITY,
        ZedDepthMode.PERFORMANCE,
    ):
        if hasattr(sl.DEPTH_MODE, candidate.name):
            return candidate
    return ZedDepthMode.NONE


def _box_rgb(height: int, width: int) -> BoxSpace:
    return BoxSpace(
        NumpyComputeBackend,
        low=0,
        high=255,
        dtype=np.uint8,
        shape=(height, width, 3),
    )


def _box_depth(height: int, width: int) -> BoxSpace:
    return BoxSpace(
        NumpyComputeBackend,
        low=0.0,
        high=np.inf,
        dtype=np.float32,
        shape=(height, width),
    )


def _box_intrinsics() -> BoxSpace:
    return BoxSpace(
        NumpyComputeBackend,
        low=-np.inf,
        high=np.inf,
        dtype=np.float32,
        shape=(3, 3),
    )


def _zed_image_to_rgb(image: Any) -> np.ndarray:
    arr = np.asarray(image.get_data())
    if arr.ndim != 3 or arr.shape[2] not in (3, 4):
        raise ValueError(f"Unexpected ZED image shape: {arr.shape}")
    rgb = arr[..., [2, 1, 0]]
    return np.array(rgb, dtype=np.uint8, copy=True)


def _zed_depth_to_array(depth: Any, invalid_value: float = 0.0) -> np.ndarray:
    arr = np.array(depth.get_data(), dtype=np.float32, copy=True)
    return np.nan_to_num(arr, nan=invalid_value, posinf=invalid_value, neginf=invalid_value, copy=False)


def _zed_camera_intrinsics(params: Any) -> np.ndarray:
    return np.array(
        [
            [params.fx, 0.0, params.cx],
            [0.0, params.fy, params.cy],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )


def list_connected_zed_cameras() -> list[dict[str, Any]]:
    devices = sl.Camera.get_device_list()
    ret: list[dict[str, Any]] = []
    for device in devices:
        item: dict[str, Any] = {
            "serial_number": int(device.serial_number),
        }
        for attr in ("camera_model", "id", "path", "input_type"):
            if hasattr(device, attr):
                value = getattr(device, attr)
                item[attr] = value if isinstance(value, (str, int, float, bool, type(None))) else str(value)
        ret.append(item)
    return ret


class ZedCameraNode(WorldNode[
    None,
    Dict[str, NumpyArrayType],
    None,
    NumpyArrayType,
    NumpyDeviceType,
    NumpyDtypeType,
    NumpyRNGType,
]):
    after_reset_priorities = {0}
    post_environment_step_priorities = {0}
    supported_render_modes = ("rgb_array",)

    def __init__(
        self,
        world: Optional[RealWorld] = None,
        name: str = "zed_camera",
        *,
        serial_number: Optional[int] = None,
        camera_id: Optional[int] = None,
        resolution: str | ZedResolution = ZedResolution.HD720,
        fps: int = 30,
        enable_left_rgb: bool = True,
        enable_right_rgb: bool = False,
        enable_left_depth: bool = False,
        enable_right_depth: bool = False,
        include_intrinsics: bool = True,
        depth_mode: Optional[str | ZedDepthMode] = None,
        flip_mode: str | ZedFlipMode = ZedFlipMode.OFF,
        unit: str | ZedUnit = ZedUnit.METER,
        enable_fill_mode: bool = False,
        confidence_threshold: Optional[int] = None,
        texture_confidence_threshold: Optional[int] = None,
        depth_minimum_distance: Optional[float] = None,
        depth_maximum_distance: Optional[float] = None,
        open_timeout_sec: Optional[float] = None,
        camera_disable_self_calib: bool = False,
        sdk_verbose: int = 0,
        control_timestep: Optional[float] = None,
        update_timestep: Optional[float] = None,
        invalid_depth_value: float = 0.0,
        raise_on_grab_failure: bool = True,
    ):
        if serial_number is not None and camera_id is not None:
            raise ValueError("Provide either serial_number or camera_id, not both.")
        if not (enable_left_rgb or enable_right_rgb or enable_left_depth or enable_right_depth):
            raise ValueError("At least one output must be enabled.")
        if isinstance(world, World):
            if world.backend != NumpyComputeBackend:
                raise TypeError("World backend must be NumpyComputeBackend.")
            if not world.is_control_timestep_compatible(control_timestep):
                raise ValueError("Control timestep must be a multiple of world timestep.")

        self.name = name
        self.world = world
        self.serial_number = serial_number
        self.camera_id = camera_id
        self.requested_resolution = _coerce_enum(resolution, ZedResolution, "resolution")
        self.requested_fps = int(fps)
        self.enable_left_rgb = bool(enable_left_rgb)
        self.enable_right_rgb = bool(enable_right_rgb)
        self.enable_left_depth = bool(enable_left_depth)
        self.enable_right_depth = bool(enable_right_depth)
        self.include_intrinsics = bool(include_intrinsics)
        self.flip_mode = _coerce_enum(flip_mode, ZedFlipMode, "flip_mode")
        self.unit = _coerce_enum(unit, ZedUnit, "unit")
        self.enable_fill_mode = bool(enable_fill_mode)
        self.confidence_threshold = confidence_threshold
        self.texture_confidence_threshold = texture_confidence_threshold
        self.depth_minimum_distance = depth_minimum_distance
        self.depth_maximum_distance = depth_maximum_distance
        self.open_timeout_sec = open_timeout_sec
        self.camera_disable_self_calib = bool(camera_disable_self_calib)
        self.sdk_verbose = int(sdk_verbose)
        self.control_timestep = control_timestep
        self.update_timestep = update_timestep
        self.invalid_depth_value = float(invalid_depth_value)
        self.raise_on_grab_failure = bool(raise_on_grab_failure)
        self._depth_enabled = self.enable_left_depth or self.enable_right_depth
        self.depth_mode = (
            _coerce_enum(depth_mode, ZedDepthMode, "depth_mode")
            if depth_mode is not None
            else (_default_supported_depth_mode() if self._depth_enabled else ZedDepthMode.NONE)
        )
        if self._depth_enabled and self.depth_mode == ZedDepthMode.NONE:
            raise ValueError("Depth output requested but depth_mode is NONE.")

        self.zed = sl.Camera()
        self._runtime_parameters = sl.RuntimeParameters()
        self._current_observation: Optional[Dict[str, NumpyArrayType]] = None
        self._render_key: Optional[str] = "left_rgb" if self.enable_left_rgb else ("right_rgb" if self.enable_right_rgb else None)

        self._left_rgb_mat = sl.Mat() if self.enable_left_rgb else None
        self._right_rgb_mat = sl.Mat() if self.enable_right_rgb else None
        self._left_depth_mat = sl.Mat() if self.enable_left_depth else None
        self._right_depth_mat = sl.Mat() if self.enable_right_depth else None

        try:
            self._open_camera()
            self._configure_runtime_parameters()
            self._refresh_camera_metadata()

            self.observation_space = self._build_observation_space()
            self.action_space = None
            self.render_mode = "rgb_array" if self._render_key is not None else None
        except Exception:
            self.close()
            raise

    @property
    def backend(self) -> ComputeBackend[NumpyArrayType, NumpyDeviceType, NumpyDtypeType, NumpyRNGType]:
        return NumpyComputeBackend

    @property
    def device(self) -> None:
        return None

    def _open_camera(self) -> None:
        init_parameters = sl.InitParameters()
        if self.serial_number is not None:
            init_parameters.set_from_serial_number(int(self.serial_number))
        elif self.camera_id is not None:
            if not hasattr(init_parameters, "set_from_camera_id"):
                raise ValueError("Installed ZED SDK does not support selecting cameras by camera_id.")
            try:
                init_parameters.set_from_camera_id(int(self.camera_id))
            except TypeError:
                if not hasattr(sl, "BUS_TYPE"):
                    raise
                init_parameters.set_from_camera_id(int(self.camera_id), sl.BUS_TYPE.AUTO)

        init_parameters.camera_resolution = _resolve_sl_enum(
            sl.RESOLUTION,
            self.requested_resolution,
            ZedResolution,
            "resolution",
        )
        init_parameters.camera_fps = self.requested_fps
        init_parameters.depth_mode = _resolve_sl_enum(
            sl.DEPTH_MODE,
            self.depth_mode,
            ZedDepthMode,
            "depth_mode",
        )
        init_parameters.camera_image_flip = _resolve_sl_enum(
            sl.FLIP_MODE,
            self.flip_mode,
            ZedFlipMode,
            "flip_mode",
        )
        init_parameters.coordinate_units = _resolve_sl_enum(
            sl.UNIT,
            self.unit,
            ZedUnit,
            "unit",
        )
        init_parameters.enable_right_side_measure = self.enable_right_depth
        init_parameters.camera_disable_self_calib = self.camera_disable_self_calib
        init_parameters.sdk_verbose = self.sdk_verbose

        if self.depth_minimum_distance is not None:
            init_parameters.depth_minimum_distance = float(self.depth_minimum_distance)
        if self.depth_maximum_distance is not None:
            init_parameters.depth_maximum_distance = float(self.depth_maximum_distance)
        if self.open_timeout_sec is not None and hasattr(init_parameters, "open_timeout_sec"):
            init_parameters.open_timeout_sec = float(self.open_timeout_sec)

        err = self.zed.open(init_parameters)
        if err != sl.ERROR_CODE.SUCCESS:
            self.zed.close()
            requested = f"serial_number={self.serial_number}" if self.serial_number is not None else (
                f"camera_id={self.camera_id}" if self.camera_id is not None else "default camera"
            )
            raise ConnectionError(f"Failed to open ZED camera ({requested}): {err}")

    def _configure_runtime_parameters(self) -> None:
        self._runtime_parameters.enable_depth = self._depth_enabled
        self._runtime_parameters.enable_fill_mode = self.enable_fill_mode
        if self.confidence_threshold is not None:
            self._runtime_parameters.confidence_threshold = int(self.confidence_threshold)
        if self.texture_confidence_threshold is not None:
            self._runtime_parameters.texture_confidence_threshold = int(self.texture_confidence_threshold)

    def _refresh_camera_metadata(self) -> None:
        self._camera_info = self.zed.get_camera_information()
        self._camera_config = self._camera_info.camera_configuration
        self._calibration = self._camera_config.calibration_parameters
        self.serial_number = int(self._camera_info.serial_number)
        self.camera_model = str(self._camera_info.camera_model)
        self.actual_fps = int(self._camera_config.fps)
        self.actual_width = int(self._camera_config.resolution.width)
        self.actual_height = int(self._camera_config.resolution.height)
        self.left_intrinsics = _zed_camera_intrinsics(self._calibration.left_cam)
        self.right_intrinsics = _zed_camera_intrinsics(self._calibration.right_cam)

    def _build_observation_space(self) -> DictSpace:
        spaces: dict[str, BoxSpace] = {}
        if self.enable_left_rgb:
            spaces["left_rgb"] = _box_rgb(self.actual_height, self.actual_width)
        if self.enable_right_rgb:
            spaces["right_rgb"] = _box_rgb(self.actual_height, self.actual_width)
        if self.enable_left_depth:
            spaces["left_depth"] = _box_depth(self.actual_height, self.actual_width)
        if self.enable_right_depth:
            spaces["right_depth"] = _box_depth(self.actual_height, self.actual_width)
        if self.include_intrinsics:
            spaces["left_intrinsics"] = _box_intrinsics()
            if self.enable_right_rgb or self.enable_right_depth:
                spaces["right_intrinsics"] = _box_intrinsics()
        return DictSpace(NumpyComputeBackend, spaces)

    def _grab(self) -> bool:
        err = self.zed.grab(self._runtime_parameters)
        if err == sl.ERROR_CODE.SUCCESS:
            return True
        if self.raise_on_grab_failure:
            raise RuntimeError(f"Failed to grab frame from ZED camera {self.serial_number}: {err}")
        return False

    def post_environment_step(self, dt: float, *, priority: int = 0) -> None:
        if not self._grab():
            return

        observation: Dict[str, NumpyArrayType] = {}
        if self.enable_left_rgb and self._left_rgb_mat is not None:
            self.zed.retrieve_image(self._left_rgb_mat, sl.VIEW.LEFT)
            observation["left_rgb"] = _zed_image_to_rgb(self._left_rgb_mat)
        if self.enable_right_rgb and self._right_rgb_mat is not None:
            self.zed.retrieve_image(self._right_rgb_mat, sl.VIEW.RIGHT)
            observation["right_rgb"] = _zed_image_to_rgb(self._right_rgb_mat)
        if self.enable_left_depth and self._left_depth_mat is not None:
            self.zed.retrieve_measure(self._left_depth_mat, sl.MEASURE.DEPTH)
            observation["left_depth"] = _zed_depth_to_array(self._left_depth_mat, self.invalid_depth_value)
        if self.enable_right_depth and self._right_depth_mat is not None:
            self.zed.retrieve_measure(self._right_depth_mat, sl.MEASURE.DEPTH_RIGHT)
            observation["right_depth"] = _zed_depth_to_array(self._right_depth_mat, self.invalid_depth_value)
        if self.include_intrinsics:
            observation["left_intrinsics"] = self.left_intrinsics.copy()
            if self.enable_right_rgb or self.enable_right_depth:
                observation["right_intrinsics"] = self.right_intrinsics.copy()

        self._current_observation = observation

    def after_reset(self, *, priority: int = 0, mask=None) -> None:
        self.post_environment_step(0.0, priority=priority)

    def get_observation(self) -> Dict[str, NumpyArrayType]:
        if self._current_observation is None:
            self.post_environment_step(0.0)
        if self._current_observation is None:
            raise RuntimeError("No ZED observation is available yet.")
        return self._current_observation

    def get_info(self) -> Dict[str, Any]:
        return {
            "serial_number": self.serial_number,
            "camera_model": self.camera_model,
            "resolution": (self.actual_width, self.actual_height),
            "fps": self.actual_fps,
            "depth_mode": self.depth_mode.value,
            "flip_mode": self.flip_mode.value,
            "unit": self.unit.value,
            "left_rgb_enabled": self.enable_left_rgb,
            "right_rgb_enabled": self.enable_right_rgb,
            "left_depth_enabled": self.enable_left_depth,
            "right_depth_enabled": self.enable_right_depth,
        }

    def render(self) -> Optional[np.ndarray]:
        if self._render_key is None:
            return None
        if self._current_observation is None:
            try:
                self.post_environment_step(0.0)
            except RuntimeError:
                return None
        obs = self._current_observation
        if obs is None:
            return None
        return obs.get(self._render_key)

    def close(self) -> None:
        if hasattr(self, "zed") and self.zed is not None and self.zed.is_opened():
            self.zed.close()


ZedCameraSensorNode = ZedCameraNode
