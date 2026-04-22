# zed_camera_adaptor

ZED camera node adaptor for UniEnv.

## Installation

Install the ZED SDK first so `pyzed.sl` is available, then install this package locally:

```bash
pip install -e .
```

## Usage

```python
from unienv_interface.backends.numpy import NumpyComputeBackend
from unienv_interface.world import RealWorld

from unienv_zed import ZedCameraNode, ZedResolution, ZedFlipMode

world = RealWorld(
    NumpyComputeBackend,
    world_timestep=1.0 / 30.0,
)

camera = ZedCameraNode(
    world=world,
    serial_number=None,            # or a specific device serial
    resolution=ZedResolution.HD720,
    fps=30,
    enable_left_rgb=True,
    enable_left_depth=True,
    enable_right_rgb=False,
    enable_right_depth=False,
    flip_mode=ZedFlipMode.OFF,
)

world.reset()
camera.after_reset()

while True:
    dt = world.step()
    camera.post_environment_step(dt)
    obs = camera.get_observation()
    rgb = obs["left_rgb"]
    depth = obs["left_depth"]
```

## Features

- Native `pyzed.sl` live-camera integration
- Configurable RGB/depth outputs for left and right sensors
- Resolution / FPS / depth mode / flip mode options
- Rectified intrinsics included in observations
- Real-world `WorldNode` lifecycle compatible with UniEnv
