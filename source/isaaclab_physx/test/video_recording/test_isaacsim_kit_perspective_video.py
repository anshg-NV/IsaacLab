# Copyright (c) 2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""CPU tests for recording an existing camera without changing its authored pose."""

import sys
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from isaaclab_physx.video_recording.isaacsim_kit_perspective_video import IsaacsimKitPerspectiveVideo


@pytest.mark.parametrize("authored", [False, True])
def test_render_camera_pose_and_product(authored):
    """Authored camera capture preserves roll/pose; default viewport positioning still runs."""
    cfg = SimpleNamespace(
        camera_prim_path="/World/envs/env_2/hubble_lab_visuals/SceneCamera" if authored else "/OmniverseKit_Persp",
        use_authored_camera_pose=authored,
        eye=(1.0, 2.0, 3.0),
        lookat=(0.0, 0.0, 0.0),
        window_width=8,
        window_height=6,
    )
    omni = MagicMock()
    rep = omni.replicator.core
    rendering = MagicMock()
    rgba = np.full((6, 8, 4), 127, dtype=np.uint8)
    rep.AnnotatorRegistry.get_annotator.return_value.get_data.side_effect = [np.array([]), {"data": rgba}]
    with patch.dict(
        sys.modules,
        {
            "omni": omni,
            "omni.kit": omni.kit,
            "omni.kit.app": omni.kit.app,
            "omni.replicator": omni.replicator,
            "omni.replicator.core": rep,
            "isaaclab_physx.renderers.kit_viewport_utils": rendering,
        },
    ):
        capture = IsaacsimKitPerspectiveVideo(cfg)
        np.testing.assert_array_equal(capture.render_rgb_array(), np.zeros((6, 8, 3), dtype=np.uint8))
        np.testing.assert_array_equal(capture.render_rgb_array(), rgba[:, :, :3])
        capture.close()
    rep.create.render_product.return_value.destroy.assert_called_once()
    rep.create.render_product.assert_called_once_with(cfg.camera_prim_path, (8, 6))
    if authored:
        rendering.set_kit_renderer_camera_view.assert_not_called()
    else:
        rendering.set_kit_renderer_camera_view.assert_called_once_with(
            eye=list(cfg.eye), target=list(cfg.lookat), camera_prim_path=cfg.camera_prim_path
        )
