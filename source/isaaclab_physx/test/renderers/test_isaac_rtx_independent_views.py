# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Regression coverage for per-environment camera render products."""

from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import warp as wp
from isaaclab_physx.renderers.isaac_rtx_renderer import IsaacRtxRenderData, IsaacRtxRenderer
from isaaclab_physx.renderers.isaac_rtx_renderer_cfg import IsaacRtxRendererCfg

from isaaclab.renderers.camera_render_spec import CameraRenderSpec
from isaaclab.utils.warp import ProxyArray


def test_tiled_renderer_remains_the_default():
    assert IsaacRtxRendererCfg().tiled_rendering


def test_independent_render_products_preserve_camera_order_and_single_view_shape():
    class Renderer(IsaacRtxRenderer):
        def create_render_data(self, spec):
            if spec.view_count == 1:
                return IsaacRtxRenderData(annotators={}, spec=spec, render_product=Mock(path=spec.camera_prim_paths[0]))
            return super().create_render_data(spec)

    renderer = Renderer.__new__(Renderer)
    renderer.cfg = IsaacRtxRendererCfg(tiled_rendering=False)
    paths = tuple(f"/World/envs/env_{i}/Cam" for i in range(5))
    spec = CameraRenderSpec(cfg=SimpleNamespace(), device="cpu", num_instances=5, view_count=5, camera_prim_paths=paths)
    with patch.object(IsaacRtxRenderData, "__post_init__"):
        data = renderer.create_render_data(spec)
    assert data.spec is spec
    assert [view.render_product.path for view in data.independent_views] == list(paths)
    assert len(data.independent_views) == 5
    for index, view in enumerate(data.independent_views):
        assert view.spec.num_instances == view.spec.view_count == 1
        assert view.spec.camera_prim_paths == (paths[index],)
    assert spec.view_count == 5


def test_independent_outputs_write_only_their_environment_without_copying():
    renderer = IsaacRtxRenderer.__new__(IsaacRtxRenderer)
    children = [IsaacRtxRenderData(annotators={}, render_product=None) for _ in range(3)]
    data = IsaacRtxRenderData(annotators={}, render_product=None, independent_views=children)
    pixels = ProxyArray(wp.zeros((3, 8, 12, 4), dtype=wp.uint8, device="cpu"))
    renderer.set_outputs(data, {"rgba": pixels})
    children[1].output_data["rgba"].torch.fill_(77)
    expected = np.zeros((3, 8, 12, 4), dtype=np.uint8)
    expected[1] = 77
    np.testing.assert_array_equal(pixels.warp.numpy(), expected)
    assert children[1].output_data["rgba"].shape == (1, 8, 12, 4)
    children[2].output_data["rgba"].torch.fill_(99)
    expected[2] = 99
    np.testing.assert_array_equal(pixels.warp.numpy(), expected)


def test_independent_cleanup_detaches_every_camera_annotator():
    renderer = IsaacRtxRenderer.__new__(IsaacRtxRenderer)
    annotators = [Mock(), Mock()]
    children = [
        IsaacRtxRenderData(annotators={"rgba": annotator}, render_product=Mock(path=f"/Render/{i}"))
        for i, annotator in enumerate(annotators)
    ]
    data = IsaacRtxRenderData(annotators={}, render_product=None, independent_views=children)
    products = [view.render_product for view in children]
    renderer.cleanup(data)
    for index, annotator in enumerate(annotators):
        annotator.detach.assert_called_once_with([f"/Render/{index}"])
        products[index].destroy.assert_called_once()
