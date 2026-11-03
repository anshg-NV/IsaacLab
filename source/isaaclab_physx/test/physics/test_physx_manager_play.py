# Copyright (c) 2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""CPU-only control-flow regression for the actual PhysxManager.play method."""

import ast
import unittest
from pathlib import Path
from types import SimpleNamespace


class TestPhysxManagerPlay(unittest.TestCase):
    def call_play(self, stopped: bool) -> list[str]:
        """Load the source method without importing a simulator or copying its logic."""
        source = Path(__file__).resolve().parents[2] / "isaaclab_physx/physics/physx_manager.py"
        module = ast.parse(source.read_text(encoding="utf-8"))
        manager = next(node for node in module.body if isinstance(node, ast.ClassDef) and node.name == "PhysxManager")
        method = next(node for node in manager.body if isinstance(node, ast.FunctionDef) and node.name == "play")
        calls = []
        app = SimpleNamespace(update=lambda: calls.append("update"))
        namespace = {"omni": SimpleNamespace(kit=SimpleNamespace(app=SimpleNamespace(get_app=lambda: app)))}
        exec(compile(ast.Module(body=[method], type_ignores=[]), str(source), "exec"), namespace)
        stub = SimpleNamespace(
            _timeline=SimpleNamespace(
                is_stopped=lambda: stopped, is_playing=lambda: False, play=lambda: calls.append("play")
            ),
            _sync_fabric_after_resume=lambda: calls.append("sync_fabric"),
        )
        namespace["play"](stub)
        return calls

    def test_initial_play_does_not_reattach_fabric(self):
        self.assertEqual(self.call_play(stopped=True), ["play", "update", "update"])

    def test_resume_keeps_fabric_workaround_after_timeline_update(self):
        self.assertEqual(self.call_play(stopped=False), ["play", "update", "sync_fabric"])


if __name__ == "__main__":
    unittest.main()
