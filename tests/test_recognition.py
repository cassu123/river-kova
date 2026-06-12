#!/usr/bin/env python3
"""
================================================================================
Project     : River Kova — Household Chore Robot Control System
File        : tests/test_recognition.py
Purpose     : Tests for learned room recognition: the classifier, the
              semantic map (including change handling and persistence),
              the point driver / explorer, and the explore initiative rule.
Author      : [Author Placeholder]
Version     : 1.0.0
Date        : 2026-06-12
License     : Proprietary — River Song AI (riversongai.com)
================================================================================
"""

from datetime import datetime

import pytest

from autonomy.initiative_engine import ExploreRule
from navigation.explorer import Explorer, PointDriver
from navigation.semantic_map import SemanticMap
from simulation.sim_world import SimWorld
from vision.room_classifier import RoomClassifier, UNKNOWN


# ─────────────────────────────────────────────────────────────────────────────
# ROOM CLASSIFIER
# ─────────────────────────────────────────────────────────────────────────────


class TestRoomClassifier:
    @pytest.fixture
    def classifier(self):
        return RoomClassifier()

    def test_kitchen_from_anchors(self, classifier):
        label, confidence = classifier.classify(["stove", "fridge", "cup"])
        assert label == "kitchen"
        assert confidence > 0.5

    def test_bathroom_from_anchors(self, classifier):
        label, _ = classifier.classify(["toilet", "bathtub", "sink"])
        assert label == "bathroom"

    def test_lone_sink_is_ambiguous(self, classifier):
        # A sink could be kitchen or bathroom — not enough evidence alone.
        label, confidence = classifier.classify(["sink"])
        assert label == UNKNOWN
        assert confidence == 0.0

    def test_empty_view_is_unknown(self, classifier):
        assert classifier.classify([]) == (UNKNOWN, 0.0)

    def test_movable_items_carry_no_evidence(self, classifier):
        assert classifier.classify(["cup", "sock", "bottle"])[0] == UNKNOWN

    def test_duplicates_do_not_inflate_evidence(self, classifier):
        single = classifier.classify(["couch", "tv"])
        repeated = classifier.classify(["couch", "couch", "tv", "tv", "tv"])
        assert single == repeated

    def test_runtime_signature_extension(self, classifier):
        classifier.add_signature("music_room", {"piano": 3.0})
        label, _ = classifier.classify(["piano"])
        assert label == "music_room"


# ─────────────────────────────────────────────────────────────────────────────
# SEMANTIC MAP
# ─────────────────────────────────────────────────────────────────────────────


class TestSemanticMap:
    @pytest.fixture
    def semantic_map(self):
        return SemanticMap(cell_size_m=1.0)

    def test_unvisited_is_unknown(self, semantic_map):
        assert semantic_map.room_at(3.0, 2.0) == (UNKNOWN, 0.0)

    def test_observation_labels_cell(self, semantic_map):
        semantic_map.observe(3.0, 2.0, ["stove", "fridge"])
        label, confidence = semantic_map.room_at(3.2, 2.8)   # same 1 m cell? (3,2)
        assert label == "kitchen"
        assert confidence > 0

    def test_known_rooms_centroid(self, semantic_map):
        semantic_map.observe(3.0, 2.0, ["stove", "fridge"])
        semantic_map.observe(4.0, 2.0, ["stove", "fridge"])
        rooms = semantic_map.known_rooms()
        assert "kitchen" in rooms
        assert rooms["kitchen"]["cells"] == 2
        assert 3.0 <= rooms["kitchen"]["x"] <= 4.5

    def test_locate_for_voice_targets(self, semantic_map):
        semantic_map.observe(7.0, 4.0, ["bed", "dresser"])
        location = semantic_map.locate("bedroom")
        assert location is not None
        assert semantic_map.locate("garage") is None

    def test_relabels_when_the_home_changes(self, semantic_map):
        # The "bedroom" becomes an office: old evidence decays away.
        semantic_map.observe(7.0, 4.0, ["bed", "dresser"])
        assert semantic_map.room_at(7.0, 4.0)[0] == "bedroom"
        classifier = semantic_map._classifier
        classifier.add_signature("office", {"desk": 3.0, "monitor": 2.0})
        for _ in range(10):
            semantic_map.observe(7.0, 4.0, ["desk", "monitor"])
        assert semantic_map.room_at(7.0, 4.0)[0] == "office"

    def test_visited_counts_unlabelled_cells(self, semantic_map):
        semantic_map.observe(0.5, 0.5, [])
        semantic_map.observe(5.5, 0.5, ["couch", "tv"])
        assert semantic_map.visited_cell_count == 2

    def test_save_and_load_roundtrip(self, semantic_map, tmp_path):
        semantic_map.observe(3.0, 2.0, ["stove", "fridge"])
        path = str(tmp_path / "semantic_map.json")
        semantic_map.save(path)

        restored = SemanticMap()
        assert restored.load(path)
        assert restored.room_at(3.0, 2.0)[0] == "kitchen"
        assert restored.visited_cell_count == 1

    def test_load_missing_file_is_false(self, semantic_map, tmp_path):
        assert not semantic_map.load(str(tmp_path / "nope.json"))

    def test_snapshot_shape(self, semantic_map):
        semantic_map.observe(3.0, 2.0, ["stove", "fridge"])
        snap = semantic_map.snapshot()
        assert {"cell_size_m", "visited_cells", "rooms"} <= set(snap)
        assert "kitchen" in snap["rooms"]


# ─────────────────────────────────────────────────────────────────────────────
# POINT DRIVER & EXPLORER (against the simulated body)
# ─────────────────────────────────────────────────────────────────────────────


class _DirectBridge:
    """Minimal IOBridge stand-in driving a SimWorld directly."""

    def __init__(self, world):
        self._world = world

    def set_motor_speeds(self, left, right):
        self._world.set_motors(left, right)


class TestPointDriver:
    @pytest.fixture
    def world(self):
        return SimWorld()

    @pytest.fixture
    def driver(self, world):
        return PointDriver(
            bridge=_DirectBridge(world),
            pose_provider=lambda: world.pose,
        )

    @pytest.mark.timeout(30)
    def test_reaches_nearby_target(self, world, driver):
        assert driver.go_to(1.5, 0.0, timeout_sec=15.0)
        x, y, _ = world.pose
        assert abs(x - 1.5) <= driver.tolerance_m + 0.05
        assert abs(y) <= driver.tolerance_m + 0.05

    @pytest.mark.timeout(10)
    def test_estopped_body_fails_safely(self, world, driver):
        world.estop_motors()
        assert not driver.go_to(2.0, 0.0, timeout_sec=2.0)

    @pytest.mark.timeout(10)
    def test_safety_check_stops_motion(self, world):
        driver = PointDriver(
            bridge=_DirectBridge(world),
            pose_provider=lambda: world.pose,
            safety_check=lambda: False,
        )
        assert not driver.go_to(2.0, 0.0, timeout_sec=5.0)


class TestExplorer:
    def test_sweep_points_cover_bounds_serpentine(self):
        explorer = Explorer(
            driver=None,
            bounds_provider=lambda: (0.0, 0.0, 4.0, 3.0),
            grid_step_m=1.5,
        )
        points = explorer.sweep_points()
        assert len(points) == 6          # 2 rows × 3 columns
        # Serpentine: second row runs right-to-left
        assert points[3][0] > points[4][0]

    def test_point_filter_skips_off_floorplan(self):
        explorer = Explorer(
            driver=None,
            bounds_provider=lambda: (0.0, 0.0, 4.0, 3.0),
            point_filter=lambda x, y: x < 2.0,
            grid_step_m=1.5,
        )
        assert all(x < 2.0 for x, _ in explorer.sweep_points())

    @pytest.mark.timeout(30)
    def test_explore_drives_and_reports_success(self):
        world = SimWorld()
        driver = PointDriver(
            bridge=_DirectBridge(world),
            pose_provider=lambda: world.pose,
        )
        explorer = Explorer(
            driver=driver,
            bounds_provider=lambda: (0.0, 0.0, 2.0, 2.0),
            grid_step_m=1.5,
        )
        assert explorer.explore()

    def test_empty_bounds_fail(self):
        explorer = Explorer(
            driver=None,
            bounds_provider=lambda: (0.0, 0.0, 0.5, 0.5),
            grid_step_m=1.5,
        )
        assert not explorer.explore()


# ─────────────────────────────────────────────────────────────────────────────
# EXPLORE INITIATIVE RULE
# ─────────────────────────────────────────────────────────────────────────────


class TestExploreRule:
    def test_proposes_while_home_unfamiliar(self):
        rule = ExploreRule(known_room_count_provider=lambda: 1, min_known_rooms=4)
        proposals = rule.evaluate(datetime(2026, 6, 12, 10, 0))
        assert len(proposals) == 1
        assert proposals[0].chore_type == "EXPLORE"
        assert proposals[0].priority < 3      # Below tidy-ups and commands

    def test_silent_once_rooms_are_known(self):
        rule = ExploreRule(known_room_count_provider=lambda: 5, min_known_rooms=4)
        assert rule.evaluate(datetime(2026, 6, 12, 10, 0)) == []

    def test_provider_error_is_swallowed(self):
        def boom():
            raise RuntimeError("no map")
        rule = ExploreRule(known_room_count_provider=boom)
        assert rule.evaluate(datetime(2026, 6, 12, 10, 0)) == []
