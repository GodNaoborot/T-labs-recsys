"""Проверки новых входов повторного расчёта без моделей и большого датасета."""
from pathlib import Path
import json
import os
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import replay_paths
import temporal_sample_data
import run_temporal_full
import compare_temporal_controls_pairs as pairs
import prepare_replay_test_plan as plans


class ReplayInputTests(unittest.TestCase):
    def test_namespace_and_saved_paths_use_current_checkout(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            with patch.object(replay_paths, "TASK_ROOT", root), patch.dict(os.environ, {}, clear=True):
                self.assertEqual(replay_paths.legacy_featuretools_dir(),
                                 root / "data/experiment-1/featuretools")
                replay_paths.configure_legacy("repeat_01")
                self.assertEqual(replay_paths.legacy_featuretools_dir(),
                                 root / "data/experiment-1/runs/repeat_01/featuretools")
                old = "/home/old/projects/repository/task_1/results/run/model.cbm"
                self.assertEqual(replay_paths.saved_path(old), root / "results/run/model.cbm")
                with self.assertRaises(ValueError):
                    replay_paths.configure_legacy("../old")

    def test_sample_root_override_stays_inside_synthetic_workspace(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            with patch.object(temporal_sample_data, "ROOT", root), patch.dict(os.environ, {}, clear=True):
                for split in ("train", "val"):
                    targets, metadata = temporal_sample_data._source_paths(split)
                    self.assertTrue(targets.resolve().is_relative_to(root.resolve()))
                    self.assertTrue(metadata.resolve().is_relative_to(root.resolve()))

    def test_full_sources_are_forwarded_to_workers_environment(self):
        args = run_temporal_full.parser().parse_args([
            "--run-name", "full_new", "--legacy-run", "history_new",
            "--selection-run", "used_new", "--definitions-run", "wide_new",
            "--baseline-dir", "results/tuning/tune_new/queue/control_manual_ranker/attempt_0001"])
        old = {key: getattr(run_temporal_full, key) for key in ("SELECTION", "BASELINE", "DEFINITIONS")}
        try:
            with patch.dict(os.environ, {}, clear=True):
                run_temporal_full.configure_sources(args)
                self.assertEqual(os.environ["TLAB_EXPERIMENT_RUN"], "history_new")
                self.assertEqual(os.environ["TLAB_TEMPORAL_SELECTION_RUN"], "used_new")
                self.assertTrue(str(run_temporal_full.SELECTION).endswith("/used_new/protocol.json"))
                child = run_temporal_full.parser().parse_args(["--run-name", "full_new"])
                self.assertEqual(child.selection_run, "used_new")
                self.assertEqual(child.definitions_run, "wide_new")
        finally:
            for key, value in old.items():
                setattr(run_temporal_full, key, value)

    def test_direct_intervals_work_from_public_daily_sums(self):
        with tempfile.TemporaryDirectory() as folder:
            run = Path(folder)
            for name in ("status", "result"):
                (run / f"{name}.json").write_text(json.dumps({"completed": True}))
            (run / "comparison.csv").write_text("variant,NDCG@10\\n")
            for name, value in (("C2b", .84), ("D462-sub", .80), ("C2a", .82)):
                pd.DataFrame({"day": np.arange(270, 285), "group_count": 10,
                              "base_sum": 7.7, "dfs_sum": value * 10}).to_csv(
                                  run / f"daily_{name}.csv", index=False)
            result = pairs.calculate(run, run / "direct_new")
            self.assertEqual(result["input_level"], "daily_sums")
            self.assertEqual(result["groups"], 150)
            deltas = [row["delta_ndcg"] for row in result["pairs"]]
            np.testing.assert_allclose(deltas, [.04, .04, .02, .02])

    def test_fresh_test_plan_freezes_new_models_without_reading_test(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            controls = root / "results/experiment-1/temporal_controls/controls_new"
            full = root / "results/experiment-1/temporal_runs/full_new"
            baseline = root / "results/tuning/tune_new/control"
            def write(path, value):
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps(value))
            names = ["M-sub", "D462-sub", "C1", "C2a", "C2b", "M-full", "D462-full"]
            models = [{"id": name, "family": "sub" if i < 5 else "full",
                       "schema": "manual49" if name in ("M-sub", "C2a", "C2b")
                       else "manual42" if name == "M-full" else "dfs462"}
                      for i, name in enumerate(names)]
            template = root / "template.json"
            write(template, {"models": models, "comparisons": [{} for _ in names],
                             "execution": {}, "new_training_allowed": False,
                             "full_C2b_included": False})
            for run in (controls, full):
                write(run / "result.json", {"completed": True})
            write(controls / "protocol.json", {"manual_features": ["manual"],
                                               "session_features": ["session"]})
            write(controls / "selection.json", {"train": {"rows": 2, "groups": 1},
                                                "val": {"rows": 2, "groups": 1}})
            targets = root / "train_targets.npz"
            np.savez(targets, label=[0, 1], group_id=[1, 1])
            write(full / "protocol.json", {"features": ["dfs"],
                  "inputs": {"train_targets": {"path": str(targets)}}})
            borders = root / "borders.tsv"
            borders.write_text("0\\t0.5\\n")
            write(full / "quantization_train.json", {"borders_path": str(borders)})
            manual49 = root / "data/experiment-1/temporal_controls/controls_new/manual_quantized/train_borders.tsv"
            manual49.parent.mkdir(parents=True)
            manual49.write_bytes(borders.read_bytes())
            for name in names:
                target = controls / "models" / name if name.endswith("-sub") or name in ("C1", "C2a", "C2b") else baseline if name == "M-full" else full / "model"
                write(target / "evaluation.json", {"tree_count": 3, "params": {"depth": 2}})
                (target / "model.cbm").write_bytes(name.encode())
                if name in names[:5]:
                    write(target / "control.json", {"active_features": ["dfs"] if name in ("C1", "D462-sub") else ["manual"],
                          "active_feature_count": 1, "ignored_indices": []})
            (root / "src").mkdir()
            (root / "src/source.py").write_text("pass\\n")
            with patch.object(plans, "ROOT", root), patch.object(plans, "TEMPLATE", template):
                plan = plans.build_plan("test_new", "controls_new", "full_new", str(baseline), str(borders))
            self.assertEqual(len(plan["models"]), 7)
            self.assertEqual(plan["models"][0]["train_rows"], 2)
            self.assertEqual(plan["models"][-1]["train_groups"], 1)
            self.assertTrue(all(str(root) in row["model"]["path"] for row in plan["models"]))
            self.assertFalse((root / "data/listens.parquet").exists())
            self.assertEqual(json.loads(template.read_text())["models"], models)


if __name__ == "__main__":
    unittest.main()
