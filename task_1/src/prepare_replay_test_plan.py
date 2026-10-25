"""Зафиксировать план оценки семи повторно обученных моделей без чтения test."""
from pathlib import Path
import argparse
import copy
import json

import numpy as np

from final_test_data import ROOT, sha256, write_json
from replay_paths import project_path, saved_path, run_name

TEMPLATE = ROOT / "results/test_plans/final_test_20261023_v1/plan.json"


def record(path):
    path = saved_path(path)
    return {"path": str(path), "sha256": sha256(path), "bytes": path.stat().st_size}


def read_json(path):
    return json.loads(Path(path).read_text())


def build_plan(name, controls_run, full_run, baseline_dir, base_borders):
    """Заменить источники шаблона на новые модели, сохранив метрики и пары."""
    run_name(name)
    controls = ROOT / "results/experiment-1/temporal_controls" / run_name(controls_run)
    full = ROOT / "results/experiment-1/temporal_runs" / run_name(full_run)
    baseline = project_path(baseline_dir)
    for folder in (controls, full):
        if not read_json(folder / "result.json")["completed"]:
            raise ValueError("Исходный запуск не завершён: " + str(folder))
    control_protocol = read_json(controls / "protocol.json")
    full_protocol = read_json(full / "protocol.json")
    selection = read_json(controls / "selection.json")
    plan = copy.deepcopy(read_json(TEMPLATE))
    plan.update(plan_id=name, status="frozen_not_executed",
                selection=selection, c4_deferred=False,
                upstream_data_provenance_from_full_run=full_protocol["inputs"])
    plan.pop("frozen_at_utc", None)
    plan["schemas"] = {
        "manual42": control_protocol["manual_features"],
        "manual49": control_protocol["manual_features"] + control_protocol["session_features"],
        "dfs462": full_protocol["features"],
    }
    quantization = read_json(full / "quantization_train.json")
    plan["train_borders"] = {
        "manual42": record(project_path(base_borders)),
        "manual49": record(ROOT / "data/experiment-1/temporal_controls" / controls_run /
                           "manual_quantized/train_borders.tsv"),
        "dfs462": record(saved_path(quantization["borders_path"])),
    }
    with np.load(saved_path(full_protocol["inputs"]["train_targets"]["path"])) as targets:
        full_rows = len(targets["label"])
        full_groups = len(np.unique(targets["group_id"]))
    for model in plan["models"]:
        identifier = model["id"]
        if model["family"] == "sub":
            folder = controls / "models" / identifier
            control = read_json(folder / "control.json")
            model.update(train_rows=selection["train"]["rows"],
                         train_groups=selection["train"]["groups"],
                         active_features=control["active_features"],
                         active_count=control["active_feature_count"],
                         ignored_indices=control["ignored_indices"])
            provenance = folder / "control.json"
        else:
            folder = baseline if identifier == "M-full" else full / "model"
            names = plan["schemas"][model["schema"]]
            model.update(train_rows=full_rows, train_groups=full_groups,
                         active_features=names, active_count=len(names), ignored_indices=[])
            provenance = folder / "evaluation.json"
        evaluation = read_json(folder / "evaluation.json")
        model.update(model=record(folder / "model.cbm"), provenance=record(provenance),
                     tree_count=evaluation["tree_count"], parameters=evaluation["params"])
    sources = sorted((ROOT / "src").glob("*.py"))
    plan["source_files"] = {path.name: record(path) for path in sources}
    reports = [controls / "protocol.json", controls / "selection.json",
               full / "protocol.json", full / "quantization_train.json",
               baseline / "evaluation.json"]
    plan["provenance"] = {str(path.relative_to(ROOT)): record(path) for path in reports}
    plan["execution"].update(output=str(ROOT / "results/final_test" / name),
                             adapter_implemented=True)
    return plan


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--controls-run", required=True)
    parser.add_argument("--full-run", required=True)
    parser.add_argument("--baseline-dir", required=True)
    parser.add_argument("--base-borders", required=True)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    plan = build_plan(args.run_name, args.controls_run, args.full_run,
                      args.baseline_dir, args.base_borders)
    if args.check_only:
        print(json.dumps({"valid": True, "models": len(plan["models"]), "test_opened": False}))
        return
    folder = ROOT / "results/test_plans" / run_name(args.run_name)
    if folder.exists():
        raise FileExistsError("Нужно новое имя тестового плана")
    folder.mkdir(parents=True)
    path = folder / "plan.json"
    write_json(path, plan)
    path.with_name("plan.sha256").write_text(sha256(path) + "  plan.json\\n")
    print(json.dumps({"plan": str(path), "test_opened": False}))


if __name__ == "__main__":
    main()
