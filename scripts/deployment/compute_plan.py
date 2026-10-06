"""Report where Core ML would place each op of a package (CPU / GPU / Neural Engine).

Uses MLComputePlan on this Mac. Placement on iPhone can differ; confirm with a device run.
"""
import argparse
import json
import shutil
import tempfile
from collections import Counter
from pathlib import Path


def plan(package, units):
    import coremltools as ct
    from coremltools.models.compute_plan import MLComputePlan
    from coremltools.models.compute_device import MLCPUComputeDevice, MLGPUComputeDevice, MLNeuralEngineComputeDevice

    names = {MLCPUComputeDevice: "cpu", MLGPUComputeDevice: "gpu", MLNeuralEngineComputeDevice: "ane"}
    temporary = Path(tempfile.mkdtemp())
    try:
        compiled = ct.utils.compile_model(str(package), str(temporary / "model.mlmodelc"))
        compute_plan = MLComputePlan.load_from_path(path=str(compiled), compute_units=getattr(ct.ComputeUnit, units))
        program = compute_plan.model_structure.program
        placement, cost, details = Counter(), Counter(), []
        for function in program.functions.values():
            for operation in function.block.operations:
                usage = compute_plan.get_compute_device_usage_for_mlprogram_operation(operation)
                if usage is None:
                    continue
                device = names.get(type(usage.preferred_compute_device), "other")
                estimate = compute_plan.get_estimated_cost_for_mlprogram_operation(operation)
                weight = estimate.weight if estimate else 0.0
                placement[device] += 1
                cost[device] += weight
                details.append({"op": operation.operator_name, "device": device, "cost": weight,
                                "supported": sorted({names.get(type(d), "other")
                                                     for d in usage.supported_compute_devices})})
        return {"package": str(package), "compute_units": units, "ops_by_device": dict(placement),
                "estimated_cost_by_device": {k: round(v, 4) for k, v in cost.items()}, "ops": details}
    finally:
        shutil.rmtree(temporary, ignore_errors=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True, help="Experiment directory with model.mlpackage")
    parser.add_argument("--compute-units", default="CPU_AND_NE", choices=("CPU_ONLY", "CPU_AND_GPU", "CPU_AND_NE", "ALL"))
    parser.add_argument("--details", action="store_true", help="Print every op")
    args = parser.parse_args()
    report = plan(args.model / "model.mlpackage", args.compute_units)
    if not args.details:
        report.pop("ops")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
