"""Export the actor of an rsl-rl checkpoint (Phase 2 / Unified) to ONNX, as the robot would run it.

The actor is a plain MLP without observation normalizer (``empirical_normalization`` off); the deterministic action is
its output (the mean of the Gaussian). Needs only torch and onnx, no Isaac Sim (run in the Isaac Lab environment).
Next to ``<out>.onnx`` it writes ``<out>_check.npz`` with random inputs and the torch outputs, which ``sim2sim/run.py``
compares with onnxruntime before it starts.

    python sim2sim/tools/export_onnx.py /abs/path/model_3000.pt --out policy
"""

import argparse

import numpy as np
import torch


def actor_from_checkpoint(path: str) -> torch.nn.Sequential:
    state = torch.load(path, map_location="cpu", weights_only=True)["actor_state_dict"]
    linear = sorted({int(k.split(".")[1]) for k in state if k.startswith("mlp.") and k.endswith(".weight")})
    layers = []
    for n, i in enumerate(linear):
        weight = state[f"mlp.{i}.weight"]
        layer = torch.nn.Linear(weight.shape[1], weight.shape[0])
        layer.weight.data.copy_(weight)
        layer.bias.data.copy_(state[f"mlp.{i}.bias"])
        layers.append(layer)
        if n < len(linear) - 1:
            layers.append(torch.nn.ELU())
    unknown = [k for k in state if not k.startswith("mlp.") and k != "distribution.std_param"]
    if unknown:
        raise ValueError(f"Unexpected actor parameters (normalizer?): {unknown}")
    return torch.nn.Sequential(*layers).eval()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("checkpoint")
    parser.add_argument("--out", type=str, required=True, help="Output path without extension.")
    args = parser.parse_args()
    actor = actor_from_checkpoint(args.checkpoint)
    obs_dim = actor[0].in_features
    torch.onnx.export(
        actor, torch.zeros(1, obs_dim), f"{args.out}.onnx", input_names=["obs"], output_names=["actions"],
        opset_version=17,
    )
    inputs = torch.randn(16, obs_dim)
    with torch.no_grad():
        outputs = actor(inputs)
    np.savez(f"{args.out}_check.npz", obs=inputs.numpy(), actions=outputs.numpy(), checkpoint=args.checkpoint)
    print(f"[export] {args.checkpoint}: {obs_dim} -> {actor[-1].out_features} -> {args.out}.onnx")


if __name__ == "__main__":
    main()
