# Vendored source snapshots

SIMPLE and the dependencies listed below are tracked as ordinary files in the
main repository. Their original license and attribution files are retained.

| Directory | Source commit | Source repository |
| --- | --- | --- |
| `SIMPLE` | `c7b77a2a625402aafe588f4af33b0537ea676b46` | https://github.com/Yunheng-Wang/SIMPLE.git |
| `third_party/XRoboToolkit-PC-Service-Pybind_X86_and_ARM64` | `2dbbc113171f9ed7a9801a8c78fae8556e9565df` | https://github.com/songlin/XRoboToolkit-PC-Service-Pybind_X86_and_ARM64.git |
| `third_party/curobo` | `5810bc9cb69129a0e1a8aad47704bd6da35370c0` | https://github.com/songlin/curobo.git |
| `third_party/decoupled_wbc` | `5b304e4d643fe89cd000068dcbba7e7519539caa` | https://github.com/songlin/decoupled_wbc.git |
| `third_party/gear_sonic` | `5b71b42ff6849cd121d93c58a7b508af8777c6c3` | https://github.com/songlin/gear_sonic.git |
| `third_party/openpi-client` | `4b8a1dd6fd93a9e90d91658361d4d7f98098a44a` | https://github.com/songlin/openpi-client.git |
| `third_party/unitree_sdk2_python` | `cf26ccef0e615c9509140322bec154adc391c052` | https://github.com/songlin/unitree_sdk2_python.git |

`third_party/AMO` and `third_party/gsnet` had no source files in the local snapshot
and are not included. MP evaluation loads its AMO controller and weights from
`src/simple/robots/policy/`.

Model weights, meshes, images, and bundled binary distributions use Git LFS.
Install Git LFS before cloning; for an existing checkout, run `git lfs pull`
from the main repository to download the actual files.

Local virtual environments, package caches, and generated artifacts remain
ignored. External evaluation datasets and model checkpoints are configured as
described in `../evaluation/simple_eval.md`.

The cuRobo package has an explicit setuptools-scm fallback version matching this
snapshot so it can be packaged without its original Git metadata.
