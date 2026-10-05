# Huyawo Adapt Environment Preflight

Status: PASS
Date: 2026-10-05

## Project isolation

- Project root: /workspace/huyawo-adapt
- Virtual environment: /workspace/huyawo-adapt/.venv
- Python: 3.12.3
- sys.executable: /workspace/huyawo-adapt/.venv/bin/python
- sys.prefix: /workspace/huyawo-adapt/.venv
- sys.base_prefix: /usr
- VIRTUAL_ENV: /workspace/huyawo-adapt/.venv
- PYTHONPATH: unset
- Sibling workspace paths in sys.path: none
- Jupyter kernel: Huyawo Adapt
- Kernel interpreter: /workspace/huyawo-adapt/.venv/bin/python

## Host session evidence

- OS: Ubuntu 24.04.3 LTS
- Architecture: x86_64
- Kernel: 6.8.0-134-generic
- glibc: 2.39
- GPU: NVIDIA RTX A5000
- GPU memory: 24564 MiB
- NVIDIA driver: 580.159.04
- Compute capability: 8.6
- CUDA toolkit: 12.8
- nvcc: V12.8.93

The observed GPU model is session evidence only.
Huyawo Adapt must remain capability-driven and must not depend on this GPU model.

## Foundation runtime

- torch: 2.11.0+cu128
- torch CUDA build: 12.8
- triton: 3.6.0
- numpy: 2.5.3
- transformers: 5.18.0
- accelerate: 1.15.0
- peft: 0.21.2
- trl: 1.14.1
- datasets: 5.1.0
- bitsandbytes: 0.50.2
- ipykernel: 7.4.0

## Validation results

- CUDA available: PASS
- BF16 support: PASS
- FP16 CUDA matrix multiplication: PASS
- BF16 CUDA matrix multiplication: PASS
- Torch to NumPy conversion: PASS
- bitsandbytes Linear8bitLt on CUDA: PASS
- Foundation package imports: PASS
- pip check: PASS
- Virtual environment prefix check: PASS
- Jupyter kernelspec interpreter check: PASS
- Sibling workspace isolation check: PASS

## Reproduction commands

    cd /workspace/huyawo-adapt
    python3 -m venv .venv
    source .venv/bin/activate
    python -m pip install --upgrade pip setuptools wheel
    python -m pip install -r environment/requirements.lock.txt
    python -m ipykernel install --prefix "${VIRTUAL_ENV}" --name "huyawo-adapt" --display-name "Huyawo Adapt"
    python -m pip check

The complete resolved Python environment is preserved in pip-freeze.txt and pip-inspect.json.
