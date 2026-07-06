from setuptools import setup, find_packages

setup(
    name    = "cdpo-financial",
    version = "0.1.0",
    description = (
        "Constraint-Decomposed Policy Optimisation (CDPO) "
        "for Financial Planning Agents — NeurIPS 2026"
    ),
    packages = find_packages(),
    python_requires = ">=3.10",
    install_requires = [
        "numpy>=1.24",
        "scipy>=1.10",
    ],
    extras_require = {
        "train": [
            "torch>=2.1",
            "transformers>=4.46.0,<4.48",
            "datasets>=2.21.0",
            "accelerate>=0.34.0",
            "trl==0.14.0",  # GRPOTrainer first appears in 0.14.0; compute_loss override targets this,
            "wandb>=0.16",
            "peft>=0.10",
        ],
        "dev": [
            "pytest>=7.0",
        ],
        # vLLM is optional and large; install separately on the GPU pod with
        # `pip install vllm` (or `pip install -e ".[vllm]"`). Pinned loosely
        # because the right wheel depends on the pod's CUDA version.
        "vllm": [
            "vllm>=0.6.3",
        ],
    },
)
