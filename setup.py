#!/usr/bin/env python3

import re
from pathlib import Path

from setuptools import find_packages, setup


ROOT = Path(__file__).resolve().parent


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def read_version() -> str:
    init_py = read_text(ROOT / "sam_3d_body" / "__init__.py")
    match = re.search(r'^__version__\s*=\s*"([^"]+)"', init_py, re.MULTILINE)
    if not match:
        raise RuntimeError("Unable to find __version__ in sam_3d_body/__init__.py")
    return match.group(1)


setup(
    name="sam-3d-body",
    version=read_version(),
    description="SAM 3D Body: robust full-body human mesh recovery",
    long_description=read_text(ROOT / "README.md"),
    long_description_content_type="text/markdown",
    author="Meta Platforms, Inc.",
    license="SAM License",
    url="https://github.com/facebookresearch/sam-3d-body",
    packages=find_packages(include=["sam_3d_body", "sam_3d_body.*"]),
    include_package_data=True,
    python_requires=">=3.10",
    install_requires=[
        "braceexpand",
        "einops",
        "huggingface_hub",
        "numpy",
        "omegaconf",
        "opencv-python",
        "Pillow",
        "pyrender",
        "pytorch-lightning",
        "roma",
        "timm",
        "torch",
        "torchvision",
        "trimesh",
        "yacs",
    ],
    extras_require={
        "detector": [
            "detectron2 @ git+https://github.com/facebookresearch/detectron2.git@a1ce2f9"
        ],
        "notebook": ["matplotlib"],
        "dev": ["black", "pytest"],
    },
    classifiers=[
        "License :: Other/Proprietary License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
)
