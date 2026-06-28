"""Editable install setup for ai-3d-studio."""
from setuptools import find_packages, setup

setup(
    name="ai3d",
    version="0.1.0",
    description="AI 3D Studio — image-to-3D generation, Blender integration, video conditioning",
    author="ai-3d-studio contributors",
    python_requires=">=3.10",
    packages=find_packages(exclude=["tests*", "scripts*"]),
    install_requires=[
        "pydantic>=2.0.0",
        "pyyaml>=6.0",
        "requests>=2.31.0",
    ],
    entry_points={
        "console_scripts": [
            "ai3d=ai3d.cli.main:main",
        ],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: POSIX :: Linux",
    ],
)
