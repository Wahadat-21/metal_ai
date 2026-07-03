# setup.py
from setuptools import setup, find_packages

setup(
    name="composite_material_ai",
    version="0.1.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "torch>=2.0.0",
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "scikit-learn>=1.3.0",
    ],
    extras_require={
        "dev": ["pytest>=7.0.0", "black>=23.0.0", "ruff>=0.0.270"],
    },
)
