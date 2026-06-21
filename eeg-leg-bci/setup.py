from setuptools import find_packages, setup

with open("requirements.txt") as f:
    requirements = [line.strip() for line in f if line.strip() and not line.startswith("#")]

setup(
    name="eeg-leg-bci",
    version="0.1.0",
    description="EEG lower-limb motor imagery BCI pipeline for OpenNeuro ds004362",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=requirements,
    python_requires=">=3.9",
)
