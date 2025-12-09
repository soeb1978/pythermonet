from pathlib import Path
from setuptools import setup, find_packages

root = Path(__file__).parent

# --- Version ---
ver_file = root / "version.txt"
if not ver_file.exists():
    raise RuntimeError("version.txt missing. Create it with a PEP 440 version, e.g. 0.1.0")
ver = ver_file.read_text(encoding="utf-8").strip()
if not ver:
    raise RuntimeError("version.txt is empty. Put a valid version like 0.1.0")

# --- Requirements ---
req_file = root / "requirements.txt"
requirements = []
if req_file.exists():
    # allow blanks and comments
    requirements = [
        line.strip() for line in req_file.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]
else:
    raise RuntimeError("No requirements.txt file was found")


long_description = (root / "README.md").read_text(encoding="utf-8")

setup(
    name="pythermonet",
    version=ver,
    author="",
    author_email="",
    description="A dimensioning tool for Thermonets",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=requirements,
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    include_package_data=True,
    python_requires=">=3.8.10",

)
