from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Core dependencies for basic functionality
install_requires = [
    "nd2reader>=3.3.0",
    "numpy>=1.20.0",
    "scikit-image>=0.19.0",
    "matplotlib>=3.5.0",
    "Pillow>=9.0.0",
]

# Phase-specific dependencies
extras_require = {
    "phase1": [
        # Phase 1 uses only core dependencies
    ],
    "phase2": [
        # Phase 2 also uses only core dependencies
    ],
    "phase3": [
        "cellpose>=2.0.0",
        "torch>=1.10.0",
        "opencv-python>=4.5.0",
    ],
    "phase3-stardist": [
        "stardist>=0.8.0",
        "tensorflow>=2.8.0",
    ],
    "phase4": [
        "cellpose>=2.0.0",
        "torch>=1.10.0",
        "opencv-python>=4.5.0",
        "scipy>=1.7.0",
    ],
    "phase5": [
        "cellpose>=2.0.0",
        "torch>=1.10.0",
        "opencv-python>=4.5.0",
        "scipy>=1.7.0",
        "pandas>=1.4.0",
        "seaborn>=0.11.0",
        "openpyxl>=3.0.0",
        "xlsxwriter>=3.0.0",
    ],
    "dev": [
        "pytest>=7.0.0",
        "pytest-cov>=3.0.0",
        "black>=22.0.0",
        "flake8>=4.0.0",
        "mypy>=0.950",
        "sphinx>=4.5.0",
        "jupyterlab>=3.3.0",
        "ipykernel>=6.0.0",
    ],
    "all": [
        "cellpose>=2.0.0",
        "torch>=1.10.0",
        "opencv-python>=4.5.0",
        "scipy>=1.7.0",
        "pandas>=1.4.0",
        "seaborn>=0.11.0",
        "openpyxl>=3.0.0",
        "xlsxwriter>=3.0.0",
        "stardist>=0.8.0",
        "tensorflow>=2.8.0",
    ],
}

setup(
    name="perinuclear-analysis",
    version="0.1.0",
    author="Gabriel Duarte",
    author_email="gabriel.duarte@osumc.edu",
    description="A phased implementation module for perinuclear signal analysis in microscopy images",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/TheSaezAtienzarLab/perinuclear-analysis",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "Topic :: Scientific/Engineering :: Image Processing",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=install_requires,
    extras_require=extras_require,
    entry_points={
        "console_scripts": [
            "perinuclear-analyze=perinuclear_analysis.cli:main",
        ],
    },
    include_package_data=True,
    package_data={
        "perinuclear_analysis": ["data/*.json", "templates/*"],
    },
)