import pathlib

import setuptools

# The directory containing this file
TOPLEVEL_DIR = pathlib.Path(__file__).parent.absolute()
ABOUT_FILE = TOPLEVEL_DIR / "omc3" / "__init__.py"
README = TOPLEVEL_DIR / "README.md"

# Information on the omc3 package
ABOUT_OMC3: dict = {}
with ABOUT_FILE.open("r") as f:
    exec(f.read(), ABOUT_OMC3)

with README.open("r") as docs:
    long_description = docs.read()

# Dependencies for the package itself
DEPENDENCIES = [
    "matplotlib>=3.2.0,<3.4.3",  # temporary limitations, try run test!
    "Pillow>=6.2.2",  # not our dependency but older versions crash with mpl
    "numpy>=1.19.0",
    "pandas>=1.0, <1.4",  # with 1.4.0 the interpolation in bbq_tools fails, pinning for now
    "scipy>=1.5.0",
    "scikit-learn>=0.23.0",
    "tfs-pandas>=3.1",
    "generic-parser>=1.1.0",
    "sdds>=0.1.3",
    "h5py>=2.9.0",
    "tables>=3.6.0",
    "uncertainties>=3.1.4",
    "optics-functions>=0.1.0",
    "turn_by_turn>=0.4.0",
]

# Extra dependencies
EXTRA_DEPENDENCIES = {
    "cern": [
        "jpype1>=1.3.0",
        "pytimber>=2.8.0",
    ],
    "test": [
        "pytest>=5.2",
        "pytest-cov>=2.7",
        "pytest-timeout>=1.4",
        "hypothesis>=5.0.0",
        "jpype1>=1.3.0",
        "attrs>=19.2.0",
    ],
    "doc": [
        "sphinx",
        "travis-sphinx",
        "sphinx_rtd_theme",
    ],
}
EXTRA_DEPENDENCIES.update(
    {"all": [elem for list_ in EXTRA_DEPENDENCIES.values() for elem in list_]}
)


setuptools.setup(
    name=ABOUT_OMC3["__title__"],
    version=ABOUT_OMC3["__version__"],
    description=ABOUT_OMC3["__description__"],
    long_description=long_description,
    long_description_content_type="text/markdown",
    author=ABOUT_OMC3["__author__"],
    author_email=ABOUT_OMC3["__author_email__"],
    url=ABOUT_OMC3["__url__"],
    packages=setuptools.find_packages(exclude=["tests*", "doc"]),
    include_package_data=True,
    python_requires=">=3.7",
    license=ABOUT_OMC3["__license__"],
    classifiers=[
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Natural Language :: English",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3 :: Only",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Physics",
        "Topic :: Scientific/Engineering :: Visualization",
    ],
    install_requires=DEPENDENCIES,
    tests_require=EXTRA_DEPENDENCIES["test"],
    extras_require=EXTRA_DEPENDENCIES,
)
