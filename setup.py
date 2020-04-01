import pathlib
import shlex
import sys

import setuptools
from setuptools.command.test import test as TestCommand


class PyTest(TestCommand):
    """ Allows passing commandline arguments to pytest.

        e.g. `python setup.py test -a='-o python_classes=BasicTests'`
        or   `python setup.py pytest -a '-o python_classes="BasicTests ExtendedTests"'
        or   `python setup.py test --pytest-args='--collect-only'`

    """
    user_options = [('pytest-args=', 'a', "Arguments to pass into pytest")]

    def initialize_options(self):
        TestCommand.initialize_options(self)
        self.pytest_args = ''

    def finalize_options(self):
        TestCommand.finalize_options(self)
        self.test_args = []
        self.test_suite = True

    def run_tests(self):
        import pytest

        # shlex.split() preserves quotes
        errno = pytest.main(shlex.split(self.pytest_args))
        sys.exit(errno)


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
    "matplotlib==3.2.*",
    "Pillow>=6.0.0",  # not our dependency but older versions crash with mpl
    "numpy>=1.14.1",
    "pandas==0.25.*",
    "scipy>=1.0.0",
    "scikit-learn>=0.20.3",
    "tfs-pandas>=1.0.3",
    "generic-parser>=1.0.6",
    "sdds>=0.1.3",
    "pytz>=2018.9",
    "h5py>=2.7.0",
    "pytimber>=2.8.0",
]

# Extra dependencies
EXTRA_DEPENDENCIES = {
    "setup": [
        "pytest-runner"
    ],
    "test": [
        "pytest>=5.2",
        "pytest-cov>=2.6",
        "hypothesis>=3.23.0",
        "attrs>=19.2.0",
    ],
    "doc": [
        "sphinx",
        "travis-sphinx",
        "sphinx_rtd_theme"
    ],
}
EXTRA_DEPENDENCIES.update(
    {'all': [elem for list_ in EXTRA_DEPENDENCIES.values() for elem in list_]}
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
    python_requires=">=3.6",
    license=ABOUT_OMC3["__license__"],
    cmdclass={'pytest': PyTest},  # pass test arguments
    classifiers=[
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Natural Language :: English",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3 :: Only",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Topic :: Scientific/Engineering :: Physics",
        "Topic :: Scientific/Engineering :: Visualization",
    ],
    install_requires=DEPENDENCIES,
    tests_require=EXTRA_DEPENDENCIES['test'],
    setup_requires=EXTRA_DEPENDENCIES['setup'],
    extras_require=EXTRA_DEPENDENCIES,
)
