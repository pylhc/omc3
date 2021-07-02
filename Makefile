# Copyright pyLHC/OMC-team <pylhc@github.com>

# Documentation for most of what you will see here can be found at the following links:
# for the GNU make special targets: https://www.gnu.org/software/make/manual/html_node/Special-Targets.html
# for python packaging: https://docs.python.org/3/distutils/introduction.html

# ANSI escape sequences for bold, cyan, dark blue, end, pink and red.
B = \033[1m
C = \033[96m
D = \033[34m
E = \033[0m
P = \033[95m
R = \033[31m

.PHONY : help archive clean doc install tests

all: install

help:
	@echo "Please use 'make $(R)<target>$(E)' where $(R)<target>$(E) is one of:"
	@echo "  $(R) archive $(E)        to create a tarball of the codebase for a specific release."
	@echo "  $(R) clean $(E)          to recursively remove build, run, and bitecode files/dirs."
	@echo "  $(R) doc $(E)            to build the documentation with `sphinx`."
	@echo "  $(R) install $(E)        to 'pip install' this package into your activated environment."
	@echo "  $(R) tests $(E)          to run tests with the the pytest package."

archive:
	@echo "$(B)Creating tarball archive of this release.$(E)"
	@echo ""
	@python setup.py sdist
	@echo ""
	@echo "$(B)Your archive is in the $(C)dist/$(E) $(B)directory. Link it to your release.$(E)"
	@echo "To install from this archive, unpack it and run '$(D)python setup.py install$(E)' from within its directory."
	@echo "You should run $(R)make clean$(E) afterwards to get rid of the output files."
	@echo ""

clean:
	@echo "Running setup clean."
	@python setup.py clean
	@echo "Cleaning up distutils remains."
	@rm -rf build
	@rm -rf dist
	@rm -rf .eggs
	@rm -rf omc3.egg-info
	@echo "Cleaning up documentation build remains."
	@rm -rf doc_build
	@echo "Cleaning up bitecode files and python cache."
	@find . -type f -name '*.py[co]' -delete -o -type d -name __pycache__ -delete
	@echo "Cleaning up pytest cache."
	@find . -type d -name '*.pytest_cache' -exec rm -rf {} + -o -type f -name '*.pytest_cache' -exec rm -rf {} + -o -type f -name 'stats.txt' -delete
	@echo "Cleaning up coverage reports."
	@find . -type f -name '.coverage' -exec rm -rf {} + -o -type f -name 'coverage.xml' -delete
	@echo "All cleaned up!\n"

doc: clean
	@echo "$(B)Creating documentation build with Sphinx.$(E)"
	@python -m sphinx -b html doc ./doc_build -d ./doc_build
	@echo "Done! Documentation source is in the $(C)doc_build/$(E) directory."

install: clean
	@echo "$(B)Installing this package to your active environment.$(E)"
	@pip install .

tests: clean
	@pytest
	@make clean

# Catch-all unknow targets without returning an error. This is a POSIX-compliant syntax.
.DEFAULT:
	@echo "Make caught an invalid target! See help output below for available targets."
	@make help