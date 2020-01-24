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

.PHONY : help archive checklist clean format install lint tests

all: install

help:
	@echo "Please use 'make $(R)<target>$(E)' where $(R)<target>$(E) is one of:"
	@echo "  $(R) archive $(E)        to create a tarball of the codebase for a specific release."
	@echo "  $(R) checklist $(E)      to print a pre-release check-list."
	@echo "  $(R) clean $(E)          to recursively remove build, run, and bitecode files/dirs."
	@echo "  $(R) format $(E)         to recursively apply PEP8 formatting through the 'Black' cli tool."
	@echo "  $(R) install $(E)        to 'pip install' this package into your activated environment."
	@echo "  $(R) lint $(E)           to lint your python code though 'pylint'."
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

checklist:
	@echo "Here is a small pre-release check-list:"
	@echo "  - Check that you have tagged the release branch."
	@echo "  - Check you have updated the version number in $(C)__init__.py$(E) according to semantic versioning."
	@echo "  - Check the branch tag matches this release's package version."
	@echo "  - After merging and pushing this release from $(P)master$(E) to $(P)origin/master$(E):"
	@echo "     - Run 'make $(R)archive$(E)'."
	@echo "     - Create a Github release and attach the created tarball to it."
	@echo "     - Run 'make $(R)clean$(E)'."

clean:
	@echo "Running setup clean."
	@python setup.py clean
	@echo "Cleaning up distutils remains."
	@rm -rf build
	@rm -rf dist
	@rm -rf .eggs
	@rm -rf omc3.egg-info
	@echo "Cleaning up bitecode files and python cache."
	@find . -type f -name '*.py[co]' -delete -o -type d -name __pycache__ -delete
	@echo "Cleaning up pytest cache."
	@find . -type f -name '*.pytest_cache' -delete -o -type f -name 'stats.txt' -delete -o -type d -name '*.pytest_cache' -delete

format:
	@echo "Formatting code to PEP8 through $(D)Black$(E), default line length is 100.\n"
	@black -l 100 .

install: clean
	@echo "$(B)Installing this package to your active environment.$(E)"
	@pip install .

lint:
	@echo "Linting code, ignoring the following message IDs:"
	@echo "  - $(P)C0330$(E) $(C)'bad-continuation'$(E) since it somehow doesn't play friendly with $(D)Black$(E)."
	@echo "  - $(P)W0106$(E) $(C)'expression-not-assigned'$(E) since it triggers on class attribute reassignment."
	@pylint --max-line-length=100 --disable=C0330,W0106 omc3/

tests: clean
	@python setup.py test -n pytest
	@make clean

# Catch-all unknow targets without returning an error. This is a POSIX-compliant syntax.
.DEFAULT:
	@echo "Make caught an invalid target! See help output below for available targets."
	@make help