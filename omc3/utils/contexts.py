"""
Contexts
--------

Provides contexts managers to use.
"""
from __future__ import annotations

import os
import sys
import time
import warnings
from contextlib import contextmanager


@contextmanager
def log_out(stdout=sys.stdout, stderr=sys.stderr):
    """Temporarily changes ``sys.stdout`` and ``sys.stderr``."""
    old_stdout = sys.stdout
    old_stderr = sys.stderr
    sys.stdout = stdout
    sys.stderr = stderr
    try:
        yield
    finally:
        sys.stdout = old_stdout
        sys.stderr = old_stderr


@contextmanager
def silence():
    """
    Suppress all console output, rerouting ``sys.stdout`` and ``sys.stderr`` to devnull.
    """
    with open(os.devnull, "w") as devnull, log_out(stdout=devnull, stderr=devnull):
        try:
            yield
        finally:
            devnull.close()


@contextmanager
def timeit(function):
    """Prints the time at the end of the context via ``function``."""
    start_time = time.time()
    try:
        yield
    finally:
        time_used = time.time() - start_time
        function(time_used)


@contextmanager
def suppress_warnings(warning_classes):
    """Suppress all warnings of given classes."""
    with warnings.catch_warnings(record=True) as warn_list:
        yield
    for w in warn_list:
        if not issubclass(w.category, warning_classes):
            print(f"{w.filename:s}:{w.lineno:d}: {w._category_name:s}: {w.message.message:s}",
                file=sys.stderr
            )
