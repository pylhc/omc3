import pytest
import os

CURRENT_DIR = os.path.dirname(__file__)

def test_kmod(  ):
    pass


@pytest.fixture()
def _test_files():
    ql1_file = os.path.join(CURRENT_DIR, os.pardir, "inputs", "kmod", "MQXA.1L1.tfs")
    qr1_file = os.path.join(CURRENT_DIR, os.pardir, "inputs", "kmod", "MQXA.1R1.tfs")
    try:
        yield ql1_file, qr1_file
    finally:
        if os.path.isfile(test_file):
            os.remove(test_file)
