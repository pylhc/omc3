import mimetypes
from dataclasses import dataclass
from pathlib import Path

import pytest
from requests import HTTPError

from omc3.scripts import create_logbook_entry
from tests.conftest import cli_args

INPUT = Path(__file__).parent.parent / "inputs"
INPUT_SPEC_FILES = INPUT / "lhc_harpy_output"


# noinspection PyTypeChecker
class TestMain:
    text = "Here is some text"
    files = (INPUT_SPEC_FILES / "spec_test.sdds.linx", INPUT_SPEC_FILES / "spec_test.sdds.liny")
    filenames = ("testfile1.linx", "testfile2.liny")
    tags=["Beam_1_Analysis", "Shift_Summary"]

    @pytest.mark.basic
    def test_run_all_vars(self, patch_rbac, patch_pylogbook):
        event: MockPylogbook = create_logbook_entry.main(
            logbook = create_logbook_entry.OMC_LOGBOOK,
            text=self.text,
            files=list(self.files),
            filenames=list(self.filenames),
            tags=self.tags,
        )
        assert event.text == self.text
        for f in self.filenames:
            assert f in event.attachments
        for t in self.tags:
            assert t in event.tags

    @pytest.mark.basic
    def test_run_no_filenames(self, patch_rbac, patch_pylogbook):
        event: MockPylogbook = create_logbook_entry.main(
            logbook = create_logbook_entry.OMC_LOGBOOK,
            text=self.text,
            files=list(self.files),
        )
        assert event.text == self.text
        for f in self.files:
            assert f.name in event.attachments
        assert not event.tags

    @pytest.mark.basic
    def test_run_no_attachments(self, patch_rbac, patch_pylogbook):
        event: MockPylogbook = create_logbook_entry.main(
            logbook = create_logbook_entry.OMC_LOGBOOK,
            text=self.text,
        )
        assert event.text == self.text
        assert not event.tags
        assert not event.attachments

    @pytest.mark.basic
    def test_run_no_arguments(self, patch_rbac, patch_pylogbook):
        event: MockPylogbook = create_logbook_entry.main({})
        assert not event.text
        assert not event.tags
        assert not event.attachments

    @pytest.mark.basic
    def test_different_logbook(self, patch_rbac, monkeypatch):
        logbook_name = "LHC_OP"
        monkeypatch.setattr(create_logbook_entry, "pylogbook", MockPylogbook(logbook_name))
        event: MockPylogbook = create_logbook_entry.main(
            logbook=logbook_name,
            text=self.text,
        )
        assert event.text == self.text
        assert not event.tags
        assert not event.attachments

    @pytest.mark.basic
    def test_run_cli_no_arguments(self, patch_rbac, patch_pylogbook):
        with cli_args():
            event: MockPylogbook = create_logbook_entry.main()
        assert not event.text
        assert not event.tags
        assert not event.attachments

    @pytest.mark.basic
    def test_run_all_cli_vars(self, patch_rbac, patch_pylogbook):
        with cli_args(
            "--logbook", create_logbook_entry.OMC_LOGBOOK,
            "--text", self.text,
            "--files", *[str(f) for f in self.files],
            "--filenames", *self.filenames,
            "--tags", *self.tags,
        ):
            event: MockPylogbook = create_logbook_entry.main()
        assert event.text == self.text
        for f in self.filenames:
            assert f in event.attachments
        for t in self.tags:
            assert t in event.tags


# Mock -------------------------------------------------------------------------

@dataclass
class MockRBAC:
    raises: str = ""
    token: str = "someToken"
    user: str = "MockedUser"
    application: str = "omc3"

    def __call__(self, **kwargs):
        self.application = kwargs.pop("application", self.application)
        return self

    def authenticate_location(self):
        if self.raises == "location":
            raise HTTPError("Location Error")

    def authenticate_kerberos(self):
        if self.raises == "kerberos":
            raise HTTPError("Location Error")

    def authenticate_explicit(self):
        if self.raises == "explicit":
            raise HTTPError("Location Error")


@pytest.fixture()
def patch_rbac(monkeypatch):
    monkeypatch.setattr(create_logbook_entry, "RBAC", MockRBAC())


class MockPylogbook:
    """ Mocks some of the functionality of pylogbook.
    This is all done on the same object, so it returns itself.
    The danger here is, that the function accidentally uses also the wrong
    object when operating... but I think the risk is managable.

    We should test this function by hand anyway, to see if there is logbook
    entries.
    """

    def __init__(self, expected_logbook = create_logbook_entry.OMC_LOGBOOK):
        self.expected_logbook = expected_logbook
        self.attachments = []
        self.text = None
        self.tags = None

    def Client(self, *args, **kwargs):  # noqa: N802 (mock the exact name)
        """ Mocks the pylogbook module. """
        assert kwargs["rbac_token"] == MockRBAC.token
        return self

    def ActivitiesClient(self, logbook, **kwargs):  # noqa: N802 (mock the exact name)
        """ Mocks the pylogbook module. """
        assert kwargs["client"] == self  # whatever Client returns
        assert logbook == self.expected_logbook
        return self  # return self so we keep mocking with this class

    def add_event(self, text, tags, activities = None):
        """ Mocks the pylogbook Client/ActivitiesClient-Class. """
        assert isinstance(text, str)
        for tag in tags:
            assert isinstance(tag, str)  # could also be Tag, but not in the tests
        if activities:
            assert activities == self
        self.text = text
        self.tags = tags
        return self

    def attach_content(self, contents, mime_type, name):
        """ Mocks the pylogbook Event-Class. """
        assert isinstance(name, str)
        assert isinstance(mime_type, str)
        assert "/" in mime_type
        assert isinstance(contents, str | bytes)
        self.attachments.append(name)


@dataclass
class MockAttachmentBuilder:
    """ Basically copied from pylogbook. """
    contents: str | bytes
    short_name: str
    mime_type: str

    @classmethod
    def from_file(cls, filename: Path | str):
        with open(filename, "rb") as f:
            contents = f.read()
        short_name = Path(filename).name
        mime_type = mimetypes.guess_type(short_name)[0]
        if mime_type is None:
            raise ValueError(f"Unable to determine the mime type of {filename}")
        return cls(contents, short_name, mime_type)

@pytest.fixture()
def patch_pylogbook(monkeypatch):
    monkeypatch.setattr(create_logbook_entry, "pylogbook", MockPylogbook())
    if create_logbook_entry.AttachmentBuilder is None:  # if package is not installed
        monkeypatch.setattr(create_logbook_entry, "AttachmentBuilder", MockAttachmentBuilder)
