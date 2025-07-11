from dataclasses import dataclass

import pytest
import requests
from requests import HTTPError, Response

from omc3.utils import rbac
from omc3.utils.rbac import RBAC, RBAC_USERNAME, get_os_username
from tests.conftest import mock_module_import, random_string


# RBAC Class -------------------------------------------------------------------
class TestRBACClass:

    @pytest.mark.basic
    def test_authenticate_location_success_give_user(self, monkeypatch, mock_post):
        rbac = RBAC(application=APPLICATION)
        user = valid_users["adam"]
        token = rbac.authenticate_location(user.name)
        assert_valid_token(token, account=user.name, application=APPLICATION)

    @pytest.mark.basic
    def test_authenticate_location_success_assume_user(self, monkeypatch, mock_post):
        rbac = RBAC(application=APPLICATION)
        user = valid_users["bertha"]
        monkeypatch.setenv(RBAC_USERNAME, user.name)
        token = rbac.authenticate_location()
        assert_valid_token(token, account=user.name, application=APPLICATION)

    @pytest.mark.basic
    def test_authenticate_location_fail(self, monkeypatch, mock_post):
        rbac = RBAC(application=APPLICATION)
        username = "TestingUser"
        with pytest.raises(HTTPError) as e:
            rbac.authenticate_location(username)
        assert REASONS["user"] in str(e)

    @pytest.mark.basic
    def test_authenticate_explicit_success(self, mock_post):
        rbac = RBAC(application=APPLICATION)
        user = valid_users["adam"]
        token = rbac.authenticate_explicit(user.name, user.password)
        assert_valid_token(token, account="adam")

    @pytest.mark.basic
    def test_authenticate_explicit_fail(self, mock_post):
        rbac = RBAC(application=APPLICATION)
        user = valid_users["adam"]
        with pytest.raises(HTTPError) as e:
            rbac.authenticate_explicit(user.name, "wrongpassword")
        assert REASONS["pw"] in str(e)

    @pytest.mark.basic
    def test_authenticate_kerberos_success(self, monkeypatch, mock_post):
        rbac = RBAC(application=APPLICATION)
        user = valid_users["bertha"]
        kerberos = MockKerberos(user)
        monkeypatch.setenv(RBAC_USERNAME, user.name)
        with mock_module_import("kerberos", kerberos):
            token = rbac.authenticate_kerberos()
        assert_valid_token(token, account=user.name)
        kerberos.validate()

    @pytest.mark.basic
    def test_authenticate_explicit_fail_with_monkeypatch(self, monkeypatch, mock_post):
        rbac = RBAC(application=APPLICATION)
        user = valid_users["bertha"]
        kerberos = MockKerberos(user)
        monkeypatch.setenv(RBAC_USERNAME, "adam")
        with mock_module_import("kerberos", kerberos):
            with pytest.raises(HTTPError) as e:
                rbac.authenticate_kerberos()
        assert REASONS["krb"] in str(e)


# Get OS Username Function -----------------------------------------------------

class TestGetOSUsername:
    username_variables = [RBAC_USERNAME, "LOGNAME", "USERNAME", "USER"]

    @pytest.mark.basic
    def test_variables(self, monkeypatch):
        self._disable_getlogin(monkeypatch)
        for var in self.username_variables:
            self._delete_all_uservars(monkeypatch)
            name = random_string(10)
            monkeypatch.setenv(var, name)
            assert name == get_os_username()

    @pytest.mark.basic
    def test_login(self, monkeypatch):
        name = random_string(10)
        self._delete_all_uservars(monkeypatch)
        monkeypatch.setattr(rbac.os, "getlogin", lambda: name)
        assert name == get_os_username()

    @pytest.mark.basic
    def test_all_fail(self, monkeypatch):
        self._disable_getlogin(monkeypatch)
        self._delete_all_uservars(monkeypatch)

        with pytest.raises(OSError) as e:
            get_os_username()

        assert RBAC_USERNAME in str(e)

    def _delete_all_uservars(self, monkeypatch):
        for delvar in self.username_variables:
            monkeypatch.delenv(delvar, raising=False)

    @staticmethod
    def _disable_getlogin(monkeypatch):
        def raise_os():
            raise OSError("Error")
        monkeypatch.setattr(rbac.os, "getlogin", raise_os)


# Mock -------------------------------------------------------------------------
VALID_RESPONSE = "VALID"
INVALID_RESPONSE = "INVALID"
APPLICATION = "rbacstuff.py"

REASONS = {
    "pw": "Wrong Password.",
    "user": "Unknown User.",
    "krb": "Wrong token.",
}


@pytest.fixture()
def mock_post(monkeypatch):
    """Replace requests."""
    monkeypatch.setattr(requests, "post", mock_post_fun)


@dataclass
class MockUserLogin:
    name: str
    password: str
    kerberos: str


valid_users = {
    user.name: user for user in (
        MockUserLogin("adam", "fhka45jh52fkas", "iwf273asfjaiowj"),
        MockUserLogin("bertha", "fjwoifjew823fo298", "fdsjdfsoiw3874290378934"),
    )
}


def mock_post_fun(address, **kwargs):
    assert address.startswith(f"{RBAC._BASE}/authenticate")
    mode = address.split("/")[-1]

    data = kwargs["data"]
    assert "AccountName" in data
    assert "Application" in data
    assert data["Application"] == APPLICATION
    assert "Lifetime" in data

    success = True
    reason = ""
    try:
        user_data = valid_users[data["AccountName"]]
    except KeyError:
        success = False
        reason = REASONS["user"]
    else:
        if mode == "location":
            assert len(data) == 3

        if mode == "kerberos":
            assert "Krb5Ticket" in data.keys()
            if user_data.kerberos != data["Krb5Ticket"]:
                success = False
                reason = REASONS["krb"]

        if mode == "explicit":
            assert "Password" in data.keys()
            if user_data.password != data["Password"]:
                success = False
                reason = REASONS["pw"]

    r = Response()
    r.status_code = 0 if success else 500
    r._content = f"{VALID_RESPONSE}/{data['AccountName']}/{data['Application']}/{data['Lifetime']}".encode("utf-8") if success else None
    r.reason = reason
    return r


def assert_valid_token(token, account=None, application=None, lifetime=None):
    assert token
    parts = token.split("/")
    for part, check in zip(parts, (VALID_RESPONSE, account, application, lifetime)):
        if check is not None:
            assert part == check


class MockKerberos:
    AUTH_GSS_COMPLETE = "yes"
    _MYCONTEXT = "SOME_CONTEXT_IS_NEEDED"

    def __init__(self, user: MockUserLogin):
        self.user = user

        # needs to perform these steps:
        self.init = False
        self.step = False
        self.response = False
        self.clean = False

    def authGSSClientInit(self, *args):
        assert len(args)
        assert args[0] == RBAC._KRB5_SERVICE
        self.init = True
        return self.AUTH_GSS_COMPLETE, self._MYCONTEXT

    def authGSSClientStep(self, *args):
        assert len(args)
        self.step = True
        assert args[0] == self._MYCONTEXT  # should be whatever context the Init provides

    def authGSSClientResponse(self, *args):
        assert len(args)
        assert args[0] == self._MYCONTEXT  # should be whatever context the Init provides
        self.response = True
        return self.user.kerberos

    def authGSSClientClean(self, *args):
        assert len(args)
        assert args[0] == self._MYCONTEXT  # should be whatever context the Init provides
        self.clean = True

    def validate(self):
        assert self.init
        assert self.step
        assert self.response
        assert self.clean
