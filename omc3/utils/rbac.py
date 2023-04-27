"""
RBAC
----

Provider for RBAC tokens.
Does not use pyRBAC, because there is no KERBEROS login, instead this follows
the poor man's (aka Michi) pure python RBAC
https://gitlab.cern.ch/mihostet/lhc-access-screenshot/-/blob/5de481ff9903f64c531feffbd8ea93be474dd11c/lhc_access_screenshot/minirbac.py
"""
import os
import pathlib
from typing import Optional

import requests

from omc3.utils.logging_tools import get_logger

LOGGER = get_logger(__name__)
CONFIG_FILE = 'rbac-krb5.conf'  # needs to be in the same directory as this file
RBAC_USERNAME = "OMCRBACUSERNAME"  # set this as environment variable to get a username


class RBAC:
    """ Class to handle RBAC authentication.
    Implemented as a class, so that the BASE and KRB5 Service variables
    can be set on __init__, if needed. """
    _BASE = "https://rbac-pro1.cern.ch:8443/rba/api/v1"
    _KRB5_SERVICE = "RBAC@rbac-pro-lb.cern.ch"

    def __init__(self, base: str = None, krb5_service: str = None,
                 application: str = "omc3", lifetime_minutes: int = 8 * 60):
        """
        Create this RBAC-Communicator instance.
        Args:
            base (str): base-URL of the RBAC REST API.
            krb5_service (str): Name of the RBAC-kerberos service to use.
            application (str): Name of the application the token is for.
            lifetime_minutes (int): lifetime in minutes of the validity of the RBAC token.
                                    Default lifetime is 8h
        """
        # Paths
        self.base = base or self._BASE
        self.krb5_service = krb5_service or self._KRB5_SERVICE

        # RBAC Variables
        self.application = application
        self.lifetime = lifetime_minutes

        # RBAC Output
        self.token = None
        self.user = None

    def authenticate_explicit(self, user: str, password: str, ) -> str:
        """
        Authenticate explicitly via username and password.

        Args:
            user (str): Username to register with
            password (str): Password for the user

        Returns:
            The RBAC token as a string.
        """
        LOGGER.debug("Trying to authenticate RBAC via username and password.")
        self._set_user(user)
        response = requests.post(
            f"{self.base}/authenticate/explicit",
            verify=False,
            data={
                "UserName": self.user,
                "Password": password,
                "AccountName": self.user,
                "Application": self.application,
                "Lifetime": self.lifetime
            })
        self.user = user
        return self._get_token(response)

    def authenticate_location(self, user: Optional[str] = None) -> str:
        """
        Authenticate explicitly via username and password.

        Args:
            user (str): Username to register with (Optional)

        Returns:
            The RBAC token as a string.
        """
        LOGGER.debug("Trying to authenticate RBAC from location.")
        self._set_user(user)
        response = requests.post(
            f"{self.base}/authenticate/location",
            verify=False,
            data={
                "AccountName": self.user,
                "Application": self.application,
                "Lifetime": self.lifetime
            })
        return self._get_token(response)

    def authenticate_kerberos(self) -> str:
        """
        Authenticate explicitly via username and password.

        Returns:
            The RBAC token as a string.
        """
        LOGGER.debug("Trying to authenticate RBAC from Kerberos.")
        os.environ['KRB5_CONFIG'] = str(pathlib.Path(__file__).parent / CONFIG_FILE)
        import kerberos  # in the CERN requirements, but cern-mock-import does not make sense

        LOGGER.debug("Starting Kerberos authentication.")
        res, ctx = kerberos.authGSSClientInit(self.krb5_service)
        if res != kerberos.AUTH_GSS_COMPLETE:
            raise RuntimeError("Kerberos could not authenticate.")
        kerberos.authGSSClientStep(ctx, "")
        krb_ticket = kerberos.authGSSClientResponse(ctx)
        if not krb_ticket:
            raise RuntimeError("Kerberos could not provide a valid ticket.")
        LOGGER.debug("Received Kerberos ticket.")

        self._set_user()
        LOGGER.debug("Sending Kerberos-Login data to RBAC.")
        response = requests.post(
            f"{self.base}/authenticate/kerberos",
            verify=False,
            data={
                "AccountName": self.user,
                "Krb5Ticket": krb_ticket,
                "Application": self.application,
                "Lifetime": self.lifetime
            })

        LOGGER.debug("Cleaning up Kerberos-Client.")
        kerberos.authGSSClientClean(ctx)

        return self._get_token(response)

    def _get_token(self, response: requests.Response) -> str:
        """ Check response and return RBAC token. """
        response.raise_for_status()
        # Possibly response.text() does the encoding automatically
        self.token = response.content.decode("utf-8")
        return response.content.decode("utf-8")

    def _set_user(self, user: str = None):
        """ Set user attribute. """
        self.user = user or self.user or get_os_username()
        LOGGER.debug(f"Set user as '{self.user}'.")


def get_os_username():
    """ Get the current username from the operating system. """
    # Check this manual environment variable first, so it can be used as
    # an override:
    try:
        return os.environ[RBAC_USERNAME]
    except KeyError:
        LOGGER.debug(f"Could not get {RBAC_USERNAME} from environment.")

    try:
        return os.getlogin()
    except OSError as e:
        LOGGER.debug(f"Could not get username from login. {str(e)}")

    for variable in ["LOGNAME", "USERNAME", "USER"]:
        try:
            return os.environ[variable]
        except KeyError:
            LOGGER.debug(f"Could not get {variable} from environment.")

    raise OSError("Could not determine username for RBAC from OS. "
                  f"Try setting the {RBAC_USERNAME} environmental variable"
                  f"to your desired username.")
