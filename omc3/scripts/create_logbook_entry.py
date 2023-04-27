"""
Create Logbook Entry
--------------------

Simple wrapper around pylogbook to create logbook entries via python
from commandline.

**Arguments:**

*--Optional--*

- **text** *(str)*:

    Text to be written into the new logbook entry.

    default: ````


- **files** *(PathOrStr)*:

    Files to attach to the new logbook entry.


- **filenames** *(OptionalStr)*:

    Filenames to be used with the given files. If omitted, the original
    filenames will be used.


- **logbook** *(str)*:

    Name of the logbook to create the entry in.

    default: ``LHC_OMC``


- **pdf2png**:

    Convert pdf files to png and also upload these.

    action: ``store_true``


- **tags** *(str)*:

    Tags to be added to the event.

"""
import mimetypes
from pathlib import Path
from typing import Iterable, Union, List

import urllib3
from requests.exceptions import HTTPError, ConnectionError, ConnectTimeout

from generic_parser import entrypoint, EntryPointParameters
from omc3.utils.iotools import PathOrStr, OptionalStr
from omc3.utils.logging_tools import get_logger
from omc3.utils.mock import cern_network_import
from omc3.utils.rbac import RBAC
pylogbook = cern_network_import("pylogbook")  # raises ImportError if used

# for typing:
try:
    from pylogbook._attachment_builder import AttachmentBuilder, AttachmentBuilderType
    from pylogbook.models import Event
except ImportError:
    AttachmentBuilderType, Event = type(None), type(None)
    AttachmentBuilder = None

# disables unverified HTTPS warning for cern-host
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Possible errors during RBAC connection
CONNECTION_ERRORS = (HTTPError, ConnectionError, ConnectTimeout, ImportError, RuntimeError)

OMC_LOGBOOK = "LHC_OMC"
PNG_DPI = 300  # dpi for converted png (from pdf)

LOGGER = get_logger(__name__)

def get_params():
    return EntryPointParameters(
        logbook=dict(
            type=str,
            help="Name of the logbook to create the entry in.",
            default=OMC_LOGBOOK,
        ),
        text=dict(
            type=str,
            help="Text to be written into the new logbook entry.",
            default="",
        ),
        files=dict(
            type=PathOrStr,
            nargs="+",
            help="Files to attach to the new logbook entry.",
        ),
        filenames=dict(
            type=OptionalStr,
            nargs="+",
            help=(
                "Filenames to be used with the given files. "
                "If omitted, the original filenames will be used."
            ),
        ),
        tags=dict(
            type=str,
            nargs="+",
            help="Tags to be added to the event.",
        ),
        pdf2png=dict(
            action="store_true",
            help="Convert pdf files to png and also upload these."
        )
    )


@entrypoint(get_params(), strict=True)
def main(opt) -> Event:
    """ Create a new entry in the logbook and attach the given files. """
    # do attachments first as it also tests if files are there etc.
    attachments = _get_attachments(opt.files, opt.filenames, opt.pdf2png)

    # initialize logbook client
    rbac_token = _get_rbac_token()
    client = pylogbook.Client(rbac_token=rbac_token)
    logbook = pylogbook.ActivitiesClient(opt.logbook, client=client)

    # create event and upload attachments
    event = logbook.add_event(opt.text, tags=opt.tags or ())
    for attachment in attachments:
        event.attach_content(
            contents=attachment.contents,
            mime_type=attachment.mime_type,
            name=attachment.short_name,
        )
    return event


# Private Functions ------------------------------------------------------------

def _get_rbac_token() -> str:
    """ Get an RBAC token, either by location or by Kerberos. """
    rbac = RBAC(application=f"omc3.{Path(__file__).stem}")
    try:
        rbac.authenticate_location()
    except CONNECTION_ERRORS as e:
        LOGGER.debug(
            f"Getting RBAC token from location failed. "
            f"{e.__class__.__name__}: {str(e)}"
        )
    else:
        LOGGER.info(f"Logged in to RBAC via location as user {rbac.user}.")
        return rbac.token

    try:
        rbac.authenticate_kerberos()
    except CONNECTION_ERRORS as e:
        LOGGER.debug(
            f"Getting RBAC token via Kerberos failed. "
            f"{e.__class__.__name__}: {str(e)}"
        )
    else:
        LOGGER.info(f"Logged in to RBAC via Kerberos as user {rbac.user}.")
        return rbac.token

    # DEBUG ONLY ---
    # try:
    #     rbac.authenticate_explicit(user=input("Username: "), password=input("Password: "))
    # except CONNTECTION_ERRORS as e:
    #     LOGGER.debug(
    #         f"Explicit RBAC failed. "
    #         f"{e.__class__.__name__}: {str(e)}"
    #     )
    # else:
    #     LOGGER.info(f"Logged in to RBAC as user {rbac.user}.")
    #     return rbac.token

    raise NameError("Could not get RBAC token.")


def _get_attachments(files: Iterable[Union[str, Path]],
                     filenames: Iterable[str] = None,
                     pdf2png: bool = False) -> List[AttachmentBuilderType]:
    """ Read the file-attachments and assign their names. """
    if files is None:
        return []

    if filenames and len(filenames) != len(files):
        raise ValueError(
            f"List of files (length {len(files)}) and "
            f"list of filenames (length: {filenames}) "
            f"need to be of equal length."
        )

    _add_mimetypes(files)
    if filenames is None:
        filenames = [None] * len(files)

    # TODO: Return iterator, reading attachments only when needed?
    attachments = []
    for filepath, filename in zip(files, filenames):
        filepath = Path(filepath)
        attachment = AttachmentBuilder.from_file(filepath)
        attachments.append(attachment)

        # Convert pdf to png if desired
        png_attachment = None
        if pdf2png and filepath.suffix.lower() == ".pdf":
            png_attachment = _convert_pdf_to_png(filepath)

        if png_attachment:
            attachments.append(png_attachment)

        # Assign new filenames
        if filename and filename.lower() != "none":
            attachment.short_name = filename
            if png_attachment:
                png_attachment.short_name = filename.replace(".pdf", ".png").replace(".PDF", ".png")

    return attachments


def _add_mimetypes(files: Iterable[Union[str, Path]]):
    """ Adds all unknown suffixes as plain/text, which should suffice for our
    purposes.
    TODO: if it's a binary sdds file, it should be 'application/octet-stream'
          see https://stackoverflow.com/a/6783972/5609590

    This is done, because the attachment builder uses the mimetypes package
    to guess the mimetype and if it doesn't find it (e.g. `.tfs` or `.dat`)
    raises an error.
    """
    if files is None:
        return

    for f in files:
        f_path = Path(f)
        mime, _ = mimetypes.guess_type(f_path.name)
        if mime is None:
            mimetypes.add_type("text/plain", f.suffix, strict=True)


def _convert_pdf_to_png(filepath: Path):
    """ Convert the first page of a pdf file into a png image. """
    try:
        import fitz  # PyMuPDF, imported as fitz for backward compatibility reasons
    except ImportError:
        LOGGER.warning("Missing `pymupdf` package. PDF conversion not possible.")
        return None

    doc = fitz.open(filepath)  # open document

    if len(doc) > 1:
        LOGGER.warning(f"Big PDF-File with {len(doc)} pages found. "
                       "Conversion only implemented for single-page files. "
                       "Skipping conversion.")
        return None

    pixmap = doc[0].get_pixmap(dpi=PNG_DPI)  # only first page
    attachment = AttachmentBuilder.from_bytes(
        contents=pixmap.tobytes("png"),
        mime_type="image/png",
        name=filepath.with_suffix(".png").name
    )
    return attachment


# Script Mode ------------------------------------------------------------------

if __name__ == '__main__':
    main()

