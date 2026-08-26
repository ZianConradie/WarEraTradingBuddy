import requests
from pathlib import Path


def send_notification(
    title,
    message,
    topic,
    priority=3,
    tags=None,
    delay=None,
    actions=None,
    click=None,
    attach=None,
    markdown=False,
    icon=None,
    filename=None,
    email=None,
    call=None,
    sequence_id=None,
    cache=True,
    firebase=True,
    unified_push=None,
):
    """
    Send a notification using the ntfy publishing API.

    Supports:

        - Normal text notifications
        - Local file attachments
        - Public URL attachments
        - Icons
        - Tags
        - Priority
        - Actions
        - Click URLs

    When a local file is attached, the file itself becomes the
    HTTP request body.

    Therefore the notification message is sent using X-Message.

    HTTP headers only support a restricted character set, so
    the attached-file message is sanitized before being placed
    into X-Message.
    """

    if priority not in (1, 2, 3, 4, 5):
        raise ValueError(
            "priority must be an integer from 1 to 5"
        )

    headers = {
        "Title": str(title),
        "Priority": str(priority),
    }

    # ========================================================
    # TAGS
    # ========================================================

    if tags:

        if isinstance(tags, list):

            headers["Tags"] = ",".join(
                str(tag)
                for tag in tags
            )

        else:

            headers["Tags"] = str(tags)

    # ========================================================
    # DELAY
    # ========================================================

    if delay is not None:
        headers["Delay"] = str(delay)

    # ========================================================
    # ACTIONS
    # ========================================================

    if actions:

        if isinstance(actions, list):

            headers["Actions"] = ";".join(
                str(action)
                for action in actions
            )

        else:

            headers["Actions"] = str(actions)

    # ========================================================
    # CLICK
    # ========================================================

    if click is not None:
        headers["Click"] = str(click)

    # ========================================================
    # MARKDOWN
    # ========================================================

    if markdown:
        headers["Markdown"] = "yes"

    # ========================================================
    # ICON
    # ========================================================

    if icon is not None:

        headers["Icon"] = str(icon)

    # ========================================================
    # EMAIL
    # ========================================================

    if email is not None:
        headers["Email"] = str(email)

    # ========================================================
    # CALL
    # ========================================================

    if call is not None:
        headers["Call"] = str(call)

    # ========================================================
    # SEQUENCE ID
    # ========================================================

    if sequence_id is not None:
        headers["Sequence-ID"] = str(sequence_id)

    # ========================================================
    # CACHE
    # ========================================================

    if not cache:
        headers["Cache"] = "no"

    # ========================================================
    # FIREBASE
    # ========================================================

    if not firebase:
        headers["Firebase"] = "no"

    # ========================================================
    # UNIFIED PUSH
    # ========================================================

    if unified_push is not None:
        headers["UnifiedPush"] = str(
            unified_push
        )

    # ========================================================
    # ATTACHMENT
    # ========================================================

    if attach is not None:

        attachment_path = Path(
            str(attach)
        )

        # ====================================================
        # LOCAL FILE
        # ====================================================

        if attachment_path.is_file():

            # ------------------------------------------------
            # Convert message into a header-safe string.
            #
            # Newlines become separators.
            #
            # Non-Latin-1 characters are removed.
            #
            # This is necessary because requests/http headers
            # cannot contain arbitrary Unicode characters.
            # ------------------------------------------------

            clean_message = (
                str(message)
                .replace("\r\n", "\n")
                .replace("\r", "\n")
                .replace("\n", " | ")
            )

            # Remove characters that cannot be represented
            # by the encoding used by HTTP headers.
            clean_message = (
                clean_message
                .encode(
                    "latin-1",
                    errors="ignore",
                )
                .decode(
                    "latin-1"
                )
                .strip()
            )

            headers["X-Message"] = (
                clean_message
            )

            # ------------------------------------------------
            # Filename
            # ------------------------------------------------

            if filename is not None:

                headers["Filename"] = str(
                    filename
                )

            else:

                headers["Filename"] = (
                    attachment_path.name
                )

            # ------------------------------------------------
            # MIME type
            # ------------------------------------------------

            suffix = (
                attachment_path
                .suffix
                .lower()
            )

            if suffix == ".png":

                headers["Content-Type"] = (
                    "image/png"
                )

            elif suffix in (
                ".jpg",
                ".jpeg",
            ):

                headers["Content-Type"] = (
                    "image/jpeg"
                )

            elif suffix == ".gif":

                headers["Content-Type"] = (
                    "image/gif"
                )

            elif suffix == ".webp":

                headers["Content-Type"] = (
                    "image/webp"
                )

            else:

                headers["Content-Type"] = (
                    "application/octet-stream"
                )

            # ------------------------------------------------
            # SEND THE SINGLE NOTIFICATION
            # ------------------------------------------------

            with attachment_path.open(
                "rb"
            ) as file:

                response = requests.put(
                    f"https://ntfy.sh/{topic}",
                    data=file,
                    headers=headers,
                    timeout=30,
                )

            response.raise_for_status()

            return response.json()

        # ====================================================
        # PUBLIC URL ATTACHMENT
        # ====================================================

        else:

            headers["Attach"] = str(
                attach
            )

            if filename is not None:

                headers["Filename"] = str(
                    filename
                )

    # ========================================================
    # NORMAL TEXT NOTIFICATION
    # ========================================================

    response = requests.post(
        f"https://ntfy.sh/{topic}",
        data=str(message).encode(
            "utf-8"
        ),
        headers=headers,
        timeout=15,
    )

    response.raise_for_status()

    return response.json()