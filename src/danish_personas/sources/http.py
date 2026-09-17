"""Shared HTTP behaviour for Statistics Denmark source acquisition."""

import json
import time

import httpx

RETRY_ATTEMPTS = 4


def request_with_retries(
    client: httpx.Client, method: str, url: str, json_payload: dict[str, object] | None
) -> httpx.Response:
    """Perform a request, retrying transient failures with exponential backoff.

    Args:
        client:
            Open HTTP client.
        method:
            HTTP method.
        url:
            Absolute request URL.
        json_payload:
            JSON body, or None for a request without one.

    Returns:
        The first successful response.

    Raises:
        httpx.HTTPError:
            If every attempt fails.
        RuntimeError:
            If the retry loop exits without a response or an error.
    """
    last_error: httpx.HTTPError | None = None
    for attempt in range(RETRY_ATTEMPTS):
        try:
            response = client.request(method=method, url=url, json=json_payload)
            response.raise_for_status()
            return response
        except httpx.HTTPError as error:
            last_error = error
            if attempt == RETRY_ATTEMPTS - 1:
                break
            time.sleep(2**attempt)
    if last_error is None:
        message = "Request failed without an HTTP error"
        raise RuntimeError(message)
    raise last_error


def response_headers_content(response: httpx.Response) -> bytes:
    """Serialise response headers in the canonical snapshot format.

    Every immutable snapshot stores its ``response-headers.json`` in this
    format, so the encoding lives here rather than in each source adapter.

    Args:
        response:
            Response whose headers are being recorded.

    Returns:
        Encoded headers ready to write into a snapshot.
    """
    return (
        json.dumps(dict(response.headers), indent=2, sort_keys=True) + "\n"
    ).encode()
