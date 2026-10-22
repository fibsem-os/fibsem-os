"""Who may reach a Device Server (FIB-1264): every request carries a token.

The token is this install's, kept in ``~/.fibsem/device-server-token.json``
(``0600``) and made on first start, so a client on the same computer reads it
with no steps. Another computer gets one by pairing (`fibsem.server.pairing`):
``--pair`` shows a code in the terminal. Either token reaches every device:
a Device Server has no agent levels, the App Server it serves does.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from fibsem.server.auth import _matches
from fibsem.server.discovery import DISCOVERY_DIR
from fibsem.server.operator_token import load_operator_token, read_operator_token
from fibsem.server.pairing import PairedTokens, Pairing

__all__ = ["DEVICE_TOKEN_FILE", "DeviceAuth", "read_device_token"]

DEVICE_TOKEN_FILE = DISCOVERY_DIR / "device-server-token.json"
DEVICE_PAIRED_FILE = DISCOVERY_DIR / "device-server-paired.json"


def read_device_token(path: Optional[Path] = None) -> Optional[str]:
    """This computer's Device Server token, or None when none has been made."""
    return read_operator_token(path if path is not None else DEVICE_TOKEN_FILE)


@dataclass
class DeviceAuth:
    """A Device Server's token and its paired computers. Shaped as the App
    Server's `AuthConfig` where the shared routes look (`require_operator`,
    the pairing routes): every valid token is the operator's."""

    token: str
    pairing: Optional[Pairing] = None
    dashboard_token: Optional[str] = None
    # No agent asks to connect here: /connect answers connect_off.
    connections: None = None

    def is_connect(self, token: Optional[str]) -> bool:
        return False

    def is_operator(self, token: Optional[str]) -> bool:
        if _matches(token, self.token):
            return True
        return self.pairing is not None and self.pairing.tokens.match(token) is not None

    def is_agent(self, token: Optional[str]) -> bool:
        return False

    @classmethod
    def from_disk(
        cls, token_file: Optional[Path] = None, paired_file: Optional[Path] = None
    ) -> "DeviceAuth":
        """This install's token, made the first time, and its paired computers."""
        token = load_operator_token(
            token_file if token_file is not None else DEVICE_TOKEN_FILE
        )
        paired = paired_file if paired_file is not None else DEVICE_PAIRED_FILE
        return cls(token=token, pairing=Pairing(PairedTokens(paired)))
