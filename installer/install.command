#!/bin/bash
# Installs fibsemOS for this user on macOS (or Linux: bash install.command).
# Double-click it. It installs uv, puts Python and fibsem[ui] in ~/fibsem,
# makes a desktop shortcut and opens the app. Run it again to upgrade.
# No admin rights are needed.

set -euo pipefail

ENV_DIR="${FIBSEM_ENV:-$HOME/fibsem}"
PACKAGE="${FIBSEM_PACKAGE:-fibsem[ui]}"
PYTHON_VERSION="3.11"
# A fixed uv release, so the installer it runs never changes under it.
UV_VERSION="0.11.32"

failed() {
    echo
    echo "The install did not finish. See the messages above, or"
    echo "https://www.fibsemos.org/docs/installation/ for the manual steps."
    read -r -p "Press Return to close this window. "
    exit 1
}
trap failed ERR

export PATH="$HOME/.local/bin:$PATH"
if ! command -v uv >/dev/null 2>&1; then
    echo "Installing uv..."
    curl -LsSf "https://astral.sh/uv/$UV_VERSION/install.sh" | UV_NO_MODIFY_PATH=1 sh
fi

if [ ! -x "$ENV_DIR/bin/python" ]; then
    echo "Creating the environment in $ENV_DIR..."
    uv venv "$ENV_DIR" --python "$PYTHON_VERSION" --seed
fi

echo "Installing $PACKAGE..."
uv pip install --python "$ENV_DIR/bin/python" --upgrade "$PACKAGE"

echo "Making the desktop shortcut..."
"$ENV_DIR/bin/python" -c "from fibsem.tools.desktop_shortcut import create_desktop_shortcut; print(create_desktop_shortcut(overwrite=True))"

echo
echo "fibsemOS is installed. Opening AutoLamella..."
nohup "$ENV_DIR/bin/fibsem-autolamella-ui" >/dev/null 2>&1 &
