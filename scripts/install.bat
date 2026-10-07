CALL conda.bat create -n fibsem python=3.11 pip
CALL activate fibsem
pip install -e .
python shortcut.py