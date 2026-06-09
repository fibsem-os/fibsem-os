@echo off
for %%f in ("%PREFIX%\fibsem-*.whl") do (
    "%PREFIX%\Scripts\pip.exe" install "%%f" --no-deps
    del "%%f"
)
