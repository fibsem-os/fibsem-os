# Release Process

## Steps

1. **Bump the version** in `pyproject.toml` and `conda.recipe/meta.yaml`:
   ```
   version = "0.5.0"
   ```
   Remove the `.dev0` suffix for a stable release. Follow [semantic versioning](https://semver.org): `MAJOR.MINOR.PATCH`.

2. **Update the changelog** in `CHANGES.md` with release notes for this version.

3. **Commit, tag, and push**:
   ```bash
   git add pyproject.toml CHANGES.md
   git commit -m "release v0.5.0"
   git tag v0.5.0
   git push && git push --tags
   ```

4. **GitHub Actions takes over** — two workflows run automatically:
   - `publish.yml`: tests run, then package is uploaded to PyPI
   - `build-installer.yml`: builds a Windows installer (`fibsem-*.exe`) and attaches it to the GitHub release

5. **After release**, bump the version to the next dev version in `pyproject.toml` and `conda.recipe/meta.yaml`:
   ```
   version = "0.5.1.dev0"
   ```
   Commit with `git commit -m "bump version to 0.5.1.dev0"`.

## Building the Windows installer locally

Constructor can cross-build a Windows `.exe` from Linux — it downloads pre-built conda packages, no compilation needed. Requires `conda` and `pip`.

```bash
# Install tools (once)
conda install -c conda-forge constructor
pip install build

# Build the fibsem wheel
python -m build --wheel

# Set up constructor directory
VERSION=0.5.0a0
cp dist/fibsem-*.whl constructor/fibsem-${VERSION}-py3-none-any.whl
sed -i "s/__VERSION__/${VERSION}/g" constructor/construct.yaml

# Build the Windows .exe
constructor constructor/ --platform win-64 --output-dir dist/

# Restore the template afterwards
git checkout constructor/construct.yaml
```

The installer will appear in `dist/fibsem-*.exe`.

## Versioning

Releases follow `MAJOR.MINOR.PATCH`:

| Bump | When |
|------|------|
| `PATCH` | Bug fixes, small improvements |
| `MINOR` | New features, backwards compatible |
| `MAJOR` | Breaking API changes |

Development versions use the `.dev0` suffix (e.g. `0.5.1.dev0`) between releases.
