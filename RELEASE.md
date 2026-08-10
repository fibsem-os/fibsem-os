# Release Process

## Versions and tags

The version in `pyproject.toml` is written by hand and the git tag is created separately, so the two can disagree. They must not: the build takes its version from `pyproject.toml` rather than from the tag, so a mismatch uploads the wrong version silently. The publish workflow checks them against each other before building.

| Version | Example | Tag | Published | Installed by |
|---------|---------|-----|-----------|--------------|
| Development | `0.5.2.dev0` | none | no | nobody — an in-repo marker for "between releases" |
| Release candidate | `0.5.2rc1` | `v0.5.2rc1` | yes, as a pre-release | Early Access, by exact pin |
| Release | `0.5.2` | `v0.5.2` | yes | everyone |

Release candidates use `rcN`, numbered from 1 — not `aN` or `bN`. `rc` means "this ships unless someone finds a problem", which is the contract with Early Access testers; `a` would say "expect churn". Two older pre-releases used `a0` (`v0.5.0a0`, `v0.5.1a0`); they stay as they are, and PEP 440 still orders them correctly ahead of the releases that followed.

`.devN` is never tagged and never published. It marks the working state of `main` between releases, and it is not worth incrementing: `fibsem_revision` (see `fibsem/versioning.py`) already identifies the exact commit, which is far finer than a hand-bumped dev number.

**The `v*` prefix is reserved for published releases.** Give build tags — a snapshot for a site, a partner, or a demo — any other prefix, such as a date. Two things key off `v*`, and neither is restricted to `main`:

- **Publishing.** A tag matching `v*.*.*` triggers the publish workflow from *any* branch, or from a commit on no branch at all: tag pushes are not branch-scoped, and no filter can make them so. The version check blocks the accidents, but not creating the tag is cheaper than recovering from one.
- **Revision reporting.** `get_revision()` measures from the nearest `v*` tag, so a stray one becomes the base of `fibsem_revision` for everyone downstream of it once the branch merges — replacing `v0.5.1-185-g…` with `v0.6.0test-12-g…` in saved experiment metadata.

pip does not install a pre-release unless asked, so a published `rc` reaches only people who pin it exactly. That is the entire distribution mechanism for Early Access — no access control is involved.

## Cutting a release candidate

1. **Update the changelog** in `CHANGES.md` for the version being prepared. Do this *before* the candidate, not at final release: testers cannot know what to exercise otherwise.

2. **Bump the version** in `pyproject.toml`:
   ```
   version = "0.5.2rc1"
   ```

3. **Commit, tag, and push**:
   ```bash
   git add pyproject.toml CHANGES.md
   git commit -m "release candidate v0.5.2rc1"
   git tag v0.5.2rc1
   git push && git push --tags
   ```

4. **GitHub Actions takes over** — the tag and the packaged version are checked against each other, the tests must pass, and the package is uploaded to PyPI, where `rcN` is treated as a pre-release automatically. A GitHub Release is then created from the tag with generated notes, marked as a pre-release so it does not take the "Latest" badge from the last real release.

5. **Put the version back** to the development marker, so `main` does not sit on a version that has already been published:
   ```
   version = "0.5.2.dev0"
   ```
   ```bash
   git add pyproject.toml
   git commit -m "bump version to 0.5.2.dev0"
   ```

6. **Tell Early Access**, with an exact pin rather than `--pre`, which would loosen resolution for every dependency in the command and not just this one:
   ```bash
   pip install fibsem==0.5.2rc1
   ```
   Members who would rather not wait to be told can subscribe to
   `https://github.com/fibsem-os/fibsem-os/releases.atom` in any feed reader —
   it carries pre-releases and their notes, and needs no GitHub account.

For a second candidate, repeat with `rc2`.

## Cutting a release

1. **Finalise the changelog** in `CHANGES.md`.

2. **Bump the version** in `pyproject.toml`, dropping the suffix:
   ```
   version = "0.5.2"
   ```
   Follow [semantic versioning](https://semver.org): `MAJOR.MINOR.PATCH`.

3. **Commit, tag, and push**:
   ```bash
   git add pyproject.toml CHANGES.md
   git commit -m "release v0.5.2"
   git tag v0.5.2
   git push && git push --tags
   ```

4. **GitHub Actions takes over** — same checks, then the upload to PyPI and the GitHub Release, which this time becomes "Latest".

5. **Open the next development version** in `pyproject.toml`:
   ```
   version = "0.5.3.dev0"
   ```
   ```bash
   git add pyproject.toml
   git commit -m "bump version to 0.5.3.dev0"
   ```

## The sequence, end to end

```
0.5.2.dev0  →  0.5.2rc1  →  0.5.2.dev0  →  0.5.2  →  0.5.3.dev0
   main         tagged         main         tagged       main
                published                  published
```

Each step is a commit. The two returns to `.dev0` exist because the version is static — `main` should never claim to be a version that has already been uploaded.

## Versioning

Releases follow `MAJOR.MINOR.PATCH`:

| Bump | When |
|------|------|
| `PATCH` | Bug fixes, small improvements |
| `MINOR` | New features, backwards compatible |
| `MAJOR` | Breaking API changes |

## If the publish fails

The version check runs alongside the tests and finishes in seconds. If it fails, the tag and `pyproject.toml` disagree and **nothing has been uploaded**. Delete the tag, fix whichever side is wrong, and tag again:

```bash
git tag -d v0.5.2rc1
git push --delete origin v0.5.2rc1
```

A version that has actually reached PyPI cannot be replaced or reused, even after yanking it. That is what the check exists to prevent.
