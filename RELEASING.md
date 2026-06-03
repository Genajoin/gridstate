# Releasing gridstate

`gridstate` is versioned straight from git tags — the tag **is** the version
(via [setuptools_scm](https://github.com/pypa/setuptools-scm)). There is no
version string to bump by hand in `pyproject.toml`. A release is just an
annotated `vX.Y.Z` tag pushed to `main`; CI does the rest.

## Pre-merge gate (every PR into `main`)

`main` is protected — a pull request can only merge after **all** required
checks pass:

- **Lint (pre-commit)** — ruff + ruff-format + file hygiene.
- **Type check (mypy)** — `mypy gridstate` must be clean (no longer advisory).
- **Test (Python 3.10 / 3.11 / 3.12)** — full `pytest` suite with coverage.
- **Build (sdist + wheel)** — `python -m build` + `twine check`, then the built
  wheel is installed into a clean virtualenv and must solve IEEE `case14`
  (a real release-readiness smoke, not just metadata validation).

Linear history is enforced and force-pushes / branch deletion are blocked. This
means whatever lands on `main` is already proven installable and solvable — so
a tag cut from `main` is always releasable.

## Cutting a release

Versioning follows [Semantic Versioning](https://semver.org/) and the version
number is derived from [Conventional Commits](https://www.conventionalcommits.org/)
by [Commitizen](https://commitizen-tools.github.io/commitizen/).

From an up-to-date, clean `main`:

```bash
# 1. Compute the next version from commit history, update CHANGELOG.md,
#    and create the annotated tag — all in one step.
cz bump

# 2. Push the commit and the tag together.
git push --follow-tags
```

`cz bump` inspects the Conventional Commit types since the last tag (`feat:` →
minor, `fix:` → patch, `BREAKING CHANGE` / `feat!:` → major) and chooses the
next version. To force a specific version (e.g. the very first release, see
below) pass it explicitly:

```bash
cz bump 0.1.0          # or: --increment MINOR / --increment PATCH
git push --follow-tags
```

Pushing a `v*` tag triggers `.github/workflows/release.yml`, which:

1. **build** — builds the sdist + wheel and runs `twine check`.
2. **publish** — uploads to PyPI via **Trusted Publishing (OIDC)** — no API
   token in secrets (environment `pypi`).
3. **github-release** — creates a GitHub Release for the tag with
   auto-generated notes and the built artifacts attached. This job depends only
   on `build`, so a Release is created even if the PyPI step is not yet wired.

## First release — one-time setup

The package has **no releases yet**. Before cutting `v0.1.0`, do the PyPI
trusted-publisher setup once (otherwise the `publish` job fails — the
`github-release` job still succeeds):

1. On [pypi.org](https://pypi.org/), register / reserve the project name
   **`gridstate`** (the first upload also works via a *pending* trusted
   publisher if the name is free).
2. Add a **trusted publisher** for the project:
   - Owner: `Genajoin`
   - Repository: `gridstate`
   - Workflow: `release.yml`
   - Environment: `pypi`
3. In the GitHub repo settings, create the **`pypi`** environment (optionally
   with required reviewers / a deployment branch rule limited to tags).

Then cut the first tag:

```bash
git switch main && git pull
cz bump 0.1.0          # writes CHANGELOG, creates tag v0.1.0
git push --follow-tags
```

> The project still carries the `Development Status :: 2 - Pre-Alpha` classifier
> and a `0.x` version line, so the public API may change between minor versions
> until `1.0.0`.

## Notes

- Don't edit the version in `pyproject.toml` — it is `dynamic` and comes from
  the tag. Between releases setuptools_scm produces a dev version
  (`0.1.dev3+g<sha>`); that is expected.
- `CHANGELOG.md` is maintained by `cz bump`; avoid editing released sections by
  hand. Keep an `## Unreleased` section for notes ahead of a bump if useful.
- A botched release cannot be re-uploaded to PyPI under the same version —
  yank it and release a new patch version instead.
