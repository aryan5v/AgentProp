# Publishing AgentProp

AgentProp publishes Python distributions to TestPyPI and PyPI from
`.github/workflows/publish.yml`.

## Current Release

`agentprop==0.1.0a1` is published on:

- PyPI: <https://pypi.org/project/agentprop/0.1.0a1/>
- TestPyPI: <https://test.pypi.org/project/agentprop/0.1.0a1/>

## Release Checklist

1. Update `version` in `pyproject.toml`.
2. Update `CHANGELOG.md` and release notes.
3. Open and merge a PR with tests passing.
4. Create and push a tag:

   ```bash
   git tag -a vX.Y.Z -m "AgentProp X.Y.Z"
   git push origin vX.Y.Z
   ```

5. Confirm the `Publish Python Package` workflow completed successfully.

The workflow uses `skip-existing` so recreating metadata around an already
published artifact does not overwrite package files.

## Trusted Publishing

The workflow already grants `id-token: write` and uses `pypi` / `testpypi`
GitHub environments, which are the GitHub-side prerequisites for PyPI Trusted
Publishing.

To finish moving from API tokens to Trusted Publishing:

1. In PyPI and TestPyPI, add a trusted publisher for:
   - owner: `aryan5v`
   - repository: `AgentProp`
   - workflow: `publish.yml`
   - environment: `pypi` or `testpypi`
2. Remove the `password` input from the matching publish step.
3. Delete the no-longer-needed API token secret from GitHub.

Until then, the workflow can publish with `PYPI_API_TOKEN` and
`TEST_PYPI_API_TOKEN` repository secrets.
