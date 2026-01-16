#!/usr/bin/env python3
"""Generate release workflow YAML files for databricks packages."""

from dataclasses import dataclass
from pathlib import Path


@dataclass
class Package:
    name: str
    working_dir: str | None = None  # None means root level


PACKAGES = [
    Package("databricks-ai-bridge"),
    Package("databricks-langchain", "integrations/langchain"),
    Package("databricks-mcp", "databricks_mcp"),
    Package("databricks-openai", "integrations/openai"),
]


def generate_workflow(pkg: Package) -> str:
    """Generate a release workflow YAML for a package."""
    is_root = pkg.working_dir is None
    dist_path = "dist/" if is_root else f"{pkg.working_dir}/dist/"

    # Build the defaults section for non-root packages
    defaults_section = ""
    if not is_root:
        defaults_section = f"""    defaults:
      run:
        working-directory: {pkg.working_dir}
"""

    # Build packages-dir parameter for non-root packages
    packages_dir_pypi = ""
    packages_dir_testpypi = ""
    if not is_root:
        packages_dir_pypi = f"""
        with:
          packages-dir: {pkg.working_dir}/dist/"""
        packages_dir_testpypi = f"""
          packages-dir: {pkg.working_dir}/dist/"""

    return f"""# GENERATED FILE - DO NOT EDIT DIRECTLY
# Regenerate with: uv run python .github/workflows/generate_release_workflows.py

name: Release {pkg.name}

on:
  push:
    tags:
      - "{pkg.name}-v*"
  workflow_dispatch:
    inputs:
      production:
        description: "Publish to PyPI? (If unchecked, will publish to TestPyPI)"
        required: true
        default: false
        type: boolean

jobs:
  release:
    runs-on:
      group: databricks-protected-runner-group
      labels: linux-ubuntu-latest

    permissions:
      id-token: write
      contents: write
    environment:
      name: ${{{{ inputs.production && 'pypi' || 'testpypi' }}}}
      url: ${{{{ inputs.production && 'https://pypi.org/p/{pkg.name}' || 'https://test.pypi.org/p/{pkg.name}' }}}}
{defaults_section}    steps:
      - uses: actions/checkout@v4

      - uses: actions/setup-python@v5
        with:
          python-version: "3.12"

      - name: Install build tools
        run: pip install build

      - name: Verify version matches tag
        id: verify-version
        run: |
          TAG_VERSION=${{GITHUB_REF#refs/tags/{pkg.name}-v}}
          PKG_VERSION=$(python -c "import tomllib; print(tomllib.load(open('pyproject.toml', 'rb'))['project']['version'])")
          echo "version=$PKG_VERSION" >> $GITHUB_OUTPUT
          if [ "$TAG_VERSION" != "$PKG_VERSION" ]; then
            echo "Tag version ($TAG_VERSION) does not match package version ($PKG_VERSION)"
            exit 1
          fi

      - name: Build package
        run: python -m build

      - name: Store the distribution packages
        uses: actions/upload-artifact@v5
        with:
          name: dist-{pkg.name}
          path: {dist_path}

      - uses: ncipollo/release-action@v1
        if: inputs.production
        with:
          artifacts: "{dist_path}*.whl,{dist_path}*.tar.gz"
          draft: true
          generateReleaseNotes: true

      - name: Publish to PyPI
        if: inputs.production
        uses: pypa/gh-action-pypi-publish@release/v1{packages_dir_pypi}

      - name: Publish to Test PyPI
        if: github.event_name == 'push' || !inputs.production
        uses: pypa/gh-action-pypi-publish@release/v1
        with:
          repository-url: https://test.pypi.org/legacy/{packages_dir_testpypi}

      - name: Wait for package to be available
        run: sleep 10

      - name: Install published package from PyPI
        if: inputs.production
        run: |
          pip install {pkg.name}

      - name: Install published package from Test PyPI
        if: github.event_name == 'push' || !inputs.production
        run: |
          pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple {pkg.name}==${{{{ steps.verify-version.outputs.version }}}}

      - name: Smoke test - verify version
        run: |
          EXPECTED_VERSION=$(python -c "import tomllib; print(tomllib.load(open('pyproject.toml', 'rb'))['project']['version'])")
          INSTALLED_VERSION=$(python -c "from importlib.metadata import version; print(version('{pkg.name}'))")
          echo "Expected version: $EXPECTED_VERSION"
          echo "Installed version: $INSTALLED_VERSION"
          if [ "$EXPECTED_VERSION" != "$INSTALLED_VERSION" ]; then
            echo "Version mismatch!"
            exit 1
          fi
          echo "Smoke test passed: versions match!"

      - name: Generate package link
        run: |
          VERSION=${{{{ steps.verify-version.outputs.version }}}}
          if [ "${{{{ inputs.production }}}}" == "true" ]; then
            LINK="https://pypi.org/project/{pkg.name}/$VERSION/"
            echo "## :rocket: Package Released" >> $GITHUB_STEP_SUMMARY
            echo "" >> $GITHUB_STEP_SUMMARY
            echo "**{pkg.name} v$VERSION** has been published to PyPI!" >> $GITHUB_STEP_SUMMARY
            echo "" >> $GITHUB_STEP_SUMMARY
            echo ":link: [$LINK]($LINK)" >> $GITHUB_STEP_SUMMARY
          else
            LINK="https://test.pypi.org/project/{pkg.name}/$VERSION/"
            echo "## :test_tube: Package Released to Test PyPI" >> $GITHUB_STEP_SUMMARY
            echo "" >> $GITHUB_STEP_SUMMARY
            echo "**{pkg.name} v$VERSION** has been published to Test PyPI!" >> $GITHUB_STEP_SUMMARY
            echo "" >> $GITHUB_STEP_SUMMARY
            echo ":link: [$LINK]($LINK)" >> $GITHUB_STEP_SUMMARY
          fi
"""


def main():
    script_dir = Path(__file__).parent

    for pkg in PACKAGES:
        workflow_content = generate_workflow(pkg)
        output_file = script_dir / f"release-{pkg.name}.yml"
        output_file.write_text(workflow_content)
        print(f"Generated {output_file.name}")


if __name__ == "__main__":
    main()
