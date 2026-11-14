"""Build matched full/lean source bundles for the unpublished worker experiment.

Run with Python >=3.11 and uv on PATH. Uses actual wheel metadata and normal uv
resolution; never removes installed packages or edits site-packages. The normal
CLI/orchestrator must be installed separately from the worker wheels.
"""

import argparse
import hashlib
import json
import shutil
import subprocess
import tempfile
from pathlib import Path

import tomllib

REPO = Path(__file__).resolve().parents[3]
PACKAGES = (
    ".",
    "integrations/agentbricks",
    "integrations/langchain",
    "integrations/openai",
    "databricks_mcp",
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source", type=Path, required=True, help="Existing LangGraph template project"
    )
    parser.add_argument("--output", type=Path, required=True, help="New output directory")
    parser.add_argument("--suffix", default="1114", help="Unique suffix for demo resources")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    wheels = args.output / "wheels"
    wheels.mkdir()
    records = {}
    for relative in PACKAGES:
        original = REPO / relative
        with tempfile.TemporaryDirectory(prefix="agentbricks-lean-wheel-") as directory:
            stage = Path(directory)
            shutil.copytree(
                original / "src",
                stage / "src",
                ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
            )
            for filename in ("pyproject.toml", "README.md", "LICENSE.txt", "NOTICE"):
                if (original / filename).exists():
                    shutil.copy2(original / filename, stage / filename)
            digest = hashlib.sha256()
            for path in sorted(stage.rglob("*")):
                if path.is_file():
                    digest.update(str(path.relative_to(stage)).encode() + b"\0" + path.read_bytes())
            metadata = stage / "pyproject.toml"
            project = tomllib.loads(metadata.read_text())["project"]
            version = project["version"] + "+lean." + digest.hexdigest()[:12]
            metadata.write_text(
                metadata.read_text().replace(
                    f'version = "{project["version"]}"', f'version = "{version}"', 1
                )
            )
            subprocess.run(
                ["uv", "build", "--wheel", "--out-dir", str(wheels.resolve()), str(stage)],
                check=True,
            )
            wheel = next(wheels.glob(project["name"].replace("-", "_") + "-" + version + "-*.whl"))
            records[project["name"]] = {
                "version": version,
                "wheel": wheel.name,
                "sha256": hashlib.sha256(wheel.read_bytes()).hexdigest(),
            }
    for variant in ("full", "lean"):
        target = args.output / variant
        target.mkdir()
        for package in ("agent", "runtime"):
            shutil.copytree(
                args.source / package,
                target / package,
                ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
            )
        shutil.copy2(args.source / "README.md", target / "README.md")
        shutil.copytree(wheels, target / "vendor")
        dependencies = [
            "databricks-agentbricks[langgraph]>=0.3.0",
            "databricks-langchain>=0.22.0",
            # uv.sources applies to direct project requirements, not arbitrary
            # transitive requirements. Declare all five unpublished wheels.
            "databricks-ai-bridge>=0.22.0",
            "databricks-openai>=0.18.0",
            "databricks-mcp>=0.9.2",
            "mcp<2",
            "uvicorn>=0.41.0",
            "python-dotenv>=1.2.1",
            # Keep the important framework and tracing versions equal in both variants.
            "databricks-sdk==0.166.0",
            "langchain==1.4.9",
            "langchain-core==1.7.6",
            "langgraph==1.2.20",
            "mlflow-tracing==3.18.0",
        ]
        if variant == "full":
            dependencies += [
                "databricks-agentbricks[cli]",
                "databricks-ai-bridge[genie,vector-search]",
                "databricks-langchain[uc]",
                "databricks-openai[uc]",
                "mlflow==3.18.0",
                "databricks-agents==1.13.0",
            ]
        sources = "\n".join(
            f'{name} = {{ path = "vendor/{record["wheel"]}" }}' for name, record in records.items()
        )
        (target / "pyproject.toml").write_text(
            '[project]\nname = "agent-langgraph"\nversion = "0.1.0"\nrequires-python = ">=3.12,<3.13"\n'
            + "dependencies = "
            + json.dumps(dependencies, indent=4)
            + "\n\n"
            + '[build-system]\nrequires = ["hatchling"]\nbuild-backend = "hatchling.build"\n\n'
            + '[tool.hatch.build.targets.wheel]\npackages = ["agent", "runtime"]\n\n'
            + "[tool.uv.sources]\n"
            + sources
            + '\n\n[[tool.uv.index]]\nurl = "https://pypi.org/simple"\ndefault = true\n'
        )
        name = f"sb-{variant}-{args.suffix}"
        (target / "agent.toml").write_text(
            f'schema_version = 1\n\n[agent]\nframework = "langgraph"\nserver = "agentbricks"\ndeployment_name = "{name}"\n\n'
            f'[session_store]\nname = "{name}-sessions"\n\n[memory_store]\nname = "{name}-memory"\n\n'
            f'[tracing]\nexperiment_name = "/Shared/agentbricks_traces/{name}"\n'
        )
        (target / "worker-build.json").write_text(
            json.dumps({"variant": variant, "wheels": records}, indent=2)
        )
    print(f"Built {args.output}. Resolve uv.lock in each project before deploying.")  # noqa: T201


if __name__ == "__main__":
    main()
