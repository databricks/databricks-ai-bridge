"""Read-only browser for the source archive used to install an agent revision."""

from html import escape
from urllib.parse import urlencode

from databricks_agentkit.sandbox.bundle import bundle_contents


def source_page(agent, path=None):
    files = bundle_contents(bytes(agent["bundle_data"]))
    path = path or ("agent.toml" if "agent.toml" in files else "pyproject.toml")
    if path not in files:
        raise KeyError("File not found in this source snapshot")
    links = "".join(
        f'<li><a href="?{escape(urlencode({"path": name}), quote=True)}">{escape(name)}</a></li>'
        for name in sorted(files)
    )
    git = agent.get("git_source")
    provenance = (
        f"Git commit {escape(git['commit'])} · {escape(git.get('subdirectory') or '/')}"
        if git
        else "Uploaded source snapshot"
    )
    try:
        content = files[path].decode("utf-8")
    except UnicodeDecodeError:
        content = f"Binary file · {len(files[path])} bytes"
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>Source · {escape(agent["name"])}</title><link rel="stylesheet" href="/static/style.css"></head>
<body><header class="topbar"><strong>{escape(agent["name"])} · Saved source</strong>
<a href="/">Back to launcher</a></header><main class="source-browser">
<section class="panel"><p>{provenance}</p><p class="hint">Source SHA-256: {escape(agent["source_hash"])}</p>
<p class="hint">This snapshot is the source used by sessions on this revision.</p></section>
<aside class="panel"><h2>Files</h2><ul>{links}</ul></aside>
<section class="panel"><h2>{escape(path)}</h2><pre>{escape(content)}</pre></section>
</main></body></html>"""
