"""Chat projections for runs produced by older, immutable agent bundles.

Raw run results, streams and model checkpoints remain unchanged. New first-party
adapters filter reasoning before emitting text; this compatibility read handles
the signed Claude JSON chunks already emitted by databricks-langchain 0.21.0.
"""

import json
import re

from databricks_agentkit.langgraph.messages import delta_contents

_PREFIX = '[{"type": "reasoning"'
_TOKEN = re.compile(r'`+|~{3,}|\[\{"type": "reasoning"')


def _answer(text: str, *, partial: bool) -> str:
    # One stream can contain several model turns separated by tool execution.
    # Decode each canonical legacy chunk, including those after answer text.
    # Code spans/fences are examples, not provider metadata.
    decoder = json.JSONDecoder()
    output = []
    position = 0
    code = None
    while match := _TOKEN.search(text, position):
        output.append(text[position : match.start()])
        token = match.group()
        if token != _PREFIX:
            if code is None:
                code = token
            elif token[0] == code[0] and len(token) >= len(code):
                code = None
        elif code is None:
            try:
                _, end = decoder.raw_decode(text, match.start())
            except json.JSONDecodeError:
                if partial:
                    # Full snapshots replace earlier text. Hold an incomplete
                    # provider block until the next snapshot can decode it.
                    return "".join(output)
            else:
                encoded = text[match.start() : end]
                output.extend(part for part in delta_contents(encoded) if isinstance(part, str))
                position = end
                continue
        output.append(token)
        position = match.end()
    output.append(text[position:])
    return "".join(output)


def chat_snapshot(row: dict) -> dict:
    """Add a display override only when a legacy text result needs filtering.

    Generic JSON results keep their existing renderer. The raw payload stays
    available for debugging; the override is never written back to Lakebase.
    """
    row = {key: value for key, value in row.items() if key != "stream_source"}
    result = (row.get("result") or {}).get("result")
    reply = result.get("reply") if isinstance(result, dict) else None
    if isinstance(reply, str):
        text, partial = reply, False
    elif result is None:
        text = (row.get("stream") or {}).get("text", "")
        # A failed/interrupted run can also end partway through a JSON chunk.
        partial = True
    else:
        return row
    if not isinstance(text, str) or _PREFIX not in text:
        return row
    answer = _answer(text, partial=partial)
    return {**row, "display_text": answer} if answer != text else row
