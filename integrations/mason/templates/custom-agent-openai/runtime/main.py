import os
from pathlib import Path
from typing import Any

import uvicorn
from agent.agent import invoke
from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel, ConfigDict, Field

# .env fills unset config only; the real environment wins (override=False). `mason dev -p` and
# the deploy platform inject DATABRICKS_* into the process, and a checked-in .env must not clobber
# them — overriding only the profile while leaving the injected host mismatches host and credential.
load_dotenv(dotenv_path=Path(__file__).parent.parent / ".env", override=False)


class InvocationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    input: list[dict[str, Any]] = Field(default_factory=list)


app = FastAPI(title="Custom OpenAI Agent")


@app.post("/invocations")
async def invocations(body: InvocationRequest) -> dict[str, Any]:
    return await invoke(body.input)


def main() -> None:
    uvicorn.run("runtime.main:app", host="0.0.0.0", port=int(os.getenv("PORT", "8000")))
