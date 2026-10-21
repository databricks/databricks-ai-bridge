"""An ordinary FastAPI server, independent of the managed invocation protocol."""

import os
from pathlib import Path
from typing import Literal

import uvicorn
from agent.agent import run_agent
from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel, ConfigDict, Field

load_dotenv(dotenv_path=Path(__file__).parent.parent / ".env", override=False)


class Message(BaseModel):
    model_config = ConfigDict(extra="forbid")
    role: Literal["user"]
    content: str = Field(min_length=1, pattern=r"\S")


class InvocationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    input: list[Message] = Field(min_length=1)


app = FastAPI(title="Custom Framework Agent")


@app.post("/invocations")
async def invocations(body: InvocationRequest) -> dict:
    message = await run_agent([item.model_dump() for item in body.input])
    return {"output": [message], "status": "completed"}


def main() -> None:
    uvicorn.run("runtime.main:app", host="0.0.0.0", port=int(os.getenv("PORT", "8000")))
