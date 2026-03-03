from dotenv import load_dotenv
from mlflow.genai.agent_server import AgentServer, setup_mlflow_git_based_version_tracking

load_dotenv(dotenv_path=".env", override=True)

import agent_server.agent  # noqa: E402, F401

agent_server = AgentServer("ResponsesAgent", enable_chat_proxy=True)
app = agent_server.app  # noqa: F841

try:
    setup_mlflow_git_based_version_tracking()
except Exception:
    pass

def main():
    agent_server.run(app_import_string="agent_server.start_server:app")
