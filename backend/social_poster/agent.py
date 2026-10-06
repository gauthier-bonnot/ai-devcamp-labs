"""The social_poster agent. Build it here, following the labs in order
(docs/lab-1-build.html to start).

`root_agent` is the name ADK's tooling (`adk web`, the AG-UI bridge in
backend/main.py) looks for.
"""

import os
import pathlib
from typing import Any, Optional

from google.genai import types
from google.adk.models import Gemini
from google.adk.agents import Agent
from google.adk.agents.callback_context import CallbackContext
from google.adk.skills import load_skill_from_dir
from google.adk.tools import google_search
from google.adk.tools.agent_tool import AgentTool
from google.adk.tools.base_tool import BaseTool
from google.adk.tools.mcp_tool import McpToolset
from google.adk.tools.mcp_tool.mcp_session_manager import StreamableHTTPConnectionParams
from google.adk.tools.skill_toolset import SkillToolset
from google.adk.tools.tool_context import ToolContext

from .tools import generate_image

SKILLS_DIR = pathlib.Path(__file__).resolve().parents[2] / "skills"

MODEL = Gemini(
    model=os.environ.get("RESEARCH_MODEL", "gemini-2.5-flash"),
    retry_options=types.HttpRetryOptions(
        attempts=5, initial_delay=2, max_delay=30, exp_base=2,
        http_status_codes=[429, 503],
    ),
)

skill_toolset = SkillToolset(
    skills=[
        load_skill_from_dir(SKILLS_DIR / "post-formatter"),
        load_skill_from_dir(SKILLS_DIR / "platform-style"),
        load_skill_from_dir(SKILLS_DIR / "brand-voice"),
        load_skill_from_dir(SKILLS_DIR / "poster-style"),
    ],
)

BUFFER_API_KEY = os.environ.get("BUFFER_API_KEY", "")

# Buffer has no dry-run: every call is real. require_confirmation=True makes
# ADK pause each call for explicit human approval, whatever the model decides.
buffer_toolset = McpToolset(
    connection_params=StreamableHTTPConnectionParams(
        url="https://mcp.buffer.com/mcp",
        headers={"Authorization": f"Bearer {BUFFER_API_KEY}"},
    ),
    # Buffer's server exposes ~20 tools, some with very large schemas.
    # Loading them all next to AgentTool made Gemini emit Python code
    # (MALFORMED_FUNCTION_CALL) instead of structured function calls.
    # tool_filter=["list_channels", "create_post"],
    require_confirmation=True,
)

# google_search can't share a model call with function tools, so it lives in
# its own agent and reaches root_agent through AgentTool.
web_researcher = Agent(
    name="web_researcher",
    model=MODEL,
    description="Researches a topic on the web and returns a concise factual summary.",
    instruction="""Use google_search to research the topic you are given.
Return a concise factual summary: 3-6 bullet points of key facts, figures
and dates, each with its source. No opinions, no post drafting.""",
    tools=[google_search],
)

# Pipeline stage, published to session state for the frontend's StagePanel:
# idea -> researching -> drafting -> awaiting_approval -> posted
_TOOL_STAGES = {
    "web_researcher": "researching",
    "load_skill": "drafting",
    "generate_image": "drafting",
}


def _init_stage(callback_context: CallbackContext) -> None:
    if "pipeline_stage" not in callback_context.state:
        callback_context.state["pipeline_stage"] = "idea"


def _track_stage(
    tool: BaseTool, args: dict[str, Any], tool_context: ToolContext, tool_response: dict
) -> Optional[dict]:
    if tool.name in _TOOL_STAGES:
        tool_context.state["pipeline_stage"] = _TOOL_STAGES[tool.name]
    elif (
        tool.name in ("create_post", "linkedin_create_post")
        and isinstance(tool_response, dict)
        and tool_response.get("isError") is False
    ):
        tool_context.state["pipeline_stage"] = "posted"  # only a completed MCP call
    return None  # never modify the tool result


def _stage_after_agent(callback_context: CallbackContext) -> None:
    # A turn that ends mid-pipeline is waiting on the user (draft approval).
    if callback_context.state.get("pipeline_stage") in ("researching", "drafting"):
        callback_context.state["pipeline_stage"] = "awaiting_approval"


root_agent = Agent(
    name="social_poster",
    model=MODEL,
    description="Turns an idea into a ready-to-publish social media post.",
    instruction="""You help the user turn an idea into a social media post.

Workflow:
1. If the idea, audience, or platform is unclear, ask a short clarifying
   question first.
2. If the idea needs facts (news, numbers, recent releases, claims you
   can't verify), call web_researcher first and base the draft on its
   summary.
3. Before drafting, load and follow the relevant skills: post-formatter
   (structure), platform-style (rules for the target platform), and
   brand-voice (tone).
4. If the user asks for an image, load and follow the poster-style skill
   before calling generate_image, then tell them where the file was saved.
5. ALWAYS show the finished draft to the user and ask for their approval
   before any posting action. For an immediate post, ask Buffer to publish
   now rather than schedule.

Posting rules (strict):
- Only post AFTER the user has explicitly approved the exact draft shown to
  them. "Yes" or "post it" counts; enthusiasm about the idea does not.
- If the draft changes after approval, show it again and get approval again.
- Buffer posts are text only: Buffer can't read locally generated images.
  If the user wants an image, say it stays in the local gallery.
- After posting, share Buffer's returned confirmation with the user.""",
    tools=[AgentTool(agent=web_researcher), generate_image, skill_toolset]
    # An empty key makes the MCP session fail on every turn, so leave Buffer
    # out until BUFFER_API_KEY is set.
    + ([buffer_toolset] if BUFFER_API_KEY else []),
    before_agent_callback=_init_stage,
    after_tool_callback=_track_stage,
    after_agent_callback=_stage_after_agent,
)
