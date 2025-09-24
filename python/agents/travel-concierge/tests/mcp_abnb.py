# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""An example of the travel-concierge agents incorporating an MCP tool."""

import asyncio
import json

from dotenv import load_dotenv
from google.adk.artifacts.in_memory_artifact_service import InMemoryArtifactService
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.tools.agent_tool import AgentTool
from google.adk.tools.mcp_tool.mcp_toolset import MCPToolset, StdioServerParameters
from google.genai import types
from travel_concierge.agent import root_agent


load_dotenv()


async def get_tools_async():
    """Gets tools from MCP Server."""
    tools, exit_stack = await MCPToolset.from_server(
        connection_params=StdioServerParameters(
            command="npx",
            args=["-y", "@openbnb/mcp-server-airbnb", "--ignore-robots-txt"],
        )
    )
    # MCP requires maintaining a connection to the local MCP Server.
    # Using exit_stack to clean up server connection before exit.
    return tools, exit_stack


def find_agent(agent, targat_name):
    """A convenient function to find an agent from an existing agent graph."""
    result = None
    print("Matching...", agent.name)
    if agent.name == targat_name:
        return agent
    for sub_agent in agent.sub_agents:
        result = find_agent(sub_agent, targat_name)
        if result:
            break
    for tool in agent.tools:
        if isinstance(tool, AgentTool):
            result = find_agent(tool.agent, targat_name)
            if result:
                break
    return result


async def get_agent_async():
    """Creates an ADK Agent with tools from MCP Server."""
    tools, exit_stack = await get_tools_async()
    print("\nInserting Airbnb MCP tools into Travel-Concierge...")
    planner = find_agent(root_agent, "planning_agent")
    if planner:
        print("FOUND", planner.name)
        planner.tools.extend(tools)
    else:
        print("NOT FOUND")
    return root_agent, exit_stack


async def async_main(question):
    """Executes one turn of the travel_concierge agents with a query that would trigger the MCP tool."""
    session_service = InMemorySessionService()
    artifacts_service = InMemoryArtifactService()
    session = session_service.create_session(
        state={}, app_name="travel-concierge", user_id="traveler0115"
    )

    query = question
    print("[user]: ", query)
    content = types.Content(role="user", parts=[types.Part(text=query)])

    agent, exit_stack = await get_agent_async()
    runner = Runner(
        app_name="travel-concierge",
        agent=agent,
        artifact_service=artifacts_service,
        session_service=session_service,
    )

    events_async = runner.run_async(
        session_id=session.id, user_id="traveler0115", new_message=content
    )

    # Results Handling
    # print(events_async)
    async for event in events_async:
        # {'error': 'Function activities_agent is not found in the tools_dict.'}
        if not event.content:
            continue

        # print(event)
        author = event.author
        # Uncomment this to see the full event payload
        # print(f"\n[{author}]: {json.dumps(event)}")

        function_calls = [
            e.function_call for e in event.content.parts if e.function_call
        ]
        function_responses = [
            e.function_response for e in event.content.parts if e.function_response
        ]

        if event.content.parts[0].text:
            text_response = event.content.parts[0].text
            print(f"\n[{author}]: {text_response}")

        if function_calls:
            for function_call in function_calls:
                print(
                    f"\n[{author}]: {function_call.name}( {json.dumps(function_call.args)} )"
                )

        elif function_responses:
            for function_response in function_responses:
                function_name = function_response.name
                # Detect different payloads and handle accordingly
                application_payload = function_response.response
                if function_name == "airbnb_search":
                    application_payload = application_payload["result"].content[0].text
                print(
                    f"\n[{author}]: {function_name} responds -> {application_payload}"
                )

    await exit_stack.aclose()


if __name__ == "__main__":
    asyncio.run(
        async_main(
            (
                "Find me an airbnb in San Diego, April 9th, to april 13th, no flights nor itinerary needed. "
                "No need to confirm, simply return 5 choices, remember to include urls."
            )
        )
    )
