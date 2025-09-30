
import asyncio
import time
from llama_index.core.tools import FunctionTool
from llama_index.core.agent.workflow import AgentWorkflow, FunctionAgent
from llama_index.llms.openai import OpenAI
from monocle_apptrace.instrumentation.common.instrumentor import setup_monocle_telemetry
from llama_index.core.agent import ReActAgent
import logging

logger = logging.getLogger(__name__)

def book_hotel(hotel_name: str):
    """Book a hotel"""
    return f"Successfully booked a stay at {hotel_name}."

def book_flight(from_airport: str, to_airport: str):
    """Book a flight"""
    return f"Successfully booked a flight from {from_airport} to {to_airport}."

def setup_agents():
    llm = OpenAI(model="gpt-4o")

    flight_tool = FunctionTool.from_defaults(
        fn=book_flight,
        name="lmx_book_flight_tool_05",
        description="Books a flight from one airport to another."
    )
    flight_agent = FunctionAgent(name="lmx_flight_booking_agent_05", tools=[flight_tool], llm=llm,
                            system_prompt="You are a flight booking agent who books flights as per the request. Once you complete the task, you handoff back to the coordinator agent only.",
                            description="Flight booking agent",
                            ###can_handoff_to=["coordinator"]
                            )

    hotel_tool = FunctionTool.from_defaults(
        fn=book_hotel,
        name="lmx_book_hotel_tool_05",
        description="Books a hotel stay."
    )
    hotel_agent = FunctionAgent(name="lmx_hotel_booking_agent_05", tools=[hotel_tool], llm=llm,
                            system_prompt="You are a hotel booking agent who books hotels as per the request. Once you complete the task, you handoff back to the coordinator agent only.",
                            description="Hotel booking agent",
                            ###can_handoff_to=["coordinator"]
                            )

    coordinator = FunctionAgent(name="lmx_coordinator_05", tools=[], llm=llm,
                            system_prompt=
                            """You are a coordinator agent who manages the flight and hotel booking agents. Separate hotel booking and flight booking tasks clearly from the input query. 
                            Delegate only hotel booking to the hotel booking agent and only flight booking to the flight booking agent.
                            Once they complete their tasks, you collect their responses and provide consolidated response to the user.""",
                            description="Travel booking coordinator agent",
                            can_handoff_to=["lmx_flight_booking_agent", "lmx_hotel_booking_agent"])

    agent_workflow = AgentWorkflow(
        agents=[coordinator, flight_agent, hotel_agent],
        root_agent=coordinator.name
    )
    return agent_workflow

async def run_agent():
    """Test multi-agent interaction with flight and hotel booking."""

    agent_workflow = setup_agents()
    requests = [
        "book a flight from San Jose to Boston and a book hotel stay at Hyatt Hotel",
#        "book a flight from San Francisco to New York and a book hotel stay at Hilton Hotel",
#        "book a flight from Los Angeles to Miami and a book hotel stay at Marriott Hotel",
    ]
    for req in requests:
        resp = await agent_workflow.run(user_msg=req)
        print(resp)

if __name__ == "__main__":
    logging.basicConfig(level=logging.WARN)
    setup_monocle_telemetry(workflow_name="travel-agent-lmx-wf-05", monocle_exporters_list='file')
    asyncio.run(run_agent())
