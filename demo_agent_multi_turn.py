"""
Demo of running a multi-turn conversation with short-term memory
The agent will behave the same as when using "langgraph dev" with LangGraph Studio
"""
import asyncio
from agent.graph import builder
from langgraph.checkpoint.memory import InMemorySaver
from dotenv import load_dotenv
load_dotenv()


memory = InMemorySaver()  # Synonymous to MemorySaver()
# This agent will have multi-turn memory
# as when running "langgraph dev"
agent_with_memory = builder.compile(checkpointer=memory)

# On thread id = one multi-turn conversation
thread_id = "42"
config = {"configurable": {"thread_id": thread_id}}


async def run():
    res1 = await agent_with_memory.ainvoke({"last_user_message": "Hello!"}, config=config)
    res2 = await agent_with_memory.ainvoke(
        {"last_user_message": "My name is Eric"}, config=config)
    res3 = await agent_with_memory.ainvoke(
        {"last_user_message": "I teach LangChain"}, config=config)
    # Should display 3
    print(agent_with_memory.get_state(config).values.get("message_count"))

if __name__ == "__main__":
    asyncio.run(run())