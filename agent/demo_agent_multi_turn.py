"""
Demo of running a multi-turn conversation with the travel recommendation agent
Compatible avec graph.py - Architecture RNCP37805BC03
"""
import asyncio
from graph import State, build_graph
from langgraph.checkpoint.memory import MemorySaver
from dotenv import load_dotenv

# Charger les variables d'environnement
load_dotenv()

# Créer le checkpointer pour la mémoire
memory = MemorySaver()

# Compiler le graphe avec mémoire
# Note: On recompile ici avec un checkpointer pour la démo locale
# (différent du graph.py qui n'a pas de checkpointer pour langgraph dev)
from graph import StateGraph, END, process_message

workflow = StateGraph(State)
workflow.add_node("process_message", process_message)
workflow.set_entry_point("process_message")
workflow.add_edge("process_message", END)
agent_with_memory = workflow.compile(checkpointer=memory)

# Thread ID pour la conversation
thread_id = "demo-42"
config = {"configurable": {"thread_id": thread_id}}


async def run():
    print("\n" + "="*60)
    print("DEMO AGENT VOYAGE - Conversation multi-tours")
    print("="*60 + "\n")
    
    # Tour 1
    print("🗣️  Utilisateur: Je cherche des vacances à la montagne")
    state1 = State(dernier_message_utilisateur="Je cherche des vacances à la montagne")
    res1 = await agent_with_memory.ainvoke(state1, config=config)
    print(f"🤖 Agent: {res1['dernier_message_ia']}\n")
    print(f"📊 Critères identifiés: {res1['criteres']}\n")
    print("-"*60 + "\n")
    
    # Tour 2
    print("🗣️  Utilisateur: J'aime le sport")
    state2 = State(dernier_message_utilisateur="J'aime le sport")
    res2 = await agent_with_memory.ainvoke(state2, config=config)
    print(f"🤖 Agent: {res2['dernier_message_ia']}\n")
    print(f"📊 Critères identifiés: {res2['criteres']}\n")
    print("-"*60 + "\n")
    
    # Tour 3
    print("🗣️  Utilisateur: Est-ce accessible aux personnes handicapées?")
    state3 = State(dernier_message_utilisateur="Est-ce accessible aux personnes handicapées?")
    res3 = await agent_with_memory.ainvoke(state3, config=config)
    print(f"🤖 Agent: {res3['dernier_message_ia']}\n")
    print(f"📊 Critères identifiés: {res3['criteres']}\n")
    print("-"*60 + "\n")
    
    print("✅ Demo terminée avec succès!")

if __name__ == "__main__":
    asyncio.run(run())
