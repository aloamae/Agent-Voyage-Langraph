# tests_manual.py — scénarios rapides (dernier message uniquement)
import asyncio
from graph import build_graph, State

graph = build_graph()

async def run(msg: str):
    res = await graph.ainvoke(State(dernier_message_utilisateur=msg))
    print("
> USER:", msg)
    print("< BOT:", res.get("dernier_message_ia"))
    # Debug: afficher critères interprétés
    if res.get("criteres"):
        print("Critères interprétés:")
        for k, v in res["criteres"].items():
            print(f" - {k}: {v}")
    if res.get("voyages_proposes"):
        print("Propositions:")
        for v in res["voyages_proposes"]:
            print(" -", v["nom"])

async def main():
    await run("Bonjour !")
    await run("Je veux aller à la plage")
    await run("Je préfère la montagne pour faire du ski")
    await run("Finalement je veux me détendre")
    await run("srrrrzzzhdj")

if __name__ == "__main__":
    asyncio.run(main())
```
