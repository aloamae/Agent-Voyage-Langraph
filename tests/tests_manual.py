# tests_manual.py — scénarios rapides (dernier message uniquement)
import asyncio
import sys
import os
from dotenv import load_dotenv

# Charger les variables d'environnement depuis le fichier .env
load_dotenv()

# Vérifier que la clé API Mistral est définie
mistral_api_key = os.getenv("MISTRAL_API_KEY")
if not mistral_api_key:
    print("ERREUR: La clé API Mistral n'est pas définie dans le fichier .env")
    sys.exit(1)
else:
    print(f"Clé API Mistral trouvée: {mistral_api_key[:5]}...")

# Ajouter le répertoire parent au chemin de recherche des modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.agent.graph import build_graph, State

graph = build_graph()

async def run(msg: str):
    res = await graph.ainvoke(State(dernier_message_utilisateur=msg))
    print("> USER:", msg)
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
