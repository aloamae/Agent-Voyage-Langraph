# graph.py — Agent de voyage (6 critères, 1 seul nœud) — LangGraph + Mistral
# --------------------------------------------------------------------------
# - Dernier message uniquement (mémoire courte)
# - Extraction des critères via structured output (Pydantic v1 API de LangChain)
# - RAG minimal en mémoire (5 voyages prédéfinis)
# - Modèle éco: mistral-small-latest (clé en .env)


from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, List
from langchain.chat_models import init_chat_model
from langchain_core.pydantic_v1 import BaseModel # compat LangChain 0.3
from langgraph.graph import START, StateGraph, END




#"nom": "5 etolles en
#rase campagne"
#"labels": ["campagne"
#"détente"],
#"accessibleHandicap":
#"oui"
#@dataclass
#class Criteres:
#    #/!! Ne can't initialize mutable objects directty in Pythor,
#    # hence a lambda
#    # We could also use "post_init"
#    # Or init the dict directly in the agent code when needed
#    criteres: dict[str, bool | None] = field(default factor=lambda: {
#        "plage": None,
#        "montagne": None,
#        "ville": None,
#        "sport": None,
#        "detente": None,
#        "acces_handicap": None,
#    })
#
#def au moins un_critere_rempli(self) → bool:
#return any(v is not None for vin self.criteres.values())

# =============================
# 1) Etat de l'agent (mémoire)
# =============================





@dataclass
class State:
    # Mémoire minimale
    premier_message_utilisateur: str = ""
    dernier_message_utilisateur: str = ""
    dernier_message_ia: str = ""
    # Critères booléens / None (= pas d'avis)
    criteres: Dict[str, Optional[bool]] = field(
        default_factory=lambda: {
            "plage": None,
            "montagne": None,
            "ville": None,
            "sport": None,
            "detente": None,
            "acces_handicap": None,
        }
    )
    # Résultats
    voyages_proposes: List[Dict[str, Any]] = field(default_factory=list)



# ===================================
# 2) Base de connaissances (RAG mémoire)
# ===================================
# 5 voyages prédéfinis conformément au cahier des charges

# Filtrage Hybride (Recommandé pour votre cas)**

# Le document mentionne le filtrage hybride qui combine recherche vectorielle (similarité sémantique) et filtres structurés


VOYAGES: List[Dict[str, Any]] = [
{"nom": "Randonnée camping en Lozère", "plage": False, "montagne": True, "ville": False, "sport": True, "detente": False, "acces_handicap": False},
{"nom": "5 étoiles à Chamonix option fondue", "plage": False, "montagne": True, "ville": False, "sport": False, "detente": True, "acces_handicap": True},
{"nom": "5 étoiles à Chamonix option ski", "plage": False, "montagne": True, "ville": False, "sport": True, "detente": False, "acces_handicap": True},
{"nom": "Palavas de paillotes en paillotes", "plage": True, "montagne": False, "ville": True, "sport": False, "detente": True, "acces_handicap": True},
{"nom": "5 étoiles en rase campagne", "plage": False, "montagne": False, "ville": False, "sport": False, "detente": True, "acces_handicap": False},
]


MODEL_NAME = "mistral-small-latest"




# ====================================
# 3) Structured output pour les critères avec Pydantic v1 API de LangChain
# ====================================
# Schéma complexe avec champs optionnels (Votre cas d'usage)
# ---------------------------------------------------------

#Pour votre cas d'usage, utilisez BaseModel avec with_structured_output car :
#
#✅ Validation automatique des types
#✅ Documentation intégrée avec Field
#✅ Compatible avec votre logique de matching _match
#✅ Facile à tester et à débugger
# ---------------------------------------------------------
# - Chaque critère peut être `True`, `False`, ou `None` (non exprimé)
# - Utilisation de `Optional[bool]` pour permettre les valeurs `None`

class Criteres(BaseModel):
    plage: Optional[bool] = None
    montagne: Optional[bool] = None
    ville: Optional[bool] = None
    sport: Optional[bool] = None
    detente: Optional[bool] = None
    acces_handicap: Optional[bool] = None

# Application au modèle
model_with_criteres = model.with_structured_output(Criteres)

# Utilisation
prompt = """Analyse cette demande et extrais les critères de destination :
'{user_query}'

Retourne un objet JSON avec les champs : plage, montagne, ville, sport, detente, acces_handicap
Mets True si le critère est demandé, False s'il est exclu, None s'il n'est pas mentionné."""

result = model_with_criteres.invoke(prompt)
# result sera un objet Criteres avec les attributs remplis




          
#La fonction `_match` est une fonction de filtrage qui sert à déterminer si un voyage correspond aux critères spécifiés par l'utilisateur. Voici son fonctionnement détaillé :
#
#1. Elle prend deux paramètres :
#   - `v` : Un dictionnaire représentant un voyage avec ses caractéristiques (plage, montagne, ville, etc.)
#   - `c` : Un dictionnaire représentant les critères de l'utilisateur (où `None` signifie que l'utilisateur n'a pas exprimé de préférence)
#
#2. Pour chaque critère dans `c` (par exemple "plage", "montagne", etc.) :
#   - Si la valeur du critère est `None`, cela signifie que l'utilisateur n'a pas exprimé de préférence pour ce critère, donc on l'ignore (`continue`)
#   - Si la valeur du critère est un booléen (`True` ou `False`) et que cette valeur ne correspond pas à celle du voyage pour ce même critère, alors le voyage ne correspond pas aux préférences de l'utilisateur et la fonction retourne `False`
#
#3. Si tous les critères exprimés par l'utilisateur correspondent aux caractéristiques du voyage, la fonction retourne `True`
#
#En résumé, cette fonction permet de filtrer les voyages pour ne garder que ceux qui correspondent exactement aux préférences exprimées par l'utilisateur. Les critères non spécifiés (valeur `None`) sont ignorés, ce qui permet une recherche flexible.
#
#Par exemple, si l'utilisateur souhaite un voyage avec plage (`plage: True`) mais n'a pas exprimé de préférence pour les autres critères, seuls les voyages avec plage seront retenus, indépendamment des autres caractéristiques.
#        


def _match(v: Dict[str, Any], c: Dict[str, Optional[bool]]) -> bool:
    for k, want in c.items():
        if want is None:
            continue
        if k in v and isinstance(want, bool) and v[k] != want:
            return False
    return True




# ======================================
# 4) Noeud unique: extraction + matching + réponse
# ======================================

async def process_message(state: State) -> Dict[str, Any]:
    """Traite le message de l'utilisateur, extrait les critères et propose des voyages."""
    
    # Initialisation du modèle
    model = init_chat_model(model=MODEL_NAME, model_provider="mistralai")
    
    # Extraction des critères via structured output
    if state.dernier_message_utilisateur:
        extraction_prompt = f"""
        Analyse le message suivant et identifie les préférences de voyage selon ces critères:
        - plage (True/False/None)
        - montagne (True/False/None)
        - ville (True/False/None)
        - sport (True/False/None)
        - detente (True/False/None)
        - acces_handicap (True/False/None)
        
        Message: {state.dernier_message_utilisateur}
        
        Réponds UNIQUEMENT avec un objet JSON contenant les critères identifiés.
        """
        
        criteres_response = await model.ainvoke(extraction_prompt)
        try:
            criteres_extraits = Criteres.parse_raw(criteres_response.content)
            # Mise à jour des critères dans l'état
            for k, v in criteres_extraits.dict().items():
                if v is not None:  # Ne mettre à jour que les critères explicitement mentionnés
                    state.criteres[k] = v
        except Exception:
            # En cas d'erreur de parsing, on continue sans modifier les critères
            pass
    
    # Recherche des voyages correspondants
    voyages_matches = []
    for voyage in VOYAGES:
        if _match(voyage, state.criteres):
            voyages_matches.append(voyage)
    
    # Mise à jour de l'état
    state.voyages_proposes = voyages_matches
    
    # Génération de la réponse
    if voyages_matches:
        voyages_str = "\n".join([f"- {v['nom']}" for v in voyages_matches])
        reponse_prompt = f"""
        L'utilisateur a demandé: "{state.dernier_message_utilisateur}"
        
        Voici les voyages qui correspondent à ses critères:
        {voyages_str}
        
        Réponds de manière amicale et professionnelle en présentant ces options.
        """
    else:
        reponse_prompt = f"""
        L'utilisateur a demandé: "{state.dernier_message_utilisateur}"
        
        Malheureusement, aucun voyage ne correspond exactement à ses critères.
        Suggère-lui de modifier ses préférences ou propose-lui les voyages les plus proches.
        """
    
    reponse = await model.ainvoke(reponse_prompt)
    state.dernier_message_ia = reponse.content
    
    return {"state": state}

def build_graph() -> StateGraph:
    """Construit le graphe LangGraph avec un seul nœud."""
    workflow = StateGraph(State)
    
    # Ajout du nœud principal
    workflow.add_node("process_message", process_message)
    
    # Configuration des transitions
    workflow.set_entry_point("process_message")
    workflow.add_edge("process_message", END)
    
    return workflow.compile()

# Création du graphe
graph = build_graph()