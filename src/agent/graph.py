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
from langgraph.graph import StateGraph, START, END


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


VOYAGES: List[Dict[str, Any]] = [
{"nom": "Randonnée camping en Lozère", "plage": False, "montagne": True, "ville": False, "sport": True, "detente": False, "acces_handicap": False},
{"nom": "5 étoiles à Chamonix option fondue", "plage": False, "montagne": True, "ville": False, "sport": False, "detente": True, "acces_handicap": True},
{"nom": "5 étoiles à Chamonix option ski", "plage": False, "montagne": True, "ville": False, "sport": True, "detente": False, "acces_handicap": True},
{"nom": "Palavas de paillotes en paillotes", "plage": True, "montagne": False, "ville": True, "sport": False, "detente": True, "acces_handicap": True},
{"nom": "5 étoiles en rase campagne", "plage": False, "montagne": False, "ville": False, "sport": False, "detente": True, "acces_handicap": False},
]


MODEL_NAME = "mistral-small-latest"




# ====================================
# 3) Structured output pour les critères
# ====================================


class Criteres(BaseModel):
    plage: Optional[bool] = None
    montagne: Optional[bool] = None
    ville: Optional[bool] = None
    sport: Optional[bool] = None
    detente: Optional[bool] = None
    acces_handicap: Optional[bool] = None




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


graph = build_graph()