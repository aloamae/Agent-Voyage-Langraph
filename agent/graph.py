"""Agent de recommandation de voyages - Examen RNCP37805BC03

Architecture minimaliste : 1 nœud unique
Structured output : extraction de 6 critères booléens
Catalogue : 5 voyages prédéfinis
RESET obligatoire des critères à chaque tour
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from pydantic import BaseModel, Field
from langchain.chat_models import init_chat_model
from langgraph.graph import StateGraph, END


# =============================
#   PYDANTIC SCHEMA (Structured Output)
# =============================

class Criteres(BaseModel):
    """Schéma pour l'extraction structurée des critères"""
    plage: Optional[bool] = Field(None, description="Vacances à la plage, mer, océan")
    montagne: Optional[bool] = Field(None, description="Vacances en montagne, ski, altitude")
    ville: Optional[bool] = Field(None, description="Vacances en ville, urbain, métropole")
    sport: Optional[bool] = Field(None, description="Activités sportives, randonnée")
    detente: Optional[bool] = Field(None, description="Séjour détente, repos, spa")
    acces_handicap: Optional[bool] = Field(None, description="Accessibilité PMR, handicap")


# =============================
#   STATE (conforme examen)
# =============================

@dataclass
class State:
    """État de l'agent - Stocke uniquement le dernier échange"""
    dernier_message_utilisateur: str = ""
    dernier_message_ia: str = ""
    criteres: Dict[str, Optional[bool]] = field(
        default_factory=lambda: {
            "plage": None,
            "montagne": None,
            "ville": None,
            "sport": None,
            "detente": None,
            "acces_handicap": None
        }
    )


# =============================
#   CATALOGUE VOYAGES
# =============================
# Ordre optimisé : voyages "premium" / spécifiques en priorité

VOYAGES = [
    {
        "nom": "5 étoiles à Chamonix option ski",
        "labels": ["montagne", "sport"],
        "accessibleHandicap": False
    },
    {
        "nom": "5 étoiles à Chamonix option fondue",
        "labels": ["montagne", "detente"],
        "accessibleHandicap": True
    },
    {
        "nom": "Palavas de paillotes en paillotes",
        "labels": ["plage", "ville", "detente"],
        "accessibleHandicap": True
    },
    {
        "nom": "5 étoiles en rase campagne",
        "labels": ["campagne", "detente"],
        "accessibleHandicap": True
    },
    {
        "nom": "Randonnée camping en Lozère",
        "labels": ["sport", "montagne", "campagne"],
        "accessibleHandicap": False
    }
]


# =============================
#   PROMPTS
# =============================

PROMPT_EXTRACTION = """Tu es un extracteur de critères de voyage.

CRITÈRES (6 clés JSON obligatoires) :
- plage, montagne, ville, sport, detente, acces_handicap

RÈGLES D'EXTRACTION :
- true  = avis POSITIF explicite (ex: "je veux", "j'aime", "avec")
- false = avis NÉGATIF explicite (ex: "pas de", "sans", "éviter")
- null  = critère NON mentionné dans le message

EXEMPLES :
- "sport en montagne" → sport=true, montagne=true, autres=null
- "plage sans sport" → plage=true, sport=false, autres=null
- "détente" → detente=true, autres=null

Message utilisateur :
"{message}"

Réponds UNIQUEMENT avec un JSON valide au format Criteres."""

PROMPT_CLARIFICATION = """Je n'ai pas identifié de critères clairs dans votre message.

Pouvez-vous préciser vos préférences parmi :
- Plage
- Montagne
- Ville
- Sport
- Détente
- Accessibilité PMR

Exemple : "Je cherche un séjour à la plage avec détente" """

PROMPT_GENERATION = """Tu es conseiller voyage. Présente CE voyage uniquement :

MESSAGE UTILISATEUR : {message}
CRITÈRES IDENTIFIÉS : {criteres}
VOYAGE : {nom}
LABELS : {labels}
ACCESSIBLE HANDICAP : {accessible}

Rédige 3-4 phrases courtes :
- Reformule la demande (critères actifs seulement)
- Présente le voyage (nom + points forts pertinents)
- Termine par : "Souhaitez-vous préciser pour d'autres idées ?"

AUCUN emoji. Ton professionnel."""

PROMPT_AUCUN_MATCH = """Aucun voyage ne correspond exactement à vos critères.

Pouvez-vous ajuster vos préférences ?
Par exemple :
- Retirer un critère contraignant
- Ajouter plus de détails
- Choisir entre plusieurs options"""


# =============================
#   FONCTIONS UTILITAIRES
# =============================

def match_criteres(voyage: Dict, criteres: Dict) -> bool:
    """Vérifie si un voyage correspond aux critères (logique simple)"""
    for critere, valeur in criteres.items():
        if valeur is None:
            continue
        
        # Cas spécial : accessibilité handicap
        if critere == "acces_handicap":
            if valeur != voyage.get("accessibleHandicap", False):
                return False
        else:
            # Mapping critère → labels attendus
            label_map = {
                "plage": ["plage"],
                "montagne": ["montagne"],
                "ville": ["ville"],
                "sport": ["sport"],
                "detente": ["detente", "détente"]
            }
            labels_requis = label_map.get(critere, [])
            has_label = any(l in voyage["labels"] for l in labels_requis)
            
            # Si critère = True, le voyage doit avoir le label
            if valeur and not has_label:
                return False
            # Si critère = False, le voyage ne doit PAS avoir le label
            if not valeur and has_label:
                return False
    
    return True


def trouver_voyage(criteres: Dict) -> Optional[Dict]:
    """Retourne le voyage correspondant le mieux aux critères"""
    # Trouver tous les voyages compatibles
    matches = []
    for voyage in VOYAGES:
        if match_criteres(voyage, criteres):
            matches.append(voyage)
    
    if not matches:
        return None
    
    # Si un seul match, le retourner
    if len(matches) == 1:
        return matches[0]
    
    # Scoring : favoriser précision et éviter le "bruit"
    def score_voyage(voyage: Dict) -> tuple:
        criteres_actifs = {k: v for k, v in criteres.items() if v is True}
        
        # Mapping critère → label
        label_map = {
            "plage": "plage",
            "montagne": "montagne",
            "ville": "ville",
            "sport": "sport",
            "detente": "detente"
        }
        
        # Compter les correspondances
        matches_count = 0
        for crit in criteres_actifs:
            label = label_map.get(crit)
            if label and label in voyage["labels"]:
                matches_count += 1
        
        # Compter les labels "pertinents" (ceux dans label_map)
        relevant_labels = [l for l in voyage["labels"] if l in label_map.values()]
        total_relevant = len(relevant_labels)
        
        # Bonus accessibilité si demandée
        acces_bonus = 0
        if criteres.get("acces_handicap") is True and voyage.get("accessibleHandicap"):
            acces_bonus = 1
        
        # Score : (correspondances, -labels_superflus, accessibilité)
        # Le "-" inverse pour favoriser MOINS de labels superflus
        return (matches_count, -total_relevant, acces_bonus)
    
    # Retourner le voyage avec le meilleur score (tuple comparé élément par élément)
    best = max(matches, key=score_voyage)
    return best


async def generer_reponse_llm(voyage: Dict, criteres: Dict, message: str) -> str:
    """Génère une réponse naturelle avec le LLM"""
    model = init_chat_model("mistral-small-latest", model_provider="mistralai")
    
    # Filtrer les critères actifs (non-None)
    criteres_actifs = {k: v for k, v in criteres.items() if v is not None}
    
    prompt = PROMPT_GENERATION.format(
        message=message,
        criteres=criteres_actifs,
        nom=voyage["nom"],
        labels=", ".join(voyage["labels"]),
        accessible="Oui" if voyage["accessibleHandicap"] else "Non"
    )
    
    response = await model.ainvoke(prompt)
    return response.content


# =============================
#   NŒUD UNIQUE (conforme examen)
# =============================

async def process_message(state: State) -> Dict[str, Any]:
    """
    Nœud unique - Cycle complet conforme examen :
    1. Extraction structured output
    2. RESET critères (obligatoire)
    3. Application nouveaux critères
    4. Validation : all(None) ?
    5. Matching + génération réponse
    """
    message = state.dernier_message_utilisateur
    
    # 1. EXTRACTION avec structured output
    model = init_chat_model("mistral-small-latest", model_provider="mistralai")
    model_struct = model.with_structured_output(Criteres)
    
    prompt_extraction = PROMPT_EXTRACTION.format(message=message)
    extraits = await model_struct.ainvoke(prompt_extraction)
    
    # 2. RESET OBLIGATOIRE des critères (pas d'héritage entre tours)
    nouveaux_criteres = {k: None for k in state.criteres}
    
    # 3. APPLICATION des nouveaux critères extraits
    for k, v in extraits.dict().items():
        if v is not None:
            nouveaux_criteres[k] = v
    
    # 4. VALIDATION : aucun critère rempli ?
    if all(v is None for v in nouveaux_criteres.values()):
        return {
            "dernier_message_ia": PROMPT_CLARIFICATION,
            "criteres": nouveaux_criteres
        }
    
    # 5. MATCHING : recherche du 1er voyage correspondant
    voyage = trouver_voyage(nouveaux_criteres)
    
    if voyage:
        # Génération réponse avec LLM
        message_ia = await generer_reponse_llm(voyage, nouveaux_criteres, message)
    else:
        # Aucun voyage ne correspond
        message_ia = PROMPT_AUCUN_MATCH
    
    return {
        "dernier_message_ia": message_ia,
        "criteres": nouveaux_criteres
    }


# =============================
#   CONSTRUCTION DU GRAPHE
# =============================

def build_graph():
    """Construit le graphe LangGraph minimaliste"""
    workflow = StateGraph(State)
    
    # 1 seul nœud
    workflow.add_node("process_message", process_message)
    
    # Arêtes simples (pas de conditionnelles)
    workflow.set_entry_point("process_message")
    workflow.add_edge("process_message", END)
    
    return workflow.compile(name="Agent Voyage Examen")


# Export pour langgraph dev
graph = build_graph()
