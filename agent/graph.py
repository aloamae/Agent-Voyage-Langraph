#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Agent de voyage — graph.py
- 1 seul nœud (LangGraph)
- init_chat_model(..., model_provider="mistralai")
- Reset des critères à chaque tour (pas d’héritage)
- Exclusivité stricte des thèmes (plage/montagne/ville)
- Heuristique "Montpellier" => plage=True, ville=True, montagne=False
- Réponse unique sans retours chariot
- State : dernier message user, dernière réponse IA, critères, proposition, dernier_voyage_id
- Gestion "oui/ok" -> détails du dernier voyage
- Export `graph` (compatible `langgraph dev`, pas de checkpointer custom)
"""
from __future__ import annotations

import os
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from pydantic import BaseModel, Field

from langchain.chat_models import init_chat_model
from langgraph.graph import StateGraph, END
from langgraph.graph.state import CompiledStateGraph

# -----------------------------------------------------------------------------
# Config & logging
# -----------------------------------------------------------------------------
load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("agent_voyage")

MODEL_NAME = os.getenv("MISTRAL_MODEL", "mistral-small-latest")
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")

# -----------------------------------------------------------------------------
# Données — 5 voyages (catalogue fixe)
# -----------------------------------------------------------------------------
VOYAGES: List[Dict[str, Any]] = [
    {
        "id": "LOZ-001",
        "nom": "Randonnée camping en Lozère",
        "description": "Aventure sportive au cœur de la nature sauvage",
        "plage": False,
        "montagne": True,
        "ville": False,
        "sport": True,
        "detente": False,
        "acces_handicap": False,
        "labels": ["sport", "montagne", "campagne"],
    },
    {
        "id": "CHAM-SPA",
        "nom": "5 étoiles à Chamonix - Option Spa & Fondue",
        "description": "Luxe et détente au pied du Mont-Blanc",
        "plage": False,
        "montagne": True,
        "ville": False,
        "sport": False,
        "detente": True,
        "acces_handicap": True,
        "labels": ["montagne", "détente"],
    },
    {
        "id": "CHAM-SKI",
        "nom": "5 étoiles à Chamonix - Option Ski",
        "description": "Sport et luxe dans la capitale de l'alpinisme",
        "plage": False,
        "montagne": True,
        "ville": False,
        "sport": True,
        "detente": False,
        "acces_handicap": False,
        "labels": ["montagne", "sport"],
    },
    {
        "id": "PAL-001",
        "nom": "Palavas de paillotes en paillotes",
        "description": "Farniente urbain sur la Méditerranée",
        "plage": True,
        "montagne": False,
        "ville": True,
        "sport": False,
        "detente": True,
        "acces_handicap": True,
        "labels": ["plage", "ville", "détente", "paillote"],
    },
    {
        "id": "CAMP-LUX",
        "nom": "5 étoiles en rase campagne",
        "description": "Havre de paix luxueux dans la nature",
        "plage": False,
        "montagne": False,
        "ville": False,
        "sport": False,
        "detente": True,
        "acces_handicap": True,
        "labels": ["campagne", "détente"],
    },
]

# -----------------------------------------------------------------------------
# Pydantic — critères (output structuré)
# -----------------------------------------------------------------------------
class Criteres(BaseModel):
    plage: Optional[bool] = Field(None)
    montagne: Optional[bool] = Field(None)
    ville: Optional[bool] = Field(None)
    sport: Optional[bool] = Field(None)
    detente: Optional[bool] = Field(None)
    acces_handicap: Optional[bool] = Field(None)

# -----------------------------------------------------------------------------
# State (dataclass) — dernier message/IA + critères + proposition
# -----------------------------------------------------------------------------
@dataclass
class State:
    premier_message_utilisateur: str = ""          # toléré, non requis
    dernier_message_utilisateur: str = ""          # requis par le cahier des charges
    dernier_message_ia: str = ""                   # requis par le cahier des charges
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
    voyages_proposes: List[Dict[str, Any]] = field(default_factory=list)
    dernier_voyage_id: Optional[str] = None        # pour gérer “oui/ok” → détails

# -----------------------------------------------------------------------------
# Prompts
# -----------------------------------------------------------------------------
PROMPT_EXTRACTION = """Tu es un assistant qui extrait les critères de voyage à partir du message utilisateur.

CRITERES (6 clés attendues dans le JSON final):
- plage: true/false/null
- montagne: true/false/null
- ville: true/false/null
- sport: true/false/null
- detente: true/false/null
- acces_handicap: true/false/null

Consignes:
- true = avis positif explicite
- false = dans tous les autres cas (avis négatif, hors-sujet, indifférent…)
- null = si le critère n'est pas mentionné
- Si le message est incompréhensible, renvoyer toutes les clés à null

Message:
"{message}"

Réponds UNIQUEMENT avec un JSON valide correspondant exactement au schéma.
"""

PROMPT_SANS_CRITERES = (
    "Je n’ai pas bien compris votre message. Pour vous aider à trouver le voyage idéal, "
    "pouvez-vous préciser vos critères (plage, montagne, ville, sport, détente, PMR) ?"
)

PROMPT_AVEC_VOYAGE = """Tu es un conseiller voyage. Présente UNIQUEMENT le voyage fourni.

MESSAGE UTILISATEUR: "{message}"

CRITERES IDENTIFIES:
{criteres_str}

VOYAGE CHOISI:
- {nom}
  Description: {description}
  Criteres: Plage={plage}, Montagne={montagne}, Ville={ville}, Sport={sport}, Detente={detente}, PMR={pmr}

Rédige 3-5 phrases maximum:
- Reformule brièvement la demande en citant UNIQUEMENT les critères identifiés (OUI/NON)
- Présente le voyage (nom + description courte + points forts s’ils sont pertinents)
- Termine par: "Souhaitez-vous préciser pour d’autres idées ?"
- AUCUN emoji. Ne rien inventer.
"""

# -----------------------------------------------------------------------------
# Utilitaires
# -----------------------------------------------------------------------------
def format_criteres_str(criteres: Dict[str, Optional[bool]]) -> str:
    # Affiche seulement OUI/NON (les None sont ignorés)
    lignes = []
    for k, v in criteres.items():
        if v is True:
            lignes.append(f"{k}: OUI")
        elif v is False:
            lignes.append(f"{k}: NON")
    return " | ".join(lignes) if lignes else "Aucun critère explicite"

def detect_hors_sujet(message: str) -> bool:
    """
    Détection légère de charabia / hors-sujet.
    """
    msg = (message or "").lower().strip()
    if len(msg) < 2:
        return True
    voyelles = sum(1 for c in msg if c in "aeiouy")
    if len(msg) > 5 and voyelles < max(1, int(len(msg) * 0.15)):
        return True
    for i in range(len(msg) - 3):
        if len(set(msg[i:i + 4])) <= 2:
            return True
    mots_voyage = [
        "voyage", "vacances", "sejour", "partir", "destination", "plage", "mer", "ocean",
        "montagne", "ski", "neige", "ville", "urbain", "campagne", "nature", "sport",
        "randonnee", "vélo", "velo", "escalade", "repos", "detente", "spa", "hotel", "camping"
    ]
    if len(msg) < 30 and not any(m in msg for m in mots_voyage):
        return True
    return False

def match_voyage(voyage: Dict[str, Any], criteres: Dict[str, Optional[bool]]) -> bool:
    for critere, valeur in criteres.items():
        if valeur is None:
            continue
        if critere in voyage and voyage[critere] != valeur:
            return False
    return True

# Intentions simples (oui/non)
AFFIRMATIONS = {
    "oui", "yes", "ok", "okay", "daccord", "d'accord", "vas y", "vas-y", "go",
    "parfait", "super", "top", "ça me va", "ca me va", "oui merci", "oui svp"
}
NEGATIONS = {
    "non", "no", "pas vraiment", "plutot pas", "plutôt pas", "bof", "nan", "nope"
}

def normalize(s: str) -> str:
    return (s or "").strip().lower().replace("’", "'").replace("  ", " ")

def is_affirmation(message: str) -> bool:
    m = normalize(message)
    return m in AFFIRMATIONS or m.startswith("oui ") or m.endswith(" oui")

def is_negation(message: str) -> bool:
    m = normalize(message)
    return m in NEGATIONS or m.startswith("non ") or m.endswith(" non")

# -----------------------------------------------------------------------------
# Nœud principal
# -----------------------------------------------------------------------------
async def process_message(state: State) -> State:
    message = state.dernier_message_utilisateur or ""

    if not state.premier_message_utilisateur:
        state.premier_message_utilisateur = message

    # Modèle
    model = init_chat_model(
        model=MODEL_NAME,
        model_provider="mistralai",
        temperature=0.2,
        api_key=MISTRAL_API_KEY,
    )

    # ----- BRANCHE "DETAILS" SI CONFIRMATION COURTE -----
    if is_affirmation(message) and state.dernier_voyage_id:
        top = next((v for v in VOYAGES if v["id"] == state.dernier_voyage_id), None)
        if top:
            prix = "Prix : 800 EUR/semaine." if top["id"] == "PAL-001" else "Prix : sur demande."
            points = ", ".join(top.get("labels", [])) or "–"
            out = (
                f"Voici plus de détails sur « {top['nom']} » : {top['description']}. "
                f"{prix} (Points forts: {points}). "
                f"Souhaitez-vous réserver, connaître les disponibilités, ou comparer avec une autre option ?"
            )
            state.dernier_message_ia = out.replace("\n", " ")
            state.voyages_proposes = [top]
            return state

    # Negation juste après une proposition → on invite à préciser
    if is_negation(message) and state.dernier_voyage_id:
        state.criteres = {k: None for k in state.criteres}
        state.voyages_proposes = []
        state.dernier_voyage_id = None
        state.dernier_message_ia = (
            "D’accord. Que souhaitez-vous changer ? (plage, montagne, ville, sport, détente, PMR)"
        ).replace("\n", " ")
        return state

    # Hors-sujet -> demander des critères (laisser None)
    if detect_hors_sujet(message):
        state.criteres = {k: None for k in state.criteres}
        state.voyages_proposes = []
        state.dernier_message_ia = (
            "Je n’ai pas bien compris votre message. Pour vous aider à trouver le voyage idéal, "
            "pouvez-vous préciser vos critères (plage, montagne, ville, sport, détente, PMR) ?"
        ).replace("\n", " ")
        return state

    # --- RESET critères à CHAQUE tour ---
    state.criteres = {
        "plage": None, "montagne": None, "ville": None,
        "sport": None, "detente": None, "acces_handicap": None
    }

    # Extraction structurée
    model_struct = model.with_structured_output(Criteres)
    extraits: Dict[str, Optional[bool]] = (await model_struct.ainvoke(
        PROMPT_EXTRACTION.format(message=message)
    )).dict()

    # Heuristique Montpellier => plage & ville, montagne False
    low = message.lower()
    if "montpellier" in low:
        extraits["plage"], extraits["ville"], extraits["montagne"] = True, True, False

    # Exclusivité stricte des thèmes (si un seul True => les autres False)
    themes = ["plage", "montagne", "ville"]
    pos = [t for t in themes if extraits.get(t) is True]
    if len(pos) == 1:
        for t in themes:
            if t != pos[0]:
                extraits[t] = False

    # Exclusivité légère sport/détente
    if extraits.get("sport") is True and extraits.get("detente") is not True:
        extraits["detente"] = False
    if extraits.get("detente") is True and extraits.get("sport") is not True:
        extraits["sport"] = False

    # Application : seulement ce tour (on ignore les None)
    for k, v in extraits.items():
        if k in state.criteres and v is not None:
            state.criteres[k] = v

    # Si aucun critère rempli → demander des critères
    if all(v is None for v in state.criteres.values()):
        state.voyages_proposes = []
        state.dernier_voyage_id = None
        state.dernier_message_ia = (
            "Je n’ai pas bien compris votre message. Pour vous aider à trouver le voyage idéal, "
            "pouvez-vous préciser vos critères (plage, montagne, ville, sport, détente, PMR) ?"
        ).replace("\n", " ")
        return state

    # Matching → top-1 (proposer un seul voyage)
    matches: List[Dict[str, Any]] = [v for v in VOYAGES if match_voyage(v, state.criteres)]
    top: Optional[Dict[str, Any]] = matches[0] if matches else None
    state.voyages_proposes = [top] if top else []
    state.dernier_voyage_id = top["id"] if top else None

    # Réponse finale
    if top:
        prompt = PROMPT_AVEC_VOYAGE.format(
            message=message,
            criteres_str=format_criteres_str(state.criteres),
            nom=top["nom"],
            description=top["description"],
            plage=top["plage"],
            montagne=top["montagne"],
            ville=top["ville"],
            sport=top["sport"],
            detente=top["detente"],
            pmr=top["acces_handicap"],
        )
        resp = await model.ainvoke(prompt)
        out = (resp.content or "")
        if "Souhaitez-vous préciser pour d’autres idées ?" not in out:
            out += " Souhaitez-vous préciser pour d’autres idées ?"
        state.dernier_message_ia = out.replace("\n", " ")
    else:
        state.dernier_voyage_id = None
        state.dernier_message_ia = (
            "Aucun voyage ne correspond exactement à vos critères. "
            "Souhaitez-vous assouplir certaines préférences (ex. plage OU ville) ?"
        ).replace("\n", " ")

    return state

# -----------------------------------------------------------------------------
# Construction du graphe — export pour `langgraph dev`
# -----------------------------------------------------------------------------
def build_graph() -> CompiledStateGraph:
    wf = StateGraph(State)
    wf.add_node("process_message", process_message)
    wf.set_entry_point("process_message")
    wf.add_edge("process_message", END)
    return wf.compile()

# VARIABLE ATTENDUE PAR `langgraph dev`
graph = build_graph()
