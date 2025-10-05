#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Agent de voyage simplifie - Version conforme au cahier des charges
Architecture: 1 noeud LangGraph, memoire court-terme, 3 prompts
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, List
import os
import logging
from dotenv import load_dotenv

from langchain.chat_models import init_chat_model
from pydantic import BaseModel, Field
from langgraph.graph import START, StateGraph, END
from langgraph.graph.state import CompiledStateGraph

# Configuration
load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

MODEL_NAME = os.getenv("MISTRAL_MODEL", "mistral-small-latest")
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")

if not MISTRAL_API_KEY:
    logger.warning("MISTRAL_API_KEY manquante dans .env")

# PROMPTS - SEULEMENT 3 PROMPTS

PROMPT_EXTRACTION = """Tu es un assistant expert en analyse de preferences voyage.
Ton role: identifier les criteres de voyage dans le message utilisateur.

CRITERES (6):
1. plage: mer, ocean, plage, cote, baignade, littoral
2. montagne: montagne, ski, neige, altitude, alpes
3. ville: ville, urbain, metropole, capitale
4. sport: sport, actif, randonnee, velo, escalade
5. detente: repos, detente, calme, spa, zen, reposer
6. acces_handicap: PMR, accessibilite, handicap

MESSAGE: "{message}"

EXEMPLES CONCRETS (FEW-SHOT):

Exemple 1:
Message: "je veux aller a la plage"
JSON: {{"plage": true, "montagne": null, "ville": null, "sport": null, "detente": null, "acces_handicap": null}}

Exemple 2:
Message: "je veux faire du ski"
JSON: {{"plage": null, "montagne": true, "ville": null, "sport": true, "detente": null, "acces_handicap": null}}

Exemple 3:
Message: "je prefere me reposer"
JSON: {{"plage": null, "montagne": null, "ville": null, "sport": false, "detente": true, "acces_handicap": null}}

Exemple 4:
Message: "plutot la montagne finalement"
JSON: {{"plage": false, "montagne": true, "ville": null, "sport": null, "detente": null, "acces_handicap": null}}

Exemple 5:
Message: "en fait je prefere la montagne"
JSON: {{"plage": false, "montagne": true, "ville": null, "sport": null, "detente": null, "acces_handicap": null}}

Exemple 6:
Message: "sqdqsdqs"
JSON: {{"plage": null, "montagne": null, "ville": null, "sport": null, "detente": null, "acces_handicap": null}}

REGLES:
- True: critere souhaite
- False: critere refuse OU change de preference
- null: non mentionne
- Si "prefere X" ou "plutot X": mettre ancien critere a false
- Si charabia: TOUT a null

Analyse maintenant ce message et reponds UNIQUEMENT avec le JSON:"""

PROMPT_SANS_CRITERES = """Tu es un conseiller voyage accueillant.
Utilisateur sans preferences precises.

MESSAGE: "{message}"

INTERDICTIONS:
- NE JAMAIS utiliser emojis
- NE PAS dire Bonjour si pas une salutation
- Rester simple et direct

INSTRUCTIONS:
- 2-3 phrases maximum
- Demande poliment ce que la personne recherche
- Propose exemples : plage, montagne, ville, sport, detente
- Ton chaleureux mais professionnel

Exemple: "Pour vous aider a trouver le voyage ideal, pouvez-vous me dire ce que vous recherchez ? Nous proposons sejours plage, montagne, ville, sportifs ou detente." """

PROMPT_AVEC_VOYAGES = """Tu es un conseiller voyage enthousiaste.
Presente UNIQUEMENT les voyages de notre catalogue ci-dessous.

MESSAGE UTILISATEUR: "{message}"

CRITERES IDENTIFIES:
{criteres_str}

VOYAGES CATALOGUE COMPLET (5 voyages):
{voyages_str}

INTERDICTIONS ABSOLUES:
- NE JAMAIS utiliser emojis
- NE JAMAIS inventer voyage (Alpes maritimes, Cote Azur = INTERDIT)
- NE JAMAIS mentionner recherche internet
- Utiliser UNIQUEMENT 5 voyages catalogue avec noms exacts

REGLES:
- Presente UNIQUEMENT voyages listes
- Si aucun match: explique et propose assouplir criteres
- 5 voyages disponibles : Lozere, Chamonix Spa, Chamonix Ski, Palavas, Campagne

INSTRUCTIONS (3-4 phrases max):
1. Reformule brievement demande
2. Si match: presente voyage(s) du catalogue (nom exact)
3. Mets en avant 1-2 points forts
4. Termine par question simple

TON: Professionnel, chaleureux, SANS emojis"""

# SCHEMA PYDANTIC

class Criteres(BaseModel):
    """Schema pour extraction structuree des criteres"""
    plage: Optional[bool] = Field(None, description="Preference plage/mer")
    montagne: Optional[bool] = Field(None, description="Preference montagne")
    ville: Optional[bool] = Field(None, description="Preference ville/urbain")
    sport: Optional[bool] = Field(None, description="Preference activites sportives")
    detente: Optional[bool] = Field(None, description="Preference repos/detente")
    acces_handicap: Optional[bool] = Field(None, description="Besoin accessibilite PMR")

# STATE - MINIMAL

@dataclass
class State:
    """Etat minimal agent - Conforme cahier charges"""
    
    # Memoire court-terme
    premier_message_utilisateur: str = ""
    dernier_message_utilisateur: str = ""
    dernier_message_ia: str = ""
    
    # Criteres de recherche (6 criteres)
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
    
    # Resultats
    voyages_proposes: List[Dict[str, Any]] = field(default_factory=list)

# BASE DE DONNEES VOYAGES - CONSTANTE GLOBALE
# IMPORTANT: Liste est SEULE source de verite
# Agent NE FAIT JAMAIS recherche web ou externe
# Utilise UNIQUEMENT ces 5 voyages predefinis

VOYAGES: List[Dict[str, Any]] = [
    {
        "id": "LOZ-001",
        "nom": "Randonnee camping en Lozere",
        "description": "Aventure sportive au coeur de la nature sauvage",
        "plage": False,
        "montagne": True,
        "ville": False,
        "sport": True,
        "detente": False,
        "acces_handicap": False,
        "points_forts": [
            "Randonnees guidees",
            "Bivouac sous les etoiles", 
            "Faune exceptionnelle"
        ],
        "prix": "450 EUR/semaine"
    },
    {
        "id": "CHAM-SPA",
        "nom": "5 etoiles a Chamonix - Option Spa & Fondue",
        "description": "Luxe et detente au pied du Mont-Blanc",
        "plage": False,
        "montagne": True,
        "ville": False,
        "sport": False,
        "detente": True,
        "acces_handicap": True,
        "points_forts": [
            "Spa panoramique",
            "Restaurant gastronomique",
            "Chambres PMR"
        ],
        "prix": "2500 EUR/semaine"
    },
    {
        "id": "CHAM-SKI",
        "nom": "5 etoiles a Chamonix - Option Ski",
        "description": "Sport et luxe dans la capitale de alpinisme",
        "plage": False,
        "montagne": True,
        "ville": False,
        "sport": True,
        "detente": False,
        "acces_handicap": True,
        "points_forts": [
            "Acces direct aux pistes",
            "Materiel haut de gamme",
            "Cours ESF inclus"
        ],
        "prix": "3000 EUR/semaine"
    },
    {
        "id": "PAL-001",
        "nom": "Palavas de paillotes en paillotes",
        "description": "Farniente urbain sur la Mediterranee",
        "plage": True,
        "montagne": False,
        "ville": True,
        "sport": False,
        "detente": True,
        "acces_handicap": True,
        "points_forts": [
            "Plages privees PMR",
            "Restaurants fruits de mer",
            "Animations nocturnes"
        ],
        "prix": "800 EUR/semaine"
    },
    {
        "id": "CAMP-LUX",
        "nom": "5 etoiles en rase campagne",
        "description": "Havre de paix luxueux dans la nature",
        "plage": False,
        "montagne": False,
        "ville": False,
        "sport": False,
        "detente": True,
        "acces_handicap": False,
        "points_forts": [
            "Calme absolu",
            "Piscine naturelle",
            "Cuisine bio"
        ],
        "prix": "1800 EUR/semaine"
    },
]

# FONCTIONS UTILITAIRES

def format_criteres_str(criteres: Dict[str, Optional[bool]]) -> str:
    """Formate les criteres pour affichage"""
    actifs = []
    for key, value in criteres.items():
        if value is True:
            actifs.append(f"{key.replace('_', ' ').title()}: OUI")
        elif value is False:
            actifs.append(f"{key.replace('_', ' ').title()}: NON")
    
    return "\n".join(actifs) if actifs else "Aucun critere specifique"

def detect_hors_sujet(message: str) -> bool:
    """Detecte si le message est hors-sujet ou charabia"""
    message_lower = message.lower().strip()
    
    # Messages trop courts ou vides
    if len(message_lower) < 2:
        return True
    
    # Ratio voyelles/consonnes anormal (charabia)
    voyelles = sum(1 for c in message_lower if c in 'aeiouy')
    if len(message_lower) > 5 and voyelles < len(message_lower) * 0.15:
        return True
    
    # Repetitions excessives
    for i in range(len(message_lower) - 3):
        if (len(set(message_lower[i:i+4])) <= 2):
            return True
    
    # Mots-cles voyage
    mots_voyage = [
        "voyage", "vacances", "sejour", "partir", "destination",
        "plage", "mer", "ocean", "montagne", "ski", "neige",
        "ville", "urbain", "campagne", "nature",
        "sport", "randonnee", "velo", "escalade",
        "repos", "detente", "spa", "calme", "zen",
        "hotel", "camping", "week-end", "weekend"
    ]
    
    # Si aucun mot voyage ET message court
    if len(message_lower) < 30 and not any(mot in message_lower for mot in mots_voyage):
        if not any(c.isalpha() for c in message_lower):
            return True
    
    return False

def match_voyage(voyage: Dict[str, Any], criteres: Dict[str, Optional[bool]]) -> bool:
    """Verifie si un voyage correspond aux criteres"""
    for critere, valeur_souhaitee in criteres.items():
        if valeur_souhaitee is None:
            continue
            
        if critere in voyage:
            if voyage[critere] != valeur_souhaitee:
                return False
                
    return True

# NOEUD PRINCIPAL

async def process_message(state: State) -> State:
    """Noeud unique de traitement - Conforme cahier charges"""
    
    logger.info(f"Message: {state.dernier_message_utilisateur[:50]}...")
    
    # Sauvegarder le premier message si vide
    if not state.premier_message_utilisateur:
        state.premier_message_utilisateur = state.dernier_message_utilisateur
        logger.info("Premier message sauvegarde")
    
    # INITIALISATION MODELE
    try:
        model = init_chat_model(
            model=MODEL_NAME,
            model_provider="mistralai",
            temperature=0.3,
            api_key=MISTRAL_API_KEY
        )
        logger.info(f"Modele {MODEL_NAME} initialise")
    except Exception as e:
        logger.error(f"Erreur initialisation: {e}")
        state.dernier_message_ia = (
            "Je rencontre une difficulte technique. "
            "Pouvez-vous reformuler votre demande de voyage ?"
        )
        return state
    
    message = state.dernier_message_utilisateur
    
    # DETECTION HORS-SUJET
    if detect_hors_sujet(message):
        logger.info("Message hors-sujet detecte - Reinitialisation")
        state.criteres = {
            "plage": None,
            "montagne": None,
            "ville": None,
            "sport": None,
            "detente": None,
            "acces_handicap": None,
        }
        state.dernier_message_ia = (
            "Je suis votre conseiller voyage et je ne comprends pas votre message. "
            "Je peux vous aider a trouver des sejours : plage, montagne, ville, "
            "sport ou detente. Que recherchez-vous pour vos prochaines vacances ?"
        )
        state.voyages_proposes = []
        return state
    
    # EXTRACTION DES CRITERES
    try:
        logger.info("Extraction des criteres...")
        
        model_with_structure = model.with_structured_output(Criteres)
        prompt = PROMPT_EXTRACTION.format(message=message)
        
        criteres_response = await model_with_structure.ainvoke(prompt)
        
        logger.info(f"  Criteres extraits du LLM: {criteres_response.dict()}")
        
        # LOGIQUE EXCLUSION MUTUELLE (plage vs montagne)
        criteres_extraits = criteres_response.dict()
        
        # Plage et Montagne sont mutuellement exclusifs
        if criteres_extraits.get("plage") == True:
            criteres_extraits["montagne"] = False
        elif criteres_extraits.get("montagne") == True:
            criteres_extraits["plage"] = False
        
        # Sport et Detente sont mutuellement exclusifs
        if criteres_extraits.get("sport") == True:
            criteres_extraits["detente"] = False
        elif criteres_extraits.get("detente") == True:
            criteres_extraits["sport"] = False
        
        logger.info(f"  Criteres apres exclusions: {criteres_extraits}")
        
        # MISE A JOUR INCREMENTALE
        for key, value in criteres_extraits.items():
            if value is not None:
                state.criteres[key] = value
                logger.info(f"  MAJ: {key} -> {value}")
        
        logger.info("Extraction terminee")
        
    except Exception as e:
        logger.error(f"Erreur extraction: {e}")
    
    # VERIFICATION SI CRITERES REMPLIS
    criteres_remplis = any(v is not None for v in state.criteres.values())
    logger.info(f"Criteres remplis: {criteres_remplis}")
    logger.info(f"Criteres actuels: {state.criteres}")
    
    # AUCUN CRITERE -> DEMANDER
    if not criteres_remplis:
        logger.info("Aucun critere -> Demande informations")
        
        try:
            prompt = PROMPT_SANS_CRITERES.format(message=message)
            response = await model.ainvoke(prompt)
            state.dernier_message_ia = response.content
        except Exception as e:
            logger.error(f"Erreur generation: {e}")
            state.dernier_message_ia = (
                "Pour vous aider a trouver le voyage ideal, "
                "pouvez-vous me dire ce que vous recherchez ? "
                "Plage, montagne, ville, sport ou detente ?"
            )
        
        state.voyages_proposes = []
        return state
    
    # AU MOINS 1 CRITERE -> PROPOSER VOYAGES
    logger.info("Recherche UNIQUEMENT dans base VOYAGES (5 voyages)")
    logger.info("AUCUNE recherche web - Base locale uniquement")
    voyages_matches = []
    
    for voyage in VOYAGES:
        if match_voyage(voyage, state.criteres):
            voyages_matches.append(voyage)
            logger.info(f"  Match trouve: {voyage['nom']}")
    
    state.voyages_proposes = voyages_matches
    logger.info(f"{len(voyages_matches)} voyage(s) trouve(s) dans CATALOGUE")
    
    # Generation de la reponse
    try:
        # TOUJOURS lister TOUS les voyages du catalogue
        tous_voyages_str = "LES 5 VOYAGES CATALOGUE COMPLET:\n"
        for v in VOYAGES:
            tous_voyages_str += f"\n- {v['nom']}"
            tous_voyages_str += f"\n  Description: {v['description']}"
            tous_voyages_str += f"\n  Criteres: "
            tous_voyages_str += f"Plage={v['plage']}, Montagne={v['montagne']}, "
            tous_voyages_str += f"Ville={v['ville']}, Sport={v['sport']}, "
            tous_voyages_str += f"Detente={v['detente']}, PMR={v['acces_handicap']}"
            tous_voyages_str += f"\n  Prix: {v['prix']}"
        
        if voyages_matches:
            tous_voyages_str += f"\n\nVOYAGES CORRESPONDANT AUX CRITERES ({len(voyages_matches)}):\n"
            for v in voyages_matches:
                tous_voyages_str += f"- {v['nom']}\n"
        else:
            tous_voyages_str += "\n\nAUCUN voyage ne correspond exactement aux criteres."
        
        prompt = PROMPT_AVEC_VOYAGES.format(
            message=message,
            criteres_str=format_criteres_str(state.criteres),
            voyages_str=tous_voyages_str
        )
        
        response = await model.ainvoke(prompt)
        state.dernier_message_ia = response.content
        logger.info("Reponse generee")
        
    except Exception as e:
        logger.error(f"Erreur generation: {e}")
        
        if voyages_matches:
            noms = [v['nom'] for v in voyages_matches[:2]]
            state.dernier_message_ia = (
                f"Trouve {len(voyages_matches)} voyage(s) pour vous ! "
                f"{', '.join(noms)}. Interet ?"
            )
        else:
            state.dernier_message_ia = (
                "Aucun voyage correspondant exactement. "
                "Pouvez-vous preciser criteres ?"
            )
    
    return state

# CONSTRUCTION DU GRAPHE

def build_graph() -> CompiledStateGraph:
    """Graphe minimal : START -> process_message -> END"""
    logger.info("Construction du graphe...")
    
    workflow = StateGraph(State)
    workflow.add_node("process_message", process_message)
    workflow.set_entry_point("process_message")
    workflow.add_edge("process_message", END)
    
    compiled = workflow.compile()
    logger.info("Graphe compile")
    return compiled

# Point entree pour LangGraph
graph = build_graph()
