#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Agent de voyage – graph.py avec Flux Décisionnel Complet
- Flux séquentiel strict avec 7 étapes
- Ajout du PROMPT_CLARIFICATION (étape 5)
- Gestion propre de tous les cas d'usage
"""
from __future__ import annotations

import os
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from pydantic import BaseModel, Field

from langchain.chat_models import init_chat_model
from langchain.prompts import ChatPromptTemplate
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
# Données – 5 voyages (catalogue fixe)
# -----------------------------------------------------------------------------
VOYAGES: List[Dict[str, Any]] = [
    {
        "id": "LOZ-001",
        "nom": "Randonnée camping en Lozère",
        "description": "Aventure sportive au cœur de la nature sauvage",
        "emoji": "🏕️",
        "plage": False,
        "montagne": True,
        "ville": False,
        "sport": True,
        "detente": False,
        "acces_handicap": False,
        "labels": ["sport", "montagne", "campagne"],
        "prix": "à partir de 650€/semaine",
    },
    {
        "id": "CHAM-SPA",
        "nom": "5 étoiles à Chamonix - Option Spa & Fondue",
        "description": "Luxe et détente au pied du Mont-Blanc",
        "emoji": "🧖",
        "plage": False,
        "montagne": True,
        "ville": False,
        "sport": False,
        "detente": True,
        "acces_handicap": True,
        "labels": ["montagne", "détente"],
        "prix": "sur demande",
    },
    {
        "id": "CHAM-SKI",
        "nom": "5 étoiles à Chamonix - Option Ski",
        "description": "Sport et luxe dans la capitale de l'alpinisme",
        "emoji": "⛷️",
        "plage": False,
        "montagne": True,
        "ville": False,
        "sport": True,
        "detente": False,
        "acces_handicap": False,
        "labels": ["montagne", "sport"],
        "prix": "sur demande",
    },
    {
        "id": "PAL-001",
        "nom": "Palavas de paillotes en paillotes",
        "description": "Farniente urbain sur la Méditerranée",
        "emoji": "🏖️",
        "plage": True,
        "montagne": False,
        "ville": True,
        "sport": False,
        "detente": True,
        "acces_handicap": True,
        "labels": ["plage", "ville", "détente", "paillote"],
        "prix": "800€/semaine",
    },
    {
        "id": "CAMP-LUX",
        "nom": "5 étoiles en rase campagne",
        "description": "Havre de paix luxueux dans la nature",
        "emoji": "🌾",
        "plage": False,
        "montagne": False,
        "ville": False,
        "sport": False,
        "detente": True,
        "acces_handicap": True,
        "labels": ["campagne", "détente"],
        "prix": "sur demande",
    },
]

# -----------------------------------------------------------------------------
# Pydantic – critères (output structuré)
# -----------------------------------------------------------------------------
class Criteres(BaseModel):
    """Critères de voyage extraits du message utilisateur."""
    plage: Optional[bool] = Field(None, description="Destination balnéaire (mer, océan, plage)")
    montagne: Optional[bool] = Field(None, description="Destination montagne (altitude, ski)")
    ville: Optional[bool] = Field(None, description="Destination urbaine (métropole, shopping)")
    sport: Optional[bool] = Field(None, description="Activités sportives (randonnée, ski, vélo)")
    detente: Optional[bool] = Field(None, description="Détente et relaxation (spa, repos)")
    acces_handicap: Optional[bool] = Field(None, description="Accessibilité PMR")

# -----------------------------------------------------------------------------
# State (dataclass)
# -----------------------------------------------------------------------------
@dataclass
class State:
    """État de l'agent (pas d'historique complet, juste dernier échange)."""
    premier_message_utilisateur: str = ""
    dernier_message_utilisateur: str = ""
    dernier_message_ia: str = ""
    criteres: Dict[str, Optional[bool]] = field(
        default_factory=lambda: {
            "plage": None, "montagne": None, "ville": None,
            "sport": None, "detente": None, "acces_handicap": None,
        }
    )
    voyages_proposes: List[Dict[str, Any]] = field(default_factory=list)
    dernier_voyage_id: Optional[str] = None

# ═════════════════════════════════════════════════════════════════════════════
# PROMPTS SYSTÈME - FLUX DÉCISIONNEL COMPLET
# ═════════════════════════════════════════════════════════════════════════════

# ┌─────────────────────────────────────────────────────────────────────────┐
# │ ÉTAPE 1 : PROMPT_SALUTATION                                              │
# │ Cas : "Bonjour", "Salut", "Hello", "Merci"                              │
# └─────────────────────────────────────────────────────────────────────────┘
PROMPT_SALUTATION = ChatPromptTemplate.from_messages([
    ("system", """Tu es un conseiller voyage accueillant et professionnel.
L'utilisateur te salue ou entame la conversation.

TON: Accueillant, professionnel, enthousiaste
CONSIGNES:
- Salutation chaleureuse et personnalisée
- Présentation brève (conseiller voyage)
- Proposition d'aide avec exemples concrets
- 2-3 phrases maximum
- Éviter les formulations trop longues ou robotiques
- Pas d'emojis"""),
    ("user", "Message reçu: {message}")
])

# ┌─────────────────────────────────────────────────────────────────────────┐
# │ ÉTAPE 4 : PROMPT_EXTRACTION_CRITERES                                     │
# │ Cas : Tout message lié au voyage (après checks préliminaires)           │
# └─────────────────────────────────────────────────────────────────────────┘
PROMPT_EXTRACTION_CRITERES = """Tu es un assistant expert en analyse de préférences de voyage.
Ta mission est d'identifier précisément les critères de voyage dans le message utilisateur.

⚠️ IMPORTANT: Si le message n'a AUCUN rapport avec un voyage ou est incompréhensible, 
retourne TOUS les critères à None (message hors contexte).

CRITÈRES À ANALYSER (6 au total):

1. PLAGE 🏖️
   - True si: mer, océan, plage, côte, baignade, bronzer, sable, littoral
   - False si: "pas de plage", "loin de la mer", "à l'intérieur des terres"
   - None si: non mentionné

2. MONTAGNE 🏔️
   - True si: montagne, altitude, sommet, ski, neige, alpes, pyrénées, randonnée alpine
   - False si: "pas de montagne", "plaine", "plat"
   - None si: non mentionné

3. VILLE 🏙️
   - True si: ville, urbain, métropole, capitale, centre-ville, cité, shopping
   - False si: "pas de ville", "campagne", "rural", "nature"
   - None si: non mentionné

4. SPORT ⚽
   - True si: sport, actif, activité physique, ski, randonnée, trek, vélo, surf, escalade
   - False si: "pas de sport", "tranquille", "sans effort", "repos uniquement"
   - None si: non mentionné

5. DETENTE 🧘
   - True si: repos, détente, calme, spa, massage, zen, tranquille, relaxation, bien-être
   - False si: "pas de repos", "actif", "dynamique", "aventure"
   - None si: non mentionné

6. ACCES_HANDICAP ♿
   - True si: PMR, accessibilité, mobilité réduite, fauteuil roulant, handicap, adapté
   - False si: "peu importe l'accessibilité", "sportif extrême"
   - None si: non mentionné

MESSAGE À ANALYSER: "{message}"

RÈGLES D'EXTRACTION:
- D'ABORD vérifier si le message parle de voyage/vacances/séjour/destination
- Si le message est du charabia ou hors sujet → TOUT à None
- Chercher les mots-clés ET le contexte voyage
- Détecter les négations ("pas de", "sans", "sauf")
- Si ambiguïté → None
- NE JAMAIS inventer de critères non mentionnés

Réponds UNIQUEMENT avec le JSON des 6 critères selon le schéma Criteres."""

# ┌─────────────────────────────────────────────────────────────────────────┐
# │ ÉTAPE 5 : PROMPT_CLARIFICATION ⭐ NOUVEAU                                │
# │ Cas : Aucun critère n'a été extrait (message vague ou ambigu)           │
# └─────────────────────────────────────────────────────────────────────────┘
PROMPT_CLARIFICATION = ChatPromptTemplate.from_messages([
    ("system", """Tu es un conseiller voyage patient et pédagogue.
Le message utilisateur semble concerner un voyage mais ne contient pas de préférence identifiable.

TON: Patient, encourageant, aidant
CONSIGNES:
- Rester positif ("Je suis là pour vous aider")
- Demander des précisions avec des exemples concrets
- Proposer 3-4 options pour orienter (mer? montagne? ville? détente?)
- 2-3 phrases maximum
- Éviter de montrer de la frustration ou de l'incompréhension
- Pas d'emojis"""),
    ("user", """Message reçu: {message}

Le message semble concerner un voyage mais manque de précisions sur les préférences.
Aide l'utilisateur à clarifier sa demande.""")
])

# ┌─────────────────────────────────────────────────────────────────────────┐
# │ ÉTAPE 7a : PROMPT_REPONSE_AVEC_VOYAGES                                   │
# │ Cas : Au moins 1 voyage correspond aux critères                         │
# └─────────────────────────────────────────────────────────────────────────┘
PROMPT_REPONSE_AVEC_VOYAGES = ChatPromptTemplate.from_messages([
    ("system", """Tu es un conseiller voyage expert et enthousiaste.
Tu dois créer une réponse engageante et personnalisée.

TON: Professionnel mais chaleureux, enthousiaste sans excès
CONSIGNES:
- 4-6 phrases maximum
- Pas d'emojis (déjà dans les données voyage)
- Reformuler brièvement la demande
- Présenter le voyage avec enthousiasme
- Mettre en avant les points forts correspondant aux critères
- Créer du désir avec des descriptions évocatrices
- Terminer par une question engageante"""),
    ("user", """CONTEXTE:
- Message client: {message}
- Critères identifiés: {criteres_str}

🎯 VOYAGE CORRESPONDANT:
{voyage_str}

Crée une réponse personnalisée qui présente ce voyage.""")
])

# ┌─────────────────────────────────────────────────────────────────────────┐
# │ ÉTAPE 7b : PROMPT_REPONSE_SANS_VOYAGES                                   │
# │ Cas : Aucun voyage ne correspond aux critères                           │
# └─────────────────────────────────────────────────────────────────────────┘
PROMPT_REPONSE_SANS_VOYAGES = ChatPromptTemplate.from_messages([
    ("system", """Tu es un conseiller voyage empathique et solution-oriented.
Aucun voyage ne correspond parfaitement, mais tu restes positif et aidant.

TON: Compréhensif, constructif, optimiste
CONSIGNES:
- 4-5 phrases maximum
- Montrer que tu as compris la demande
- Expliquer brièvement pourquoi aucune offre ne correspond
- Proposer 1-2 alternatives proches avec leurs avantages
- Suggérer une adaptation des critères
- Rester positif et orienté solution
- Pas d'emojis"""),
    ("user", """SITUATION:
- Demande client: {message}
- Critères recherchés: {criteres_details}
- Nombre de voyages en catalogue: {nb_voyages}

CATALOGUE DISPONIBLE:
{voyages_disponibles}

Crée une réponse empathique qui propose des alternatives.""")
])

# ═════════════════════════════════════════════════════════════════════════════
# UTILITAIRES
# ═════════════════════════════════════════════════════════════════════════════

def format_criteres_str(criteres: Dict[str, Optional[bool]]) -> str:
    """Formate les critères pour affichage (OUI/NON uniquement)."""
    lignes = []
    for k, v in criteres.items():
        if v is True:
            lignes.append(f"{k}: OUI")
        elif v is False:
            lignes.append(f"{k}: NON")
    return " | ".join(lignes) if lignes else "Aucun critère explicite"

def format_voyage_str(voyage: Dict[str, Any]) -> str:
    """Formate un voyage pour le prompt."""
    return f"""• {voyage['emoji']} {voyage['nom']}
  Description: {voyage['description']}
  Prix: {voyage['prix']}
  Critères: Plage={voyage['plage']}, Montagne={voyage['montagne']}, Ville={voyage['ville']}, 
            Sport={voyage['sport']}, Détente={voyage['detente']}, PMR={voyage['acces_handicap']}
  Points forts: {', '.join(voyage['labels'])}"""

def format_catalogue_str() -> str:
    """Formate le catalogue complet."""
    return "\n\n".join([
        f"• {v['emoji']} {v['nom']}: {v['description']}"
        for v in VOYAGES
    ])

def detect_salutation(message: str) -> bool:
    """Détecte si le message est une salutation."""
    if not message:
        return False
    
    msg = message.lower().strip()
    
    # Liste des salutations (au lieu d'un set)
    salutations = [
        "bonjour", "bonsoir", "salut", "hello", "hi", "hey", "coucou",
        "merci", "merci beaucoup", "ok merci", "super merci",
        "bonne journée", "bonne soirée"
    ]
    
    # Vérification exacte (message = salutation)
    if msg in salutations:
        return True
    
    # Vérification si le message commence par une salutation courte
    for sal in salutations:
        if msg == sal:  # Match exact
            return True
        # Message commence par salutation + espace/virgule et reste court
        if (msg.startswith(sal + " ") or msg.startswith(sal + ",")) and len(msg.split()) <= 6:
            return True
    
    # Cas spéciaux : juste un mot court très courant
    if len(msg.split()) == 1 and msg in ["bonjour", "bonsoir", "salut", "hello", "hi", "hey", "coucou"]:
        return True
    
    return False
    # Salutation courte ou commence par une salutation
    if msg in salutations:
        return True
    if any(msg.startswith(s + " ") or msg.startswith(s + ",") for s in salutations):
        return len(msg.split()) <= 5  # Salutation courte
    return False

def detect_question_pratique(message: str) -> Optional[str]:
    """Détecte les questions pratiques (prix, dates, réservation)."""
    msg = message.lower()
    if any(w in msg for w in ["prix", "cout", "coût", "tarif", "€", "euro"]):
        return "prix"
    if any(w in msg for w in ["quand", "date", "période", "moment", "saison"]):
        return "dates"
    if any(w in msg for w in ["réserv", "réserv", "book", "command", "contact"]):
        return "reservation"
    return None

def detect_hors_sujet(message: str) -> bool:
    """Détection légère de charabia / hors-sujet."""
    msg = (message or "").lower().strip()
    if len(msg) < 2:
        return True
    
    # Vérification densité voyelles
    voyelles = sum(1 for c in msg if c in "aeiouyàâäéèêëïîôùûüÿæœ")
    if len(msg) > 5 and voyelles < max(1, int(len(msg) * 0.15)):
        return True
    
    # Vérification répétitions
    for i in range(len(msg) - 3):
        if len(set(msg[i:i + 4])) <= 2:
            return True
    
    return False

def match_voyage(voyage: Dict[str, Any], criteres: Dict[str, Optional[bool]]) -> bool:
    """Vérifie si un voyage correspond aux critères."""
    for critere, valeur in criteres.items():
        if valeur is None:
            continue
        if critere in voyage and voyage[critere] != valeur:
            return False
    return True

# ═════════════════════════════════════════════════════════════════════════════
# NŒUD PRINCIPAL - FLUX DÉCISIONNEL COMPLET
# ═════════════════════════════════════════════════════════════════════════════

async def process_message(state: State) -> State:
    """
    Nœud unique qui traite chaque message utilisateur.
    
    FLUX DÉCISIONNEL (7 ÉTAPES):
    ═══════════════════════════════════════════════════════════════════
    1. Salutation ?         → PROMPT_SALUTATION        → FIN
    2. Question pratique ?  → Code Python direct       → FIN
    3. Hors contexte ?      → Réponse directe          → FIN
    4. Extraction critères  → PROMPT_EXTRACTION_CRITERES
    5. Aucun critère ?      → PROMPT_CLARIFICATION     → FIN
    6. Matching voyages
    7a. Voyages trouvés ?   → PROMPT_REPONSE_AVEC_VOYAGES
    7b. Aucun voyage ?      → PROMPT_REPONSE_SANS_VOYAGES
    ═══════════════════════════════════════════════════════════════════
    """
    message = state.dernier_message_utilisateur or ""
    
    message = message.strip()
    message = ' '.join(message.split())  # Normalise les espaces
    state.dernier_message_utilisateur = message
    
    # Initialisation modèle
    model = init_chat_model(
        model=MODEL_NAME,
        model_provider="mistralai",
        temperature=0.2,
        api_key=MISTRAL_API_KEY,
    )
    
    logger.info("="*70)
    logger.info(f"📨 MESSAGE REÇU: '{message}'")
    logger.info("="*70)
    
    # ┌─────────────────────────────────────────────────────────────────────┐
    # │ ÉTAPE 1 : DÉTECTION SALUTATION                                       │
    # └─────────────────────────────────────────────────────────────────────┘
    if detect_salutation(message):
        logger.info("👋 ÉTAPE 1: SALUTATION DÉTECTÉE → PROMPT_SALUTATION → FIN")
        try:
            prompt = PROMPT_SALUTATION.format_messages(message=message)
            response = await model.ainvoke(prompt)
            state.dernier_message_ia = response.content.replace("\n", " ").strip()
            state.voyages_proposes = []
            logger.info(f"✅ Réponse: {state.dernier_message_ia[:100]}...")
            return state
        except Exception as e:
            logger.error(f"❌ Erreur salutation: {e}")
            state.dernier_message_ia = (
                "Bonjour ! Je suis votre conseiller voyage. "
                "Dites-moi ce qui vous ferait plaisir : "
                "plage, montagne, ville, sport, détente ?"
            )
            return state
    
    # ┌─────────────────────────────────────────────────────────────────────┐
    # │ ÉTAPE 2 : DÉTECTION QUESTION PRATIQUE                                │
    # └─────────────────────────────────────────────────────────────────────┘
    question_type = detect_question_pratique(message)
    if question_type:
        logger.info(f"❓ ÉTAPE 2: QUESTION PRATIQUE ({question_type}) → CODE PYTHON → FIN")
        if question_type == "prix":
            state.dernier_message_ia = (
                "Nos tarifs varient selon les destinations : "
                "Palavas à partir de 800€/semaine, "
                "Lozère à partir de 650€/semaine, "
                "autres destinations sur demande. "
                "Pour un devis personnalisé, contactez-nous au 0800 VOYAGE."
            )
        elif question_type == "dates":
            state.dernier_message_ia = (
                "Nos destinations sont disponibles toute l'année. "
                "Pour la plage : mai à septembre. "
                "Pour la montagne : décembre à avril (ski), juin à septembre (randonnée). "
                "Quelle période vous intéresse ?"
            )
        else:  # reservation
            state.dernier_message_ia = (
                "Pour réserver : 📞 0800 VOYAGE (appel gratuit) "
                "ou 📧 reservations@votreagence.com. "
                "Notre équipe vous répondra sous 24h."
            )
        state.voyages_proposes = []
        logger.info(f"✅ Réponse: {state.dernier_message_ia[:100]}...")
        return state
    
    # ┌─────────────────────────────────────────────────────────────────────┐
    # │ ÉTAPE 3 : DÉTECTION HORS CONTEXTE                                    │
    # └─────────────────────────────────────────────────────────────────────┘
    if detect_hors_sujet(message):
        logger.info("⚠️ ÉTAPE 3: HORS CONTEXTE → RÉPONSE DIRECTE → FIN")
        state.criteres = {k: None for k in state.criteres}
        state.voyages_proposes = []
        state.dernier_message_ia = (
            "Je n'ai pas bien compris votre message. "
            "Pour vous aider à trouver le voyage idéal, "
            "pouvez-vous préciser vos critères : "
            "plage, montagne, ville, sport, détente, accessibilité PMR ?"
        )
        logger.info(f"✅ Réponse: {state.dernier_message_ia[:100]}...")
        return state
    
    # ┌─────────────────────────────────────────────────────────────────────┐
    # │ ÉTAPE 4 : EXTRACTION DES CRITÈRES                                    │
    # └─────────────────────────────────────────────────────────────────────┘
    logger.info("🔍 ÉTAPE 4: EXTRACTION CRITÈRES → PROMPT_EXTRACTION_CRITERES")
    
    # Reset critères à chaque tour
    state.criteres = {
        "plage": None, "montagne": None, "ville": None,
        "sport": None, "detente": None, "acces_handicap": None
    }
    
    try:
        model_struct = model.with_structured_output(Criteres)
        criteres_obj = await model_struct.ainvoke(
            PROMPT_EXTRACTION_CRITERES.format(message=message)
        )
        extraits: Dict[str, Optional[bool]] = criteres_obj.dict()
        logger.info(f"   📊 Critères bruts extraits: {extraits}")
    except Exception as e:
        logger.error(f"   ❌ Erreur extraction: {e}")
        extraits = {k: None for k in state.criteres.keys()}
    
    # Heuristiques post-extraction
    low = message.lower()
    if "montpellier" in low:
        extraits["plage"] = True
        extraits["ville"] = True
        extraits["montagne"] = False
        logger.info("   🎯 Heuristique Montpellier appliquée")
    
    # Exclusivité stricte des thèmes
    themes = ["plage", "montagne", "ville"]
    pos = [t for t in themes if extraits.get(t) is True]
    if len(pos) == 1:
        for t in themes:
            if t != pos[0]:
                extraits[t] = False
        logger.info(f"   🎯 Exclusivité appliquée: {pos[0]}")
    
    # Exclusivité légère sport/détente
    if extraits.get("sport") is True and extraits.get("detente") is not True:
        extraits["detente"] = False
    if extraits.get("detente") is True and extraits.get("sport") is not True:
        extraits["sport"] = False
    
    # Mise à jour état
    for k, v in extraits.items():
        if k in state.criteres and v is not None:
            state.criteres[k] = v
    
    logger.info(f"   ✅ Critères finaux: {state.criteres}")
    
    # ┌─────────────────────────────────────────────────────────────────────┐
    # │ ÉTAPE 5 : VÉRIFICATION CRITÈRES → CLARIFICATION SI AUCUN            │
    # └─────────────────────────────────────────────────────────────────────┘
    if all(v is None for v in state.criteres.values()):
        logger.info("⚠️ ÉTAPE 5: AUCUN CRITÈRE → PROMPT_CLARIFICATION → FIN")
        try:
            prompt = PROMPT_CLARIFICATION.format_messages(message=message)
            response = await model.ainvoke(prompt)
            state.dernier_message_ia = response.content.replace("\n", " ").strip()
        except Exception as e:
            logger.error(f"   ❌ Erreur clarification: {e}")
            state.dernier_message_ia = (
                "Je n'ai pas identifié de critères de voyage dans votre message. "
                "Pouvez-vous préciser ce qui vous ferait plaisir : "
                "plage, montagne, ville, sport, détente ?"
            )
        state.voyages_proposes = []
        state.dernier_voyage_id = None
        logger.info(f"✅ Réponse: {state.dernier_message_ia[:100]}...")
        return state
    
    # ┌─────────────────────────────────────────────────────────────────────┐
    # │ ÉTAPE 6 : MATCHING VOYAGES                                           │
    # └─────────────────────────────────────────────────────────────────────┘
    logger.info("🔎 ÉTAPE 6: MATCHING VOYAGES")
    matches = []
    for voyage in VOYAGES:
        result = match_voyage(voyage, state.criteres)
        status = "✅" if result else "❌"
        logger.info(f"   {status} {voyage['nom']}")
        if result:
            matches.append(voyage)
    
    top = matches[0] if matches else None
    state.voyages_proposes = [top] if top else []
    state.dernier_voyage_id = top["id"] if top else None
    logger.info(f"   📊 Résultat: {len(matches)} voyage(s) correspondant(s)")
    
    # ┌─────────────────────────────────────────────────────────────────────┐
    # │ ÉTAPE 7a : RÉPONSE AVEC VOYAGES                                      │
    # └─────────────────────────────────────────────────────────────────────┘
    if top:
        logger.info(f"🎉 ÉTAPE 7a: VOYAGE TROUVÉ → PROMPT_REPONSE_AVEC_VOYAGES")
        logger.info(f"   🎯 Voyage proposé: {top['nom']}")
        try:
            prompt = PROMPT_REPONSE_AVEC_VOYAGES.format_messages(
                message=message,
                criteres_str=format_criteres_str(state.criteres),
                voyage_str=format_voyage_str(top)
            )
            response = await model.ainvoke(prompt)
            out = response.content.strip()
            
            # Ajouter question d'engagement si manquante
            if "Souhaitez-vous" not in out and "?" not in out:
                out += " Souhaitez-vous plus de détails ?"
            
            state.dernier_message_ia = out.replace("\n", " ")
            logger.info(f"✅ Réponse: {state.dernier_message_ia[:100]}...")
        except Exception as e:
            logger.error(f"   ❌ Erreur génération réponse: {e}")
            state.dernier_message_ia = (
                f"Je vous propose {top['emoji']} « {top['nom']} » : {top['description']}. "
                f"Prix : {top['prix']}. Souhaitez-vous plus de détails ?"
            )
    
    # ┌─────────────────────────────────────────────────────────────────────┐
    # │ ÉTAPE 7b : RÉPONSE SANS VOYAGES                                      │
    # └─────────────────────────────────────────────────────────────────────┘
    else:
        logger.info("⚠️ ÉTAPE 7b: AUCUN VOYAGE → PROMPT_REPONSE_SANS_VOYAGES")
        try:
            prompt = PROMPT_REPONSE_SANS_VOYAGES.format_messages(
                message=message,
                criteres_details=format_criteres_str(state.criteres),
                nb_voyages=len(VOYAGES),
                voyages_disponibles=format_catalogue_str()
            )
            response = await model.ainvoke(prompt)
            state.dernier_message_ia = response.content.replace("\n", " ").strip()
            logger.info(f"✅ Réponse: {state.dernier_message_ia[:100]}...")
        except Exception as e:
            logger.error(f"   ❌ Erreur génération sans voyage: {e}")
            state.dernier_message_ia = (
                "Aucun voyage ne correspond exactement à vos critères. "
                "Souhaitez-vous assouplir certaines préférences ?"
            )
    
    logger.info("="*70)
    return state

# ═════════════════════════════════════════════════════════════════════════════
# CONSTRUCTION DU GRAPHE
# ═════════════════════════════════════════════════════════════════════════════

def build_graph() -> CompiledStateGraph:
    """Construit le graphe LangGraph avec un seul nœud."""
    wf = StateGraph(State)
    wf.add_node("process_message", process_message)
    wf.set_entry_point("process_message")
    wf.add_edge("process_message", END)
    return wf.compile()

# VARIABLE ATTENDUE PAR `langgraph dev`
graph = build_graph()

if __name__ == "__main__":
    print("="*70)
    print("✅ AGENT DE VOYAGE - FLUX DÉCISIONNEL COMPLET")
    print("="*70)
    print(f"📊 Modèle: {MODEL_NAME}")
    print(f"🗺️  Voyages disponibles: {len(VOYAGES)}")
    print(f"\n📋 FLUX DÉCISIONNEL (7 ÉTAPES):")
    print("   1. Salutation        → PROMPT_SALUTATION")
    print("   2. Question pratique → Code Python direct")
    print("   3. Hors contexte     → Réponse directe")
    print("   4. Extraction        → PROMPT_EXTRACTION_CRITERES")
    print("   5. Aucun critère     → PROMPT_CLARIFICATION ⭐ NOUVEAU")
    print("   6. Matching voyages")
    print("   7a. Voyages trouvés  → PROMPT_REPONSE_AVEC_VOYAGES")
    print("   7b. Aucun voyage     → PROMPT_REPONSE_SANS_VOYAGES")
    print("\n🚀 Utiliser: langgraph dev")
    print("="*70)