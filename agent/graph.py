#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
================================================================================
                    AGENT DE VOYAGE INTELLIGENT - LANGGRAPH
================================================================================
Description:
    Agent conversationnel spécialisé dans la recommandation de voyages.
    Utilise Mistral AI pour l'extraction de critères et la génération de réponses.
    
Architecture:
    - 1 nœud unique de traitement (process_message)
    - 6 critères de recherche (plage, montagne, ville, sport, détente, PMR)
    - 5 voyages prédéfinis en base
    - Structured output avec Pydantic pour l'extraction robuste
    
Utilisation:
    1. Créer un fichier langgraph.json:
       {
         "dependencies": ["."],
         "graphs": {
           "agent": "./agent/graph.py:graph"
         },
         "env": ".env",
         "image_distro": "wolfi"
       }
    
    2. Créer un fichier .env avec:
       MISTRAL_API_KEY=votre_clé_api_mistral
    
    3. Lancer avec:
       langgraph dev
       
Auteur: Agent de Voyage v2.0
Date: 2024
License: MIT
================================================================================
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, List, Annotated
import os
import logging
from datetime import datetime
from dotenv import load_dotenv

from langchain.chat_models import init_chat_model
from pydantic import BaseModel, Field
from langgraph.graph import START, StateGraph, END
from langgraph.graph.state import CompiledStateGraph

# ================================================================================
# CONFIGURATION ET INITIALISATION
# ================================================================================

# Chargement des variables d'environnement
load_dotenv()

# Configuration du logging pour debug
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration du modèle Mistral
MODEL_NAME = os.getenv("MISTRAL_MODEL", "mistral-small-latest")
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")

# Vérification de la clé API
if not MISTRAL_API_KEY:
    logger.warning("⚠️ MISTRAL_API_KEY non définie dans .env - L'agent ne pourra pas fonctionner")

# ================================================================================
# PROMPTS SYSTÈME - CŒUR DE L'INTELLIGENCE
# ================================================================================

PROMPT_EXTRACTION_CRITERES = """Tu es un assistant expert en analyse de préférences de voyage.
Ta mission est d'identifier précisément les critères de voyage dans le message utilisateur.

⚠️ IMPORTANT: Si le message n'a AUCUN rapport avec un voyage ou est incompréhensible, 
retourne TOUS les critères à None (message hors contexte).

CRITÈRES À ANALYSER (6 au total):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
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
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

MESSAGE À ANALYSER: "{message}"

RÈGLES D'EXTRACTION:
• D'ABORD vérifier si le message parle de voyage/vacances/séjour/destination
• Si le message est du charabia ou hors sujet → TOUT à None
• Chercher les mots-clés ET le contexte voyage
• Détecter les négations ("pas de", "sans", "sauf")
• Si ambiguïté → None
• NE JAMAIS inventer de critères non mentionnés

Réponds UNIQUEMENT avec le JSON des 6 critères."""

# --------------------------------------------------------------------------------

PROMPT_REPONSE_AVEC_VOYAGES = """Tu es un conseiller voyage expert et enthousiaste.
Tu dois créer une réponse engageante et personnalisée.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CONTEXTE:
• Message client: "{message}"
• Critères identifiés: {criteres_str}

🎯 VOYAGES CORRESPONDANTS:
{voyages_str}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

STRUCTURE DE TA RÉPONSE (4-6 phrases max):
1. Reformule brièvement la demande du client
2. Présente les voyages avec enthousiasme:
   - Utilise les emojis des voyages
   - Mets en avant les points forts correspondant aux critères
   - Crée du désir avec des descriptions évocatrices
3. Si plusieurs options: aide à choisir en soulignant les différences
4. Termine par une question engageante ou une proposition d'aide

TON: Professionnel mais chaleureux, enthousiaste sans excès."""

# --------------------------------------------------------------------------------

PROMPT_REPONSE_SANS_VOYAGES = """Tu es un conseiller voyage empathique et solution-oriented.
Aucun voyage ne correspond parfaitement, mais tu restes positif et aidant.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SITUATION:
• Demande client: "{message}"
• Critères recherchés: {criteres_details}
• Nombre de voyages en catalogue: {nb_voyages}

CATALOGUE DISPONIBLE:
{voyages_disponibles}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

STRUCTURE DE TA RÉPONSE (4-5 phrases max):
1. Montre que tu as compris la demande (reformulation empathique)
2. Explique brièvement pourquoi aucune offre ne correspond parfaitement
3. Propose 1-2 alternatives proches avec leurs avantages
4. Suggère une adaptation des critères ("Si vous êtes flexible sur...")
5. Reste positif et orienté solution

TON: Compréhensif, constructif, optimiste."""

# --------------------------------------------------------------------------------

PROMPT_SALUTATION = """Tu es un conseiller voyage accueillant et professionnel.
L'utilisateur te salue ou entame la conversation.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Message reçu: "{message}"
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

RÉPONSE EN 2-3 PHRASES:
1. Salutation chaleureuse et personnalisée
2. Présentation brève (conseiller voyage)
3. Proposition d'aide avec exemples (plage, montagne, détente...)

TON: Accueillant, professionnel, enthousiaste
ÉVITER: Les formulations trop longues ou robotiques."""

# --------------------------------------------------------------------------------

PROMPT_INFORMATION_PRATIQUE = """Tu es un conseiller voyage qui répond à une question pratique.
Le client a déjà reçu des propositions de voyage et demande maintenant des informations complémentaires.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Question du client: "{message}"
Type de question identifié: {question_type}

VOYAGES PROPOSÉS PRÉCÉDEMMENT:
{voyages_info}

INFORMATIONS DISPONIBLES:
- Prix indicatifs inclus dans chaque voyage
- Meilleures périodes pour voyager
- Points forts de chaque destination
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

INSTRUCTIONS DE RÉPONSE:
1. Répondre directement à la question posée
2. Si c'est sur les prix: lister les tarifs indicatifs
3. Si c'est sur les dates: donner les meilleures périodes
4. Toujours mentionner de contacter l'agence pour finaliser
5. Rester dans le contexte des voyages déjà proposés

CONTACT AGENCE:
📞 0800 VOYAGE (appel gratuit)
📧 reservations@votreagence.com
🌐 www.votreagence.com

TON: Informatif, précis, serviable"""

# --------------------------------------------------------------------------------

PROMPT_INCOMPREHENSIBLE = """Tu es un conseiller voyage patient et pédagogue.
Le message utilisateur n'est pas clair ou ne contient pas de préférence identifiable.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Message reçu: "{message}"
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

RÉPONSE EN 2-3 PHRASES:
1. Reste positif ("Je suis là pour vous aider")
2. Demande des précisions avec des exemples
3. Propose 3-4 options pour orienter (mer? montagne? ville? détente?)

TON: Patient, encourageant, aidant
ÉVITER: Montrer de la frustration ou de l'incompréhension."""

# ================================================================================
# MODÈLE DE DONNÉES - ÉTAT DE L'AGENT
# ================================================================================

@dataclass
class State:
    """
    État complet de l'agent de voyage pour LangGraph.
    
    Attributes:
        dernier_message_utilisateur: Message actuel de l'utilisateur
        dernier_message_ia: Dernière réponse générée par l'IA
        injection: Flag de détection d'injection de prompt (sécurité)
        erreur_ia: Flag indiquant une erreur lors du traitement
        done: Flag indiquant la fin du traitement du message
        criteres: Dictionnaire des 6 critères (True/False/None)
        voyages_proposes: Liste des voyages correspondant aux critères
        derniers_voyages_proposes: Sauvegarde des derniers voyages proposés
        metadata: Métadonnées supplémentaires (timestamps, version, etc.)
    """
    
    # Messages
    dernier_message_utilisateur: str = ""
    dernier_message_ia: str = ""
    
    # Flags de contrôle
    injection: bool = False
    erreur_ia: bool = False
    done: bool = False
    
    # Critères de recherche (6 critères booléens)
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
    
    # Résultats de recherche
    voyages_proposes: List[Dict[str, Any]] = field(default_factory=list)
    derniers_voyages_proposes: List[Dict[str, Any]] = field(default_factory=list)  # Historique
    
    # Métadonnées pour tracking
    metadata: Dict[str, Any] = field(
        default_factory=lambda: {
            "timestamp": datetime.now().isoformat(),
            "version": "2.0",
            "model": MODEL_NAME
        }
    )

# ================================================================================
# BASE DE DONNÉES DES VOYAGES (5 voyages prédéfinis)
# ================================================================================

VOYAGES: List[Dict[str, Any]] = [
    {
        "id": "LOZ-001",
        "nom": "🥾 Randonnée camping en Lozère",
        "description": "Aventure sportive au cœur de la nature sauvage des Cévennes",
        "plage": False,
        "montagne": True,
        "ville": False,
        "sport": True,
        "detente": False,
        "acces_handicap": False,
        "points_forts": [
            "Randonnées guidées quotidiennes",
            "Nuits en bivouac sous les étoiles", 
            "Faune et flore exceptionnelles"
        ],
        "prix_indicatif": "450€/semaine",
        "meilleure_periode": "Mai-Septembre"
    },
    {
        "id": "CHAM-SPA",
        "nom": "⭐ 5 étoiles à Chamonix - Option Spa & Fondue",
        "description": "Luxe absolu et détente au pied du Mont-Blanc",
        "plage": False,
        "montagne": True,
        "ville": False,
        "sport": False,
        "detente": True,
        "acces_handicap": True,
        "points_forts": [
            "Spa panoramique avec vue sur les glaciers",
            "Restaurant gastronomique savoyard",
            "Chambres adaptées PMR avec balcon"
        ],
        "prix_indicatif": "2500€/semaine",
        "meilleure_periode": "Toute l'année"
    },
    {
        "id": "CHAM-SKI",
        "nom": "🎿 5 étoiles à Chamonix - Option Ski",
        "description": "Sport et luxe dans la capitale mondiale de l'alpinisme",
        "plage": False,
        "montagne": True,
        "ville": False,
        "sport": True,
        "detente": False,
        "acces_handicap": True,
        "points_forts": [
            "Accès direct aux pistes mythiques",
            "Matériel haut de gamme inclus",
            "Cours avec moniteurs ESF"
        ],
        "prix_indicatif": "3000€/semaine",
        "meilleure_periode": "Décembre-Avril"
    },
    {
        "id": "PAL-001",
        "nom": "🏖️ Palavas de paillotes en paillotes",
        "description": "Farniente urbain sur la Méditerranée avec ambiance festive",
        "plage": True,
        "montagne": False,
        "ville": True,
        "sport": False,
        "detente": True,
        "acces_handicap": True,
        "points_forts": [
            "Plages privées accessibles PMR",
            "Restaurants de fruits de mer",
            "Animations nocturnes en bord de mer"
        ],
        "prix_indicatif": "800€/semaine",
        "meilleure_periode": "Mai-Octobre"
    },
    {
        "id": "CAMP-LUX",
        "nom": "🌿 5 étoiles en rase campagne",
        "description": "Havre de paix luxueux dans la nature préservée",
        "plage": False,
        "montagne": False,
        "ville": False,
        "sport": False,
        "detente": True,
        "acces_handicap": False,
        "points_forts": [
            "Calme absolu et déconnexion totale",
            "Piscine naturelle et jardins",
            "Cuisine bio du potager"
        ],
        "prix_indicatif": "1800€/semaine",
        "meilleure_periode": "Avril-Octobre"
    },
]

# ================================================================================
# SCHÉMA PYDANTIC POUR STRUCTURED OUTPUT
# ================================================================================

class Criteres(BaseModel):
    """
    Schéma de validation Pydantic pour l'extraction structurée des critères.
    Utilisé par Mistral AI pour garantir un format JSON valide.
    """
    plage: Optional[bool] = Field(None, description="Préférence plage/mer")
    montagne: Optional[bool] = Field(None, description="Préférence montagne")
    ville: Optional[bool] = Field(None, description="Préférence ville/urbain")
    sport: Optional[bool] = Field(None, description="Préférence activités sportives")
    detente: Optional[bool] = Field(None, description="Préférence repos/détente")
    acces_handicap: Optional[bool] = Field(None, description="Besoin accessibilité PMR")

# ================================================================================
# FONCTIONS UTILITAIRES
# ================================================================================

def detect_salutation(message: str) -> bool:
    """
    Détecte si le message est une salutation ou formule de politesse.
    
    Args:
        message: Message utilisateur à analyser
        
    Returns:
        True si c'est une salutation, False sinon
    """
    salutations = [
        "bonjour", "salut", "hello", "bonsoir", "coucou", "hey",
        "bonne journée", "au revoir", "bye", "à bientôt", "merci",
        "s'il vous plaît", "svp", "stp"
    ]
    message_lower = message.lower().strip()
    
    # Vérifier si c'est une salutation courte
    if len(message_lower) < 30:
        for sal in salutations:
            if sal in message_lower:
                return True
    return False

def detect_question_info(message: str) -> str:
    """
    Détecte si le message est une question sur des informations pratiques.
    
    Args:
        message: Message utilisateur à analyser
        
    Returns:
        Type de question détecté ('prix', 'date', 'reservation', 'info', '') ou vide si pas une question
    """
    message_lower = message.lower().strip()
    
    # Questions sur les prix (plus de variantes)
    if any(mot in message_lower for mot in [
        'prix', 'coût', 'cout', 'tarif', 'budget', 'combien', 
        'cher', '€', 'euro', 'payer', 'coute', 'valeur',
        'quel prix', 'quels prix', 'le prix', 'les prix'
    ]):
        return 'prix'
    
    # Questions sur les dates/périodes
    if any(mot in message_lower for mot in [
        'quand', 'date', 'période', 'saison', 'mois', 
        'disponible', 'disponibilité', 'partir quand'
    ]):
        return 'date'
    
    # Questions sur la réservation
    if any(mot in message_lower for mot in [
        'réserver', 'réservation', 'comment faire', 'procédure', 
        'inscription', 'booking', 'commander'
    ]):
        return 'reservation'
    
    # Questions générales d'information
    if any(mot in message_lower for mot in [
        'info', 'information', 'détail', 'précision', 
        'savoir plus', 'renseigner', 'renseignement'
    ]):
        return 'info'
    
    return ''
    
def detect_hors_contexte(message: str) -> bool:
    """
    Détecte si le message est hors contexte voyage ou incompréhensible.
    
    Args:
        message: Message utilisateur à analyser
        
    Returns:
        True si le message semble hors contexte ou du charabia
    """
    message_lower = message.lower().strip()
    
    # Vérifier si c'est du charabia (peu de voyelles, caractères répétés)
    voyelles = sum(1 for c in message_lower if c in 'aeiouy')
    if len(message_lower) > 5 and voyelles < len(message_lower) * 0.2:
        return True
    
    # Vérifier les répétitions excessives
    for i in range(len(message_lower) - 2):
        if message_lower[i] == message_lower[i+1] == message_lower[i+2]:
            if message_lower[i] not in 'elo':  # Exceptions pour "elle", "ooo"
                return True
    
    # Mots-clés de voyage
    mots_voyage = [
        "voyage", "vacances", "séjour", "partir", "destination", "week-end", "weekend",
        "plage", "mer", "montagne", "ville", "campagne", "nature", "ski", "sport",
        "détente", "repos", "spa", "hotel", "hôtel", "camping", "randonnée",
        "soleil", "découvrir", "visiter", "explorer", "tourisme", "escapade"
    ]
    
    # Si le message est très court et ne contient aucun mot voyage
    if len(message_lower) < 15 and not any(mot in message_lower for mot in mots_voyage):
        # Vérifier si ça ressemble à du texte normal
        if not any(c.isalpha() for c in message_lower):
            return True
        # Si que des consonnes ou presque
        if len(message_lower) > 3 and voyelles < 2:
            return True
    
    return False

def format_criteres_str(criteres: Dict[str, Optional[bool]]) -> str:
    """
    Formate les critères pour un affichage élégant avec emojis.
    
    Args:
        criteres: Dictionnaire des critères
        
    Returns:
        Chaîne formatée pour l'affichage
    """
    emojis = {
        "plage": "🏖️", "montagne": "🏔️", "ville": "🏙️",
        "sport": "⚽", "detente": "🧘", "acces_handicap": "♿"
    }
    
    actifs = []
    exclus = []
    
    for key, value in criteres.items():
        if value is True:
            actifs.append(f"{emojis.get(key, '')} {key.replace('_', ' ').title()}")
        elif value is False:
            exclus.append(f"Pas de {key.replace('_', ' ')}")
    
    parts = []
    if actifs:
        parts.append(f"✅ Recherché: {', '.join(actifs)}")
    if exclus:
        parts.append(f"❌ À éviter: {', '.join(exclus)}")
    
    return "\n".join(parts) if parts else "Aucun critère spécifique"

def match_voyage(voyage: Dict[str, Any], criteres: Dict[str, Optional[bool]]) -> bool:
    """
    Vérifie si un voyage correspond aux critères utilisateur.
    
    Args:
        voyage: Dictionnaire représentant un voyage
        criteres: Dictionnaire des critères utilisateur
        
    Returns:
        True si le voyage correspond à TOUS les critères exprimés
    """
    for critere, valeur_souhaitee in criteres.items():
        if valeur_souhaitee is None:
            continue  # Critère non exprimé, on l'ignore
            
        if critere in voyage:
            valeur_voyage = voyage[critere]
            if isinstance(valeur_souhaitee, bool) and valeur_voyage != valeur_souhaitee:
                return False  # Le voyage ne correspond pas
                
    return True

# ================================================================================
# NŒUD PRINCIPAL - TRAITEMENT DU MESSAGE
# ================================================================================

async def process_message(state: State) -> State:
    """
    Nœud unique de traitement des messages.
    
    Workflow complet:
    1. Initialisation du modèle Mistral
    2. Détection de salutation
    3. Extraction des critères via structured output
    4. Recherche des voyages correspondants
    5. Génération de la réponse personnalisée
    
    Args:
        state: État actuel de l'agent
        
    Returns:
        État mis à jour après traitement
    """
    
    logger.info(f"📥 Traitement du message: {state.dernier_message_utilisateur[:50]}...")
    
    # ------------------------------------------------------------------------
    # PHASE 1: INITIALISATION DU MODÈLE
    # ------------------------------------------------------------------------
    try:
        model = init_chat_model(
            model=MODEL_NAME,
            model_provider="mistralai",
            temperature=0.3,  # Température basse pour extraction précise
            api_key=MISTRAL_API_KEY
        )
        logger.info(f"✅ Modèle {MODEL_NAME} initialisé")
    except Exception as e:
        logger.error(f"❌ Erreur initialisation modèle: {e}")
        state.erreur_ia = True
        state.dernier_message_ia = "Désolé, je rencontre un problème technique. Veuillez vérifier la configuration."
        state.done = True
        return state
    
    message = state.dernier_message_utilisateur
    
    # ------------------------------------------------------------------------
    # PHASE 2: DÉTECTION DE SALUTATION
    # ------------------------------------------------------------------------
    if detect_salutation(message):
        logger.info("👋 Salutation détectée")
        try:
            response = await model.ainvoke(PROMPT_SALUTATION.format(message=message))
            state.dernier_message_ia = response.content
        except:
            state.dernier_message_ia = (
                "Bonjour ! 👋 Je suis votre conseiller voyage personnel. "
                "Je peux vous aider à trouver le séjour idéal : plage, montagne, "
                "détente ou sport... Que recherchez-vous ?"
            )
        state.done = True
        return state
    
    # ------------------------------------------------------------------------
    # PHASE 2.3: DÉTECTION QUESTION INFORMATION (prix, dates, réservation)
    # ------------------------------------------------------------------------
    question_type = detect_question_info(message)
    if question_type:
        logger.info(f"💬 Question d'information détectée: {question_type}")
        
        # Utiliser les derniers voyages proposés s'ils existent
        voyages_a_considerer = state.voyages_proposes or state.derniers_voyages_proposes
        
        # Si toujours pas de voyages mais des critères actifs, faire une recherche
        if not voyages_a_considerer and any(v is not None for v in state.criteres.values()):
            logger.info("Recherche basée sur les critères existants...")
            voyages_matches = []
            for voyage in VOYAGES:
                if match_voyage(voyage, state.criteres):
                    voyages_matches.append(voyage)
            voyages_a_considerer = voyages_matches
        
        # Si on a des voyages à présenter
        if voyages_a_considerer:
            if question_type == 'prix':
                prix_info = []
                for v in voyages_a_considerer[:3]:  # Limiter à 3 pour ne pas surcharger
                    prix = v.get('prix_indicatif', 'Prix sur demande')
                    prix_info.append(f"• {v['nom']} : **{prix}**")
                
                state.dernier_message_ia = (
                    f"Voici les tarifs indicatifs de nos voyages :\n\n"
                    f"{chr(10).join(prix_info)}\n\n"
                    f"💡 Prix par personne, base double. Variables selon saison et options.\n"
                    f"📞 Pour un devis précis : 0800 VOYAGE (gratuit)\n"
                    f"📧 Email : reservations@votreagence.com"
                )
            elif question_type == 'date':
                dates_info = []
                for v in voyages_a_considerer[:3]:
                    periode = v.get('meilleure_periode', 'Toute l\'année')
                    dates_info.append(f"• {v['nom']} : **{periode}**")
                
                state.dernier_message_ia = (
                    f"Les meilleures périodes pour voyager :\n\n"
                    f"{chr(10).join(dates_info)}\n\n"
                    f"📅 Ces périodes offrent les meilleures conditions météo et tarifaires.\n"
                    f"Contactez-nous pour vérifier les disponibilités sur vos dates !"
                )
            elif question_type == 'reservation':
                state.dernier_message_ia = (
                    "**Réserver votre voyage, c'est simple !** ✈️\n\n"
                    "3 moyens de nous contacter :\n"
                    "📞 **0800 VOYAGE** (appel gratuit)\n"
                    "📧 **reservations@votreagence.com**\n"
                    "🌐 **www.votreagence.com**\n\n"
                    "Notre équipe s'occupe de tout :\n"
                    "✓ Vérification des disponibilités\n"
                    "✓ Personnalisation de votre séjour\n"
                    "✓ Réservation et paiement sécurisé\n\n"
                    "Quel voyage vous tente le plus ?"
                )
            else:
                state.dernier_message_ia = (
                    "Pour tous les détails sur nos voyages, notre équipe est là pour vous !\n\n"
                    "**Contactez-nous :**\n"
                    "📞 0800 VOYAGE (gratuit)\n"
                    "📧 reservations@votreagence.com\n\n"
                    "Nous pourrons discuter options, assurances, transferts...\n"
                    "Y a-t-il un voyage qui vous attire particulièrement ?"
                )
        else:
            # Aucun voyage en mémoire
            if question_type == 'prix':
                state.dernier_message_ia = (
                    "Pour vous donner les prix, j'ai besoin de savoir quel type de voyage vous intéresse ! 💰\n\n"
                    "Nos tarifs varient selon vos préférences :\n"
                    "• **Économique** : Camping Lozère (~450€/sem)\n"
                    "• **Moyen** : Palavas plage (~800€/sem)\n"
                    "• **Luxe** : Chamonix 5⭐ (2500-3000€/sem)\n\n"
                    "Dites-moi : plage 🏖️, montagne 🏔️, ou détente 🧘 ?"
                )
            else:
                state.dernier_message_ia = (
                    "Je serais ravi de vous donner ces informations ! 📋\n\n"
                    "D'abord, aidez-moi à identifier le voyage qui vous correspond.\n"
                    "Que préférez-vous ?\n\n"
                    "• Mer & Plage 🏖️\n"
                    "• Montagne & Nature 🏔️\n"
                    "• Ville & Culture 🏙️\n"
                    "• Sport & Aventure ⚽\n"
                    "• Repos & Bien-être 🧘\n\n"
                    "Une fois votre choix fait, je vous donnerai tous les détails !"
                )
        
        state.done = True
        return state
    
    # ------------------------------------------------------------------------
    # PHASE 2.5: DÉTECTION MESSAGE HORS CONTEXTE
    # ------------------------------------------------------------------------
    if detect_hors_contexte(message):
        logger.info("❓ Message hors contexte ou incompréhensible détecté")
        state.dernier_message_ia = (
            "Je suis désolé, je suis un conseiller voyage et je ne comprends pas votre message. 🤔 "
            "Je peux vous aider à trouver des séjours : plage 🏖️, montagne 🏔️, ville 🏙️, "
            "sport ⚽ ou détente 🧘. Que recherchez-vous pour vos prochaines vacances ?"
        )
        state.done = True
        return state
    
    # ------------------------------------------------------------------------
    # PHASE 3: EXTRACTION DES CRITÈRES
    # ------------------------------------------------------------------------
    criteres_extraits = False
    
    if message and not detect_hors_contexte(message):  # Ne pas extraire si hors contexte
        try:
            logger.info("🔍 Extraction des critères...")
            
            # Configuration pour structured output
            model_with_structure = model.with_structured_output(Criteres)
            prompt = PROMPT_EXTRACTION_CRITERES.format(message=message)
            
            # Extraction via Mistral
            criteres_response = await model_with_structure.ainvoke(prompt)
            
            # Mise à jour des critères
            nouveaux_criteres = 0
            for key, value in criteres_response.dict().items():
                if value is not None:
                    state.criteres[key] = value
                    criteres_extraits = True
                    nouveaux_criteres += 1
                    logger.info(f"  ✓ {key}: {value}")
                    
            logger.info(f"✅ {nouveaux_criteres} critère(s) extrait(s)")
            
        except Exception as e:
            logger.warning(f"⚠️ Erreur extraction critères: {e}")
            state.erreur_ia = True
    
    # ------------------------------------------------------------------------
    # PHASE 4: GESTION MESSAGE PEU CLAIR (si Mistral n'a extrait aucun critère)
    # ------------------------------------------------------------------------
    if not criteres_extraits and not any(v is not None for v in state.criteres.values()):
        logger.info("❓ Aucun critère voyage identifié")
        
        # Double vérification : est-ce vraiment un message voyage ?
        if detect_hors_contexte(message):
            state.dernier_message_ia = (
                "Je suis un conseiller voyage spécialisé et votre message ne semble pas "
                "concerner une recherche de voyage. 🤷 "
                "Je peux vous aider à trouver : des séjours à la plage 🏖️, "
                "en montagne 🏔️, en ville 🏙️, sportifs ⚽ ou détente 🧘. "
                "Qu'est-ce qui vous ferait plaisir ?"
            )
        else:
            # Message qui pourrait être voyage mais pas clair
            try:
                response = await model.ainvoke(PROMPT_CLARIFICATION.format(message=message))
                state.dernier_message_ia = response.content
            except:
                state.dernier_message_ia = (
                    "Je ne suis pas sûr de comprendre votre demande de voyage. 🤔 "
                    "Pourriez-vous préciser ce que vous recherchez ? "
                    "Par exemple : mer, montagne, ville, sport ou détente ?"
                )
        state.done = True
        return state
    
    # ------------------------------------------------------------------------
    # PHASE 5: RECHERCHE DES VOYAGES
    # ------------------------------------------------------------------------
    logger.info("🔎 Recherche des voyages correspondants...")
    voyages_matches = []
    
    for voyage in VOYAGES:
        if match_voyage(voyage, state.criteres):
            voyages_matches.append(voyage)
            logger.info(f"  ✅ {voyage['nom']}")
    
    state.voyages_proposes = voyages_matches
    
    # Sauvegarder les voyages proposés pour les questions ultérieures
    if voyages_matches:
        state.derniers_voyages_proposes = voyages_matches
    
    logger.info(f"📊 {len(voyages_matches)} voyage(s) trouvé(s)")
    
    # ------------------------------------------------------------------------
    # PHASE 6: GÉNÉRATION DE LA RÉPONSE
    # ------------------------------------------------------------------------
    try:
        # Modèle avec température plus élevée pour créativité
        model_generation = init_chat_model(
            model=MODEL_NAME,
            model_provider="mistralai",
            temperature=0.6,
            api_key=MISTRAL_API_KEY
        )
        
        if voyages_matches:
            # Formater les voyages pour le prompt
            voyages_str = ""
            for v in voyages_matches:
                voyages_str += f"\n• {v['nom']}"
                voyages_str += f"\n  📝 {v['description']}"
                if v.get('points_forts'):
                    voyages_str += f"\n  ⭐ Points forts: {', '.join(v['points_forts'])}"
                if v.get('prix_indicatif'):
                    voyages_str += f"\n  💰 Prix: {v['prix_indicatif']}"
            
            prompt = PROMPT_REPONSE_AVEC_VOYAGES.format(
                message=message,
                criteres_str=format_criteres_str(state.criteres),
                voyages_str=voyages_str
            )
        else:
            # Aucun voyage ne correspond
            voyages_disponibles = "\n".join([
                f"• {v['nom']}: {v['description']}"
                for v in VOYAGES
            ])
            
            prompt = PROMPT_REPONSE_SANS_VOYAGES.format(
                message=message,
                criteres_details=format_criteres_str(state.criteres),
                nb_voyages=len(VOYAGES),
                voyages_disponibles=voyages_disponibles
            )
        
        response = await model_generation.ainvoke(prompt)
        state.dernier_message_ia = response.content
        logger.info("✅ Réponse générée avec succès")
        
    except Exception as e:
        logger.error(f"❌ Erreur génération réponse: {e}")
        
        # Réponse de secours
        if voyages_matches:
            noms = [v['nom'] for v in voyages_matches[:3]]
            state.dernier_message_ia = (
                f"J'ai trouvé {len(voyages_matches)} voyage(s) parfait(s) pour vous ! "
                f"{', '.join(noms)}. "
                f"Lequel vous attire le plus ? 🌟"
            )
        else:
            state.dernier_message_ia = (
                "Je n'ai pas trouvé de voyage correspondant exactement à vos critères. "
                "Seriez-vous flexible sur certains points ? "
                "Nos meilleures offres incluent des séjours en montagne, à la plage et en ville. 🌍"
            )
    
    state.done = True
    state.metadata["processing_time"] = datetime.now().isoformat()
    
    return state

# ================================================================================
# CONSTRUCTION DU GRAPHE LANGGRAPH
# ================================================================================

def build_graph() -> CompiledStateGraph:
    """
    Construit le graphe LangGraph pour l'agent de voyage.
    
    Architecture simple:
        START → process_message → END
    
    Returns:
        Graphe compilé prêt pour LangGraph Studio
    """
    logger.info("🏗️ Construction du graphe LangGraph...")
    
    # Création du workflow
    workflow = StateGraph(State)
    
    # Ajout du nœud unique
    workflow.add_node("process_message", process_message)
    
    # Configuration des transitions
    workflow.set_entry_point("process_message")
    workflow.add_edge("process_message", END)
    
    # Compilation
    compiled = workflow.compile()
    
    logger.info("✅ Graphe compilé avec succès")
    return compiled

# ================================================================================
# POINT D'ENTRÉE PRINCIPAL POUR LANGGRAPH
# ================================================================================

# Variable exportée pour LangGraph (référencée dans langgraph.json)
graph = build_graph()

# ================================================================================
# SECTION TEST (si exécuté directement)
# ================================================================================

if __name__ == "__main__":
    """
    Section de test pour exécution directe (hors LangGraph).
    Permet de tester l'agent localement.
    """
    import asyncio
    
    async def test_local():
        """Test local de l'agent"""
        print("\n" + "="*70)
        print(" AGENT DE VOYAGE - TEST LOCAL ")
        print("="*70 + "\n")
        
        # Messages de test
        test_messages = [
            "Bonjour !",
            "Je cherche un séjour à la plage pour me détendre",
            "Quel prix ?",  # Question sur les prix après proposition
            "Plutôt montagne et sport, sans accessibilité PMR",
            "Et les tarifs ?",  # Autre question prix
            "Une ville au bord de mer",
            "srrrrzzzhdj",  # Charabia
            "comment réserver ?",  # Question réservation
            "123456789",  # Nombres
            "Je veux tout : plage, montagne, ville, sport et détente !",
        ]
        
        for msg in test_messages:
            print(f"\n{'─'*50}")
            print(f"💬 USER: {msg}")
            print(f"{'─'*50}")
            
            # Invocation du graphe
            result = await graph.ainvoke({
                "dernier_message_utilisateur": msg
            })
            
            # Affichage des résultats
            if any(v is not None for v in result["criteres"].values()):
                print(f"📋 Critères: {format_criteres_str(result['criteres'])}")
            
            if result["voyages_proposes"]:
                print(f"🎯 {len(result['voyages_proposes'])} voyage(s) trouvé(s)")
            
            print(f"🤖 AGENT: {result['dernier_message_ia']}")
            
            # Pause entre les tests
            await asyncio.sleep(1)
        
        print(f"\n{'='*70}")
        print(" FIN DES TESTS ")
        print(f"{'='*70}\n")
    
    # Lancer les tests
    asyncio.run(test_local())

# ================================================================================
# FIN DU FICHIER
# ================================================================================
"""
Documentation complète disponible sur: https://github.com/votre-repo
Support: contact@votreagence.com
Version: 2.0 - Compatible LangGraph Studio
"""