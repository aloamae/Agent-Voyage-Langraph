"""Agent de recommandation de voyages - Examen RNCP37805BC03

Architecture minimaliste : 1 nœud unique
Structured output : extraction de 6 critères booléens
Catalogue : 5 voyages prédéfinis
RESET obligatoire des critères à chaque tour

INTÉGRATION LANGSMITH pour monitoring et traçabilité
"""

from __future__ import annotations
import os
from dotenv import load_dotenv
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from pydantic import BaseModel, Field
from langchain.chat_models import init_chat_model
from langgraph.graph import StateGraph, END
from langsmith import traceable
# =============================
#   CHARGEMENT .ENV (NOUVEAU)
# =============================



# Charger le fichier .env s'il existe
# Par défaut, cherche .env dans le répertoire courant
load_dotenv()

# =============================
#   CONFIGURATION LANGSMITH
# =============================

# Variables d'environnement LangSmith (à définir dans .env)
# LANGSMITH_API_KEY=votre_clé_api
# LANGSMITH_PROJECT=voyage-agent-examen
# LANGSMITH_TRACING=true

# Lecture des variables LangSmith depuis .env
LANGSMITH_TRACING = os.getenv("LANGSMITH_TRACING", "false")
LANGSMITH_PROJECT = os.getenv("LANGSMITH_PROJECT", "voyage-agent-examen")
LANGSMITH_API_KEY = os.getenv("LANGSMITH_API_KEY")

# Affichage de l'état de LangSmith (lecture depuis .env uniquement)
if LANGSMITH_TRACING.lower() == "true":
    LANGSMITH_ENABLED = True
    print("✅ LangSmith activé depuis .env - Traçage des opérations")
    print(f"   Projet: {LANGSMITH_PROJECT}")
    
    if not LANGSMITH_API_KEY:
        print("⚠️  ATTENTION: LANGSMITH_API_KEY non définie dans .env")
        print("   Le traçage ne fonctionnera pas sans clé API")
else:
    LANGSMITH_ENABLED = False
    print("⚠️  LangSmith désactivé - Définir LANGSMITH_TRACING=true dans .env")


# =============================
#   CONFIGURATION MISTRAL AI
# =============================

# Variable d'environnement requise pour Mistral AI (à définir dans .env)
# MISTRAL_API_KEY=votre_clé_api_mistral

# Vérification de la clé API Mistral
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")

if not MISTRAL_API_KEY:
    print("❌ ERREUR: MISTRAL_API_KEY non définie")
    print("   Ajoutez MISTRAL_API_KEY=votre_clé dans le fichier .env")
    raise ValueError("MISTRAL_API_KEY est requis pour utiliser le modèle Mistral AI")
else:
    print("✅ Clé API Mistral configurée")


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

@traceable(name="match_criteres")
def match_criteres(voyage: Dict, criteres: Dict) -> bool:
    """Vérifie si un voyage correspond aux critères (logique simple)
    
    Tracé dans LangSmith pour analyser la logique de matching
    """
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


@traceable(name="trouver_voyage")
def trouver_voyage(criteres: Dict) -> Optional[Dict]:
    """Retourne le voyage correspondant le mieux aux critères
    
    Tracé dans LangSmith pour analyser le processus de sélection
    """
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


@traceable(name="generer_reponse_llm")
async def generer_reponse_llm(voyage: Dict, criteres: Dict, message: str) -> str:
    """Génère une réponse naturelle avec le LLM
    
    Tracé dans LangSmith pour monitorer les appels LLM et réponses
    """
    try:
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
    
    except Exception as e:
        # Log de l'erreur pour le débogage
        print(f"❌ Erreur lors de la génération de réponse: {type(e).__name__}: {str(e)}")
        
        # Retourner une réponse de secours user-friendly
        return f"""Je vous recommande : {voyage['nom']}

Ce voyage correspond à vos critères. Malheureusement, je rencontre un problème technique pour générer une description détaillée.

Caractéristiques :
- Type: {', '.join(voyage['labels'])}
- Accessibilité PMR: {'Oui' if voyage['accessibleHandicap'] else 'Non'}

Souhaitez-vous plus d'informations ou explorer d'autres options ?"""


# =============================
#   NŒUD UNIQUE (conforme examen)
# =============================

@traceable(
    name="process_message",
    metadata={
        "node_type": "main_processor",
        "description": "Nœud unique - Extraction + Matching + Génération"
    }
)
async def process_message(state: State) -> Dict[str, Any]:
    """
    Nœud unique - Cycle complet conforme examen :
    1. Extraction structured output
    2. RESET critères (obligatoire)
    3. Application nouveaux critères
    4. Validation : all(None) ?
    5. Matching + génération réponse
    
    Entièrement tracé dans LangSmith pour analyse complète
    """
    message = state.dernier_message_utilisateur
    
    # 1. EXTRACTION avec structured output
    try:
        model = init_chat_model("mistral-small-latest", model_provider="mistralai")
        model_struct = model.with_structured_output(Criteres)
        
        prompt_extraction = PROMPT_EXTRACTION.format(message=message)
        extraits = await model_struct.ainvoke(prompt_extraction)
        
        # Log des critères extraits (visible dans LangSmith)
        print(f"📊 Critères extraits: {extraits.dict()}")
    
    except Exception as e:
        # Log de l'erreur pour le débogage
        print(f"❌ Erreur lors de l'extraction des critères: {type(e).__name__}: {str(e)}")
        
        # En cas d'erreur d'extraction, retourner un message d'erreur user-friendly
        message_erreur = """Je rencontre un problème technique pour analyser votre demande.

Pourriez-vous reformuler votre demande en précisant vos préférences parmi :
- Plage
- Montagne  
- Ville
- Sport
- Détente
- Accessibilité PMR

Exemple : "Je cherche un séjour à la plage avec détente" """
        
        return {
            "dernier_message_ia": message_erreur,
            "criteres": {k: None for k in state.criteres}
        }
    
    # 2. RESET OBLIGATOIRE des critères (pas d'héritage entre tours)
    nouveaux_criteres = {k: None for k in state.criteres}
    
    # 3. APPLICATION des nouveaux critères extraits
    for k, v in extraits.dict().items():
        if v is not None:
            nouveaux_criteres[k] = v
    
    # 4. VALIDATION : aucun critère rempli ?
    if all(v is None for v in nouveaux_criteres.values()):
        print("⚠️  Aucun critère identifié - Demande de clarification")
        return {
            "dernier_message_ia": PROMPT_CLARIFICATION,
            "criteres": nouveaux_criteres
        }
    
    # 5. MATCHING : recherche du 1er voyage correspondant
    voyage = trouver_voyage(nouveaux_criteres)
    
    if voyage:
        print(f"✅ Voyage trouvé: {voyage['nom']}")
        # Génération réponse avec LLM
        message_ia = await generer_reponse_llm(voyage, nouveaux_criteres, message)
    else:
        print("❌ Aucun voyage correspondant aux critères")
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
    """Construit le graphe LangGraph minimaliste
    
    Le graphe sera automatiquement tracé dans LangSmith via langgraph dev
    
    Note: Pas de checkpointer défini ici car langgraph dev gère 
    automatiquement la persistance via sa plateforme.
    """
    workflow = StateGraph(State)
    
    # 1 seul nœud
    workflow.add_node("process_message", process_message)
    
    # Arêtes simples (pas de conditionnelles)
    workflow.set_entry_point("process_message")
    workflow.add_edge("process_message", END)
    
    # Compilation SANS checkpointer - géré automatiquement par langgraph dev
    graph = workflow.compile(name="Agent Voyage Examen")
    
    if LANGSMITH_ENABLED:
        print("🔍 Graphe compilé - Traçage actif dans LangSmith")
        print("💾 Persistance gérée automatiquement par LangGraph API")
    
    return graph


# Export pour langgraph dev
graph = build_graph()
