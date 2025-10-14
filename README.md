# Nouveau projet LangGraph + Mistral

Ce template est utilisé dans le cadre des formations **LBKE** à [LangChain et LangGraph](https://www.lbke.fr/formations/developpeur-llm-langgraph-langchain/cpf).

### Différences avec le template de base
- Connexion avec **Mistral** déjà prête.
- Version fixée pour le CLI **LangGraph (>= 0.6.0)** pour accéder à la Runtime.
- Fichier **Render.yaml** inclus pour faciliter le déploiement gratuit sur Render.
- Workflows GitHub **désactivés par défaut**.

---

# Agent (LangGraph) — README d’installation

Ce dépôt est un **starter** pour créer un agent basé sur **LangGraph** et **LangChain** avec Mistral. Le fichier `pyproject.toml` fourni (ci‑dessous) définit les dépendances de base, les extras de dev et la configuration de packaging.

> Python requis : **3.9+**

---

## 1) Prérequis

- **Python 3.9+** (3.11/3.12 recommandé)
- **Git**
- Une **clé API Mistral** (variable d’environnement `MISTRAL_API_KEY`)

Optionnel :
- **uv** (gestionnaire rapide compatible PEP 517/518)
- ou **pip** classique

---

## 2) Cloner le projet

```bash
git clone <votre-repo-ou-ce-starter> agent
cd agent
```

---

## 3) Configuration de l’environnement

### Option A — avec **uv** (recommandé)

```bash
uv venv
source .venv/bin/activate  # (Windows: .venv\Scripts\activate)
uv sync
uv sync --group dev
```

### Option B — avec **pip**

```bash
python -m venv .venv
source .venv/bin/activate  # (Windows: .venv\Scripts\activate)

pip install -e .
pip install '.[dev]' 'langgraph-cli[inmem]>=0.2.8' 'pytest>=8.3.5' 'anyio>=4.7.0'
```

---

## 4) Variables d’environnement

Créez un fichier **`.env`** à la racine :

```env
MISTRAL_API_KEY=sk-...
LANGCHAIN_TRACING_V2=true
LANGCHAIN_PROJECT=agent
```

> Le paquet `python-dotenv` charge automatiquement ces variables si vous appelez `load_dotenv()` dans votre code.

---

## 5) Lancer l’agent en local

### Avec LangGraph CLI (mode dev)

```bash
langgraph dev
```

➡️ Cela démarre une session locale de développement avec un **checkpointer en mémoire volatile** : la mémoire est conservée pendant la session mais perdue au redémarrage.

### Persistance simple (SQLite)

```python
from langgraph.checkpoint.sqlite import SqliteSaver
checkpointer = SqliteSaver.from_conn_string("checkpoints.sqlite")
app = graph.compile(checkpointer=checkpointer)
```

---

## 6) Organisation du code

```text
src/
  agent/
    __init__.py   
    graph.py         # définition des nœuds/edges
   langgraph.json
   render.yaml

---

## 7) Tests & Qualité

```bash
avec langgraph sur render.com
```

---

## 8) Commandes rapides

```bash
uv venv && source .venv/bin/activate && uv sync && uv sync --group dev
python -m venv .venv && source .venv/bin/activate && pip install -e . '.[dev]' 'langgraph-cli[inmem]>=0.2.8'
langgraph dev
pytest -q
ruff check . && ruff format .
mypy src
```

---

## 9) Dépannage

- **`langgraph: command not found`** → installez `langgraph-cli[inmem]`.
- **Mémoire perdue après redémarrage**
- **Erreur Mistral** → vérifiez `MISTRAL_API_KEY`.

---

## 10) Licence

Licence **MIT** (voir `pyproject.toml`).

---

## 🔧 Comment personnaliser

1. **Définir le contexte d’exécution** : modifiez la classe `Context` dans `graph.py` pour exposer les paramètres à configurer (prompt système, modèle LLM, etc.).
   → [Documentation sur le runtime context](https://langchain-ai.github.io/langgraph/agents/context/?h=context#static-runtime-context)

2. **Étendre le graphe** : ajoutez ou modifiez les nœuds et les liens dans `src/agent/graph.py` pour orchestrer des workflows plus complexes.

### execution local avec le sript agent/demo_agent_multi_turn.py

```bash
cd agent ; python demo_agent_multi_turn.py
```

---

## 🧪 Développement

En utilisant **LangGraph Studio**, vous pouvez :
- éditer des états précédents et relancer des nœuds spécifiques ;
- profiter du **hot reload** pour tester vos modifications ;
- créer de nouveaux threads en un clic (`+`) ;
- tracer et analyser les exécutions via **LangSmith**.

📚 Pour aller plus loin :
- [Documentation LangGraph](https://langchain-ai.github.io/langgraph/)
- [LangGraph Studio](https://langchain-ai.github.io/langgraph/concepts/langgraph_studio/)
- [LangSmith](https://smith.langchain.com/) pour le suivi et la collaboration.

