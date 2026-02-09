# Backlog PFE : Génération d'Objets 3D et Serveur MCP Unity

## Vue d'ensemble

Le planning est dense. La phase MCP est intégrée comme une **évolution majeure** en fin de parcours pour transformer le projet d'un simple "générateur" en un **"assistant intelligent"**.

---

## Phase 1 : Infrastructure & Ingestion

**Objectif :** Socle technique et nettoyage des données.

| ID | User Story | Description |
|----|------------|-------------|
| US-1.1 | [Infra] Setup Docker-Compose | Mise en place de Docker-Compose (FastAPI, Redis, Celery) |
| US-1.2 | Pipeline unstructured | Implémentation du pipeline unstructured pour parser PDF et Emails (.eml) |
| US-1.3 | Filtres Regex | Création de filtres Regex pour nettoyer le "bruit" (signatures, headers) |
| US-1.4 | [API] Endpoint /upload asynchrone | Endpoint qui renvoie un task_id |

---

## Phase 2 : Le Cerveau - LLM & Extraction

**Objectif :** Comprendre la demande utilisateur.

| ID | User Story | Description |
|----|------------|-------------|
| US-2.1 | [IA] Intégration Llama-3-8B-Instruct | Intégration via llama-cpp-python |
| US-2.2 | [IA] Prompt Système JSON | Développement du Prompt Système pour forcer la sortie JSON (Schema Enforcement) |
| US-2.3 | [IA] Extraction de tableaux | Gestion des cas complexes : extraction de tableaux de dimensions depuis un PDF |
| US-2.4 | Validation Pydantic | Validation Pydantic des métadonnées extraites |

---

## Phase 3 : Le Constructeur - Génération 3D

**Objectif :** Créer l'asset numérique.

| ID | User Story | Description |
|----|------------|-------------|
| US-3.1 | Intégration TripoSR | Intégration du modèle TripoSR pour la génération Text-to-Mesh rapide |
| US-3.2 | [IA Code] Scripts C# procéduraux | Pipeline alternatif : génération de scripts C# procéduraux pour les formes géométriques |
| US-3.3 | Conversion et optimisation | Conversion et optimisation des assets (OBJ vers GLB) |
| US-3.4 | Stockage temporaire | Gestion du stockage temporaire des fichiers générés |

---

## Phase 4 : Serveur MCP Unity - L'Intégration Agentique

**Objectif :** Remplacer l'import manuel par un pilotage automatique via MCP.

| ID | User Story | Description |
|----|------------|-------------|
| US-4.1 | [MCP Python] Client MCP | Implémenter un Client MCP dans le microservice Python (utilisant le SDK mcp) |
| US-4.2 | [MCP Unity] Serveur MCP C# | Développer (ou adapter unity-mcp) un Serveur MCP en C# dans Unity qui écoute sur un port WebSocket/Stdio |
| US-4.3 | Définition des Outils Unity | Définir les "Outils" Unity exposés à l'IA :<br>- `SpawnGLB(url, position)`<br>- `CreatePrimitive(type, scale)`<br>- `LogMessage(text)` |
| US-4.4 | [Orchestration] Connexion pipeline | Connecter le pipeline de fin de génération (Celery) à l'appel d'outil MCP. Dès que le fichier est prêt, l'ordre d'import est envoyé |

---

## Phase 5 : UX & Finalisation

**Objectif :** Expérience utilisateur et livrables.

| ID | User Story | Description |
|----|------------|-------------|
| US-5.1 | [Unity] Moniteur IA | Créer une fenêtre "Moniteur IA" dans Unity pour voir les logs de réception MCP |
| US-5.2 | Test de bout en bout | Test complet : Email → Extraction → Génération → Apparition dans Unity sans clic |
| US-5.3 | Documentation | Rédaction du mémoire et documentation technique de l'architecture MCP |
| US-5.4 | [Optim] Nettoyage et optimisation | Nettoyage du code et optimisation de la VRAM |

---

## Planning des Sprints

| Sprint | Durée (Semaines) | Tâches Incluses (US) | Objectif Principal du Sprint |
|--------|-----------------|---------------------|---------------------------|
| 1 | 2 | US-1.1, US-1.2, US-1.3 | Mise en place de l'infrastructure Docker/FastAPI/Celery et le pipeline d'ingestion des documents (unstructured) |
| 2 | 2 | US-1.4, US-2.1, US-2.2 | Endpoint d'upload asynchrone et intégration initiale du LLM (Llama-3) avec la contrainte de sortie JSON |
| 3 | 2 | US-2.3, US-2.4, US-3.1 | Finalisation de l'extraction de métadonnées complexes (tableaux), validation Pydantic, et intégration du modèle TripoSR (Text-to-Mesh) |
| 4 | 2 | US-3.2, US-3.3, US-3.4 | Développement du pipeline alternatif de génération de scripts C# procéduraux, conversion et gestion du stockage des assets (.glb) |
| 5 | 2 | US-4.1, US-4.2, US-4.3 | Implémentation du Client MCP (Python SDK) et du Serveur MCP (Unity C#) avec la définition des premiers "Outils" Unity |
| 6 | 2 | US-4.4, US-5.1, US-5.2 | Connexion du pipeline de génération à l'appel d'outil MCP (Orchestration), création du "Moniteur IA" dans Unity, et test de bout en bout complet |
| 7 | 2 | US-5.3 | Focus : Rédaction du mémoire de fin d'études et de la documentation technique de l'architecture MCP |
| 8 | 2 | US-5.4, Révision, Polissage | Nettoyage final du code, optimisation de la VRAM et révision/polissage général avant la livraison |

**Durée totale :** 16 semaines

---

## Ressources Spécifiques

### Modèles IA (Hugging Face)

*Peuvent changer selon les contraintes matérielles*

| Usage | Modèle | Description |
|-------|--------|-------------|
| Extraction | meta-llama/Meta-Llama-3-8B-Instruct | Le standard actuel pour suivre des instructions |
| Code C# | Qwen/Qwen2.5-Coder-7B-Instruct | Excellent pour le code Unity et le JSON |
| Génération 3D | stabilityai/TripoSR | Le plus rapide et stable pour un usage local |

### Bibliothèques MCP & Unity

| Technologie | Package/Référence | Description |
|-------------|------------------|-------------|
| Python SDK | `pip install mcp` | Officiel Anthropic |
| Unity MCP | CoplayDev/unity-mcp ou ModelContextProtocol/csharp-sdk | Base du serveur C# |
| Ingestion | `pip install unstructured[pdf,email]` | Parsing de documents |

### Datasets (Kaggle / Hugging Face)

| Dataset | Usage |
|---------|-------|
| Objaverse-XL | Pour comprendre la structure des objets 3D (si besoin de fine-tuning) |
| DeepCAD | Pour les descriptions techniques (CAO) |

---

## Sources des Citations

1. **Quickstart - Unstructured document** - https://docs.unstructured.io/open-source/introduction/quick-start (février 3, 2026)
2. **Partitioning - Unstructured Documentation** - https://docs.unstructured.io/open-source/core-functionality/partitioning (février 3, 2026)
3. **TripoSR: Fast 3D Object Reconstruction from a Single Image** - arXiv, https://arxiv.org/html/2403.02151v1 (février 3, 2026)
4. **Best AI 3D Generator? I Tested Every Major Platform (Shocking Results!)** - YouTube, https://www.youtube.com/watch?v=jfk8e4ykp-s (février 3, 2026)
5. **The 11 best open-source LLMs for 2025** - n8n Blog, https://blog.n8n.io/open-source-llm/ (février 3, 2026)
6. **SteliosGian/fastapi-celery-redis-flower** - GitHub, https://github.com/SteliosGian/fastapi-celery-redis-flower (février 3, 2026)
7. **Building Scalable Background Jobs with FastAPI + Celery + Redis** - Medium, https://medium.com/@shaikhasif03/building-scalable-background-jobs-with-fastapi-celery-redis-e43152829c61 (février 3, 2026)
8. **UnityCopilot** - GitHub, https://github.com/Danejw/UnityCopilot (février 3, 2026)

---

## Architecture du Projet

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│   Documents     │────▶│  Phase 1 & 2     │────▶│   Phase 3       │
│  (PDF, Email)   │     │  Infra & LLM     │     │  Génération 3D  │
└─────────────────┘     └──────────────────┘     └────────┬────────┘
                                                          │
                                                          ▼
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│   Unity Editor  │◀────│  Phase 4         │◀────│  US-4.4         │
│   (MCP Server)  │     │  MCP Unity       │     │  Orchestration  │
└─────────────────┘     └──────────────────┘     └─────────────────┘
```

---

*Document généré à partir du Backlog PFE - Génération d'objets 3D et Serveur MCP Unity*
