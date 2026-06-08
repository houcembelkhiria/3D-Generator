# Soutenance technique — Questions & Réponses

**Projet :** 3D Generator — Microservice IA agentique de génération 3D
**Stack :** FastAPI + Celery + LangGraph + Hunyuan3D + React 19 + Unity
**Date :** Préparation soutenance

---

## Table des matières

1. [Architecture générale](#1-architecture-générale)
2. [FastAPI & backend Python](#2-fastapi--backend-python)
3. [Celery — file de tâches](#3-celery--file-de-tâches)
4. [LangGraph — orchestration agentique](#4-langgraph--orchestration-agentique)
5. [Hunyuan3D — pipeline ML 3D](#5-hunyuan3d--pipeline-ml-3d)
6. [Génération multi-vues & substitution texture](#6-génération-multi-vues--substitution-texture)
7. [Cache vectoriel ChromaDB](#7-cache-vectoriel-chromadb)
8. [Parsing documents & LLM](#8-parsing-documents--llm)
9. [Frontend React 19 & TypeScript](#9-frontend-react-19--typescript)
10. [WebSocket, polling, temps réel](#10-websocket-polling-temps-réel)
11. [Intégration Unity Editor](#11-intégration-unity-editor)
12. [Déploiement Docker](#12-déploiement-docker)
13. [Sécurité](#13-sécurité)
14. [Performance & scalabilité](#14-performance--scalabilité)
15. [Tests & qualité de code](#15-tests--qualité-de-code)
16. [Choix techniques & alternatives](#16-choix-techniques--alternatives)
17. [Limitations connues](#17-limitations-connues)
18. [Questions pièges & comment répondre](#18-questions-pièges--comment-répondre)

---

## 1. Architecture générale

### Q1.1 Décrivez l'architecture globale du système.
**R :** Architecture microservice à 4 couches :
1. **Frontend SPA** (React 19 + Vite + TypeScript) sur port 3001
2. **API REST** (FastAPI) sur port 8001
3. **File de tâches** (Celery + Redis broker) — file `3d_generation` dédiée GPU
4. **Workers ML** (Celery workers) qui chargent les modèles Hunyuan3D et exécutent l'inférence

Communication : Frontend ↔ FastAPI via HTTP/WebSocket, FastAPI ↔ Workers via Celery/Redis, état persistant via SQLite (gallery_db, pipeline_checkpoints) + ChromaDB (cache vectoriel).

### Q1.2 Pourquoi un microservice et pas un monolithe ?
**R :** Trois raisons :
- **Isolation GPU** : le worker ML peut crasher (OOM CUDA, modèle corrompu) sans tuer FastAPI
- **Scalabilité indépendante** : on peut déployer N workers GPU sur des machines séparées, FastAPI reste léger
- **Découplage temporel** : les requêtes utilisateur (ms) ne bloquent jamais les jobs ML (20 min)

### Q1.3 Pourquoi 4 modes de génération (Document, Image, Texte, Multi-vues) ?
**R :** Chaque mode adresse un cas d'usage différent :
- **Image→3D** : créateurs ayant déjà un visuel
- **Texte→3D** : prototypage rapide depuis prompt (passe par T2I puis I2I)
- **Multi-vues→3D** : qualité géométrique supérieure (mv_pipeline 1.1B vs i23d mini 0.6B)
- **Document→3D** : automatisation B2B (extraction LLM + génération)

### Q1.4 Comment les composants communiquent-ils ?
**R :**
- HTTP REST pour les requêtes synchrones (upload, status check)
- WebSocket `/ws/generation/{uid}` pour la progression temps réel
- Celery via Redis pour le dispatch worker
- SQLite pour la persistance (gallery_db, pipeline_checkpoints.db)
- ChromaDB pour le cache vectoriel (embeddings + métadonnées)
- Système de fichiers JSON pour Unity (`SpawnRequests/`)

### Q1.5 Quelle est la différence entre votre projet et un wrapper SaaS comme Meshy/Luma ?
**R :** Quatre contributions originales :
1. **Document PDF/Email → 3D** automatique via LangGraph + LLM local
2. **Intégration Unity Editor** directe via SpawnBridge (sans plugin tiers)
3. **Cache vectoriel** par similarité d'embeddings CLIP/DINO (cosine ≥ 0.95)
4. **Pipeline agentique** avec retry/fallback/checkpointer (vs appel API opaque)

Et auto-hébergeable : pas de cloud, pas de fuite de propriété intellectuelle.

---

## 2. FastAPI & backend Python

### Q2.1 Pourquoi FastAPI plutôt que Flask ou Django ?
**R :** Trois critères :
- **Async natif** (Starlette + ASGI) : WebSocket et long polling sans threads
- **Validation automatique** via Pydantic : un Pydantic model = schéma + validation + documentation OpenAPI
- **Documentation auto-générée** : `/docs` (Swagger) et `/redoc` sans code supplémentaire
- **Performance** : un des frameworks Python les plus rapides selon TechEmpower

Django serait surdimensionné (ORM, admin, sessions inutilisés). Flask manquerait l'async + Pydantic intégré.

### Q2.2 Comment gérez-vous le cycle de vie de l'application FastAPI ?
**R :** Via `lifespan` async context manager dans `app/main.py` :
```python
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup : init vector store, warm up models
    yield
    # Shutdown : cleanup
```
Cela garantit que les ressources lourdes (vector store ChromaDB) sont initialisées une fois et nettoyées proprement.

### Q2.3 Comment FastAPI gère les fichiers uploadés ?
**R :** `UploadFile` (multipart/form-data). Lecture en mémoire ou stream selon la taille. Validation du `content_type` à la main (whitelist), limite de taille via `MAX_FILE_SIZE=50MB` dans `Settings`. Fichier sauvé avec UUID dans `UPLOAD_DIR` puis dispatché à Celery par chemin (pas en bytes).

### Q2.4 Comment validez-vous les requêtes ?
**R :** Pydantic models pour chaque endpoint :
- `ImageTo3DRequest`, `TextTo3DRequest`, `MultiViewTo3DRequest` — validés automatiquement par FastAPI
- Champs avec `Field(default=..., ge=..., le=...)` pour les bornes (ex : `octree_resolution` ∈ [64, 256])
- `model_config = ConfigDict(extra="forbid")` pour rejeter les champs inconnus

### Q2.5 Comment gérez-vous CORS ?
**R :** Middleware `CORSMiddleware` avec `BACKEND_CORS_ORIGINS = ["http://localhost:3001", "http://localhost:8001"]` (configurable via env). En production, restreint au domaine final pour éviter les requêtes cross-origin malicieuses.

### Q2.6 Pourquoi Pydantic V2 et pas V1 ?
**R :** V2 est 5-50x plus rapide (cœur en Rust via `pydantic-core`), supporte le mode strict, et FastAPI 0.135+ l'exige.

### Q2.7 Comment FastAPI gère les routes async et sync ?
**R :** Une route `async def` s'exécute dans la boucle ASGI (event loop). Une route `def` synchrone est exécutée dans un thread pool (`run_in_threadpool`). On utilise async pour les I/O (HTTP, DB async) et sync pour les opérations CPU bound ou les libs non-async.

---

## 3. Celery — file de tâches

### Q3.1 Qu'est-ce que Celery et pourquoi l'utiliser ?
**R :** Celery est un système distribué de queues de tâches asynchrones pour Python. On l'utilise pour :
- **Découpler** : FastAPI accepte la requête en ms, le worker traite en minutes
- **Distribuer** : N workers sur N machines (scaling horizontal)
- **Persistance** : tâches en file Redis survivent à un crash FastAPI
- **Isolation processus** : un worker qui crash n'impacte pas le reste

### Q3.2 Pourquoi Redis comme broker ?
**R :** Faible latence (microsecondes pour push/pop), simple à déployer (un container), supporte Pub/Sub (utilisé par Celery pour les notifications de revoke), et déjà utilisé comme result backend. Alternatives : RabbitMQ (plus robuste mais plus lourd), SQS (cloud, payant).

### Q3.3 Expliquez votre configuration Celery production.
**R :** Dans `worker.py` :
```python
task_acks_late=True              # ack après succès — requeue si worker crash
task_reject_on_worker_lost=True  # SIGKILL → requeue
worker_prefetch_multiplier=1     # 1 tâche à la fois (GPU sériel)
task_time_limit=2400             # 40 min hard limit
task_soft_time_limit=2100        # 35 min soft (SoftTimeLimitExceeded)
task_track_started=True          # état STARTED visible
result_expires=86400             # 24h dans Redis
result_extended=True             # nom de tâche + args dans le résultat
```

Sans `acks_late`, un OOM mid-mesh-gen perd la tâche. Sans `prefetch_multiplier=1`, le worker s'attribue plusieurs tâches et OOM.

### Q3.4 Comment fonctionne le routage des tâches ?
**R :** `task_routes` dans la config Celery :
```python
"app.tasks.run_pipeline":            {"queue": "3d_generation"},
"app.tasks_3d.image_to_3d_task":     {"queue": "3d_generation"},
...
```
Toutes les tâches lourdes vont sur la file `3d_generation`. Le worker démarre avec `celery -A app.worker worker -Q 3d_generation` pour ne consommer que cette file.

### Q3.5 Comment annule-t-on une tâche en cours ?
**R :** `celery_app.control.revoke(uid, terminate=True, signal='SIGTERM')` :
- Si la tâche est en file → retirée immédiatement
- Si en cours → SIGTERM envoyé au worker process
- Le signal est traité au prochain bytecode Python, donc une inférence CUDA en cours peut prendre quelques secondes avant d'être interrompue
- Le worker crashe, le pool en lance un nouveau qui recharge le modèle (~30-60s)

### Q3.6 Pourquoi `task_id` est-il l'UUID que vous générez vous-même ?
**R :** `send_task(task_id=uid)` permet au frontend de connaître l'UID AVANT que la tâche n'aille en file. Sans ça, on devrait attendre le retour de `send_task()` puis renvoyer son ID — moins prévisible. Cet UID sert aussi de nom de fichier pour le GLB de sortie et de `thread_id` LangGraph.

### Q3.7 Que se passe-t-il si Redis tombe ?
**R :** Les workers se reconnectent automatiquement (avec backoff exponentiel). Les tâches en file pendant le downtime sont conservées par Redis (persistence AOF). Si Redis est définitivement perdu, on perd la file mais pas les checkpoints LangGraph (qui sont en SQLite séparé).

### Q3.8 Comment Celery gère les exceptions ?
**R :** Si une tâche lève une exception non gérée, Celery la sérialise dans le result backend et passe l'état à FAILURE. `AsyncResult(uid).result` renvoie l'exception. Pour reprises automatiques : `@celery_app.task(autoretry_for=(ConnectionError,), max_retries=3, default_retry_delay=10)`.

### Q3.9 Pourquoi `bind=True` sur vos tâches ?
**R :** `bind=True` injecte le contexte de la tâche (`self`) en premier argument, donnant accès à :
- `self.request.id` (task ID)
- `self.update_state(state, meta)` pour le streaming de progression
- `self.retry()` pour les retries explicites

---

## 4. LangGraph — orchestration agentique

### Q4.1 Qu'est-ce que LangGraph ?
**R :** Framework Python de LangChain pour orchestrer des **workflows à état** sous forme de graphes orientés. Chaque nœud lit/écrit un état partagé (TypedDict), les arêtes (conditionnelles ou non) déterminent le nœud suivant. Différent d'une chaîne LangChain (linéaire) car supporte les boucles, les bifurcations, les sous-graphes, et le checkpointing pour reprise.

### Q4.2 Pourquoi LangGraph pour ce projet ?
**R :** Le pipeline document→3D a besoin de retry (LLM peut renvoyer du JSON invalide), fallback (spec hand-crafted), checkpointing (mesh gen prend 20 min), et streaming (UI live). Une chaîne LangChain linéaire ne couvrirait que le cas heureux. Un state machine Python pur fonctionnerait aussi mais sans le checkpointing/streaming intégrés.

### Q4.3 Décrivez votre topologie.
**R :**
```
parse_document → validate_parsed_document → spec_extraction (subgraph) →
  mesh_generation (subgraph) → store_result → END
```
- **spec_extraction** : sous-graphe avec `extract_spec_llm ⇄ validate_spec ⇄ build_fallback_spec`
- **mesh_generation** : sous-graphe avec `generate_mesh ⇄ validate_mesh`

### Q4.4 Pourquoi des sous-graphes et pas un graphe plat ?
**R :** Modularité et lisibilité. Le sous-graphe `spec_extraction` encapsule sa propre logique de retry (3 tentatives → fallback) ; depuis le parent, c'est une "boîte noire" qui prend du texte et renvoie un spec. Permet aussi de tester/évoluer chaque sous-graphe indépendamment.

### Q4.5 Qu'est-ce qu'un reducer dans LangGraph ?
**R :** Une fonction qui agrège les écritures multiples sur un même champ. Par défaut, écrire un champ écrase la valeur. Avec un reducer, on combine. Exemple :
```python
errors: Annotated[List[str], operator.add]
```
Chaque nœud retourne `{"errors": ["..."]}`, le reducer `operator.add` concatène. À la fin, on a la liste de tous les avertissements/erreurs traversés.

### Q4.6 Comment fonctionne le checkpointer ?
**R :** `SqliteSaver(conn)` enregistre l'état après chaque transition de nœud dans une SQLite. La clé est `thread_id` (fourni à `invoke`/`stream`). Si le worker crashe, on relance avec le même `thread_id` et `invoke(None, config)` — `None` signifie "reprends depuis le checkpoint". Le graphe redémarre au nœud suivant celui qui a terminé en dernier.

### Q4.7 Quelle est la limite du checkpointer dans votre cas ?
**R :** Les sous-graphes sont "un nœud" du point de vue parent. Si `mesh_generation` crashe à la 18e minute, la reprise relance `mesh_generation` depuis son entry point (= du début). Le checkpoint sauve les ~5 secondes de `parse_document` + `spec_extraction`. Trade-off intentionnel (modularité > granularité de reprise).

### Q4.8 Qu'est-ce que les arêtes conditionnelles ?
**R :** Au lieu de `add_edge("A", "B")` (toujours vers B), `add_conditional_edges("A", router_fn, {"label1": "B", "label2": "C"})`. Le `router_fn(state)` retourne un label, le dict mappe label→nœud destination. Permet le retry (router renvoie le même nœud d'origine), fallback (autre branche), ou exit (mappe vers `END`).

### Q4.9 Comment le streaming fonctionne-t-il ?
**R :** `pipeline.stream(state, config, subgraphs=True)` retourne un générateur. À chaque transition de nœud, il yield un dict `{node_name: state_update}`. On itère et appelle un callback (`on_event`) à chaque step. Dans `tasks.py`, ce callback appelle `self.update_state(meta={"current_node": node_name, ...})` ce qui rend la progression visible via `AsyncResult.info`.

### Q4.10 Pourquoi avoir séparé producer et validator ?
**R :** Choix d'architecture, pas une règle. Avantages : validation centralisée, facile à faire évoluer sans toucher les producers. Inconvénients : 2 nœuds par étape logique, état avec `*_valid` flags. Pour un projet plus simple, on aurait pu auto-valider dans le producer (lever exception si invalide) et router sur exception. On a choisi le split par lisibilité.

### Q4.11 Comment les timeouts par nœud fonctionnent-ils ?
**R :** Gestionnaire de contexte `_node_timeout(seconds, label)` qui pose un `signal.SIGALRM` (Unix uniquement). Si l'alarme déclenche, on lève `NodeTimeoutError`. Le nœud catch et écrit l'erreur dans `state.errors`, le router de retry agit normalement. Budgets configurables : `LG_TIMEOUT_LLM=120`, `LG_TIMEOUT_MESH=1200`. Fallback no-op si pas dans le main thread (Celery prefork = main thread du child, donc OK).

### Q4.12 Que se passe-t-il avec `interrupt_after` ?
**R :** Au compilation : `graph.compile(interrupt_after=["mesh_generation"])`. Le graphe s'arrête après `mesh_generation`, l'état est sauvé dans le checkpointer. Un opérateur peut inspecter, modifier l'état via `update_state()`, puis reprendre avec `invoke(None, config)`. Cas d'usage : valider un mesh avant de le stocker.

---

## 5. Hunyuan3D — pipeline ML 3D

### Q5.1 Qu'est-ce que Hunyuan3D ?
**R :** Modèle open-source de Tencent (2024) pour la génération 3D depuis image/texte/multi-vues. Architecture : un **DiT** (Diffusion Transformer) génère des latents 3D, un **VAE 3D** les décode en mesh occupancy grid, puis isosurface extraction (marching cubes) produit le mesh triangulé.

### Q5.2 Combien de modèles utilisez-vous ?
**R :** Plusieurs variantes :
- **hunyuan3d-dit-v2-mini** (0.6B) : i23d (image→3D), rapide
- **hunyuan3d-dit-v2-mv** (1.1B) : multi-vues, qualité supérieure
- **hunyuan3d-delight-v2-0** : retire l'éclairage de l'image source
- **hunyuan3d-paint-v2-0** : génère les multi-vues texture
- **HunyuanDiT / Hyper-SDXL** : text-to-image pour le mode T2I→I2I

### Q5.3 Qu'est-ce qu'un DiT ?
**R :** Diffusion Transformer : remplace la U-Net classique des diffusion models par un Transformer. Avantages : meilleur scaling, contexte global, fonctionne bien sur latents (pas pixels). Sortie : latents 3D bruités progressivement débruités sur N steps.

### Q5.4 Comment marche un VAE 3D ?
**R :** Variational Autoencoder pour la 3D. Encode une représentation 3D dense (octree/voxel grid) en latents compacts. Decode l'inverse. Permet au DiT de travailler dans un espace latent plus petit que la grille volumétrique brute. Notre VAE travaille avec `octree_resolution` (64-256) et `num_chunks` (chunks de décodage pour rester en mémoire).

### Q5.5 Quelles sont les étapes de génération image→3D ?
**R :**
1. Décodage base64 de l'image
2. **rembg** (U2-Net) pour retirer le fond
3. Sauvegarde debug de l'image nettoyée
4. **i23d_pipeline** (DiT + VAE) : `num_inference_steps=30`, `octree_resolution=128`
5. **Mesh cleanup** : floater removal, degenerate face removal, plus grande composante connexe
6. **Texture** (si activé) : `face_reducer` puis `texgen_pipeline` (delight + multiview + bake + inpaint)
7. **Export GLB** via trimesh

### Q5.6 Pourquoi rembg avant le DiT ?
**R :** Le DiT est entraîné sur des images avec fond blanc/transparent. Un fond complexe ferait apparaître des artefacts dans le mesh généré (le modèle interprète le fond comme géométrie). U2-Net est léger (~5MB) et fonctionne sans GPU dédié.

### Q5.7 Qu'est-ce que FlashVDM et pourquoi le plafonner à 256 ?
**R :** FlashVDM est l'algorithme d'extraction d'isosurface accéléré. À `octree_resolution > 256` (niveau 3), il y a un bug d'indexation qui produit des meshes corrompus. On plafonne à 192-256 en sécurité. Pour plus de détail, on augmente plutôt `num_chunks` ou `face_count`.

### Q5.8 Pourquoi `with torch.inference_mode()` ?
**R :** Désactive l'autograd (pas de tracking des gradients) — ~10-20% plus rapide et économise de la mémoire. Plus strict que `torch.no_grad()` : les tenseurs créés sont marqués "inference" et ne peuvent pas être utilisés en training.

### Q5.9 Comment vous gérez la mémoire GPU/MPS ?
**R :** `DeviceManager` (`_dm`) avec offload manuel :
- Charge un pipeline sur device (GPU) quand nécessaire
- `_offload_to_cpu("i23d_pipeline")` avant `texgen_pipeline` pour libérer la VRAM
- Sur Mac (MPS), pas de `torch.cuda.empty_cache()` natif — utilise `empty_cache()` custom

### Q5.10 Que fait `_keep_largest_component` ?
**R :** Après floater + degenerate removal, il reste parfois des débris (composantes connexes parasites). On utilise `mesh.split(only_watertight=False)` puis on garde le mesh avec le plus grand nombre de faces. Élimine les artefacts type "petite boule détachée".

### Q5.11 Comment fonctionne la texture generation ?
**R :** Pipeline `Hunyuan3DPaintPipeline` :
1. **Delight** : retire l'éclairage de l'image prompt (pour avoir l'albedo)
2. **Render normal/position maps** : 6 vues du mesh (front, left, back, right, top, bottom)
3. **Multiview diffusion** : génère les 6 vues texture conditionnées sur l'image delight + normals
4. **Substitution utilisateur** (notre custom) : remplace les vues IA par les photos utilisateur quand disponibles
5. **Back project** : projette chaque vue sur l'UV map du mesh
6. **Fast bake** : combine les projections pondérées par cos² × weight
7. **UV inpaint** : remplit les pixels UV non visibles (cv2.inpaint Navier-Stokes)

### Q5.12 Pourquoi `face_count` à 60000 par défaut ?
**R :** Compromis qualité/poids. Le mesh brut peut avoir 200k+ faces. `face_reducer` (quadric mesh simplification) réduit à 60k pour : (a) garder un GLB raisonnable (~5MB), (b) faciliter le texturing (UV unwrap plus rapide), (c) être lisible par Unity / model-viewer sans lag.

---

## 6. Génération multi-vues & substitution texture

### Q6.1 Pourquoi avez-vous implémenté une substitution de texture custom ?
**R :** Le modèle multi-vues IA génère des vues hallucinées pour les côtés non vus. Si l'utilisateur fournit des photos réelles (front, back, left, right), on veut utiliser ces vraies textures plutôt que les hallucinations. Hunyuan3D upstream n'a PAS cette fonctionnalité — entièrement custom.

### Q6.2 Quel était le problème initial ?
**R :** Trois problèmes en cascade :
1. **Centrage** : les photos utilisateur n'étaient pas centrées comme le mesh
2. **Échelle** : la case (boîtier) débordait sur le bracelet (texture leak)
3. **Distorsions** : tentatives de warping (per-row, shear) ont causé des artefacts patchwork

### Q6.3 Quelle est la solution finale ?
**R :** Approche en deux temps :
1. **Re-centrage du mesh sur médiane des vertices** dans `load_mesh()` : le `auto_center` natif utilise le centre bbox, qui est biaisé par le bracelet. La médiane des vertices est dominée par le boîtier (région dense), donc translater par `-median` place le boîtier à l'origine du mesh → centré sur le canvas dans toutes les vues caméra.
2. **target_case_size depuis la vue IA** : détecte la case dans la sortie multi-vues IA (qui rend la vraie géométrie du mesh). Scale les photos utilisateur pour matcher cette taille exacte. Plus d'overflow.

### Q6.4 Pourquoi la médiane et pas la moyenne ?
**R :** La médiane est robuste aux outliers. Pour une montre avec bracelet long, les extrémités du bracelet sont des outliers qui tireraient la moyenne. La médiane reste sur la région dense (le boîtier). Vérifié empiriquement.

### Q6.5 Comment détectez-vous le boîtier dans une image ?
**R :** Heuristique HSV :
```python
case_mask = (alpha > 10) & (hsv[..., 1] < 60) & (hsv[..., 2] > 100)
```
Basse saturation + haute luminosité = métal. Plus morphological closing (kernel ~4% min_dim) pour combler les lettres FOSSIL gravées qui cassent le masque. Puis convex hull pour englober tout l'intérieur du boîtier.

### Q6.6 Pourquoi le bbox center et pas le centroid ?
**R :** Le centroid pondéré est biaisé vers la moitié du masque qui a le plus de pixels (si les lettres cassent le ring en deux arcs inégaux). Le bbox center (midpoint min/max) reste sur le centre géométrique du boîtier, robuste aux gravures asymétriques.

### Q6.7 Pourquoi avoir abandonné les approches de warping ?
**R :** Toutes les approches de warping (per-row remap, shear simple, shear par moitié, hull-composite) causent un mismatch de bordure boîtier-bracelet entre les vues. Le bake multi-vues combine ces vues et produit un effet patchwork/camouflage sur le boîtier. Vérifié sur 4 itérations. La conclusion : accepter le décalage du bracelet plutôt que casser le bake.

### Q6.8 Comment fonctionne la skip de front substitution ?
**R :** Quand on substitue la vue front, on doit downscaler la photo pour matcher la taille de la case du mesh (~16% du canvas). Le texte FOSSIL fin se retrouve à 30-40px et flou ("melted dial"). Solution : on NE substitue PAS la vue front. Le multi-vues IA, qui rend la vraie géométrie du mesh, produit une face sans distorsion (mais sans les détails exacts de la photo utilisateur). Trade-off pour le réalisme géométrique.

### Q6.9 Et pour les vues top/bottom ?
**R :** L'utilisateur n'a jamais de photos top/bottom (vues physiquement impossibles à prendre pour une montre posée). L'IA hallucine ces vues avec des couleurs souvent fausses. Solution : on remplace par une **couleur unie médiane** calculée depuis les pixels du sujet de la photo back. Pas de gradient, pas de blur — couleur exacte du cuir du bracelet.

### Q6.10 Et si l'utilisateur ne fournit que 3 vues sur 4 (manque left ou right) ?
**R :** Synthèse par mirror : `ImageOps.mirror(user_views['right'])` devient la vue left. Géométrie approximative (la couronne se retrouve du mauvais côté), mais couleurs/matériaux corrects. Si les deux côtés manquent : fallback couleur médiane comme pour top/bottom.

---

## 7. Cache vectoriel ChromaDB

### Q7.1 Pourquoi un cache vectoriel ?
**R :** La génération 3D prend 20 min. Si l'utilisateur uploade deux fois la même image (ou une très similaire), on évite la regen. Recherche par similarité d'embedding : seuil cosine ≥ 0.95.

### Q7.2 Pourquoi ChromaDB et pas Faiss/Qdrant/Pinecone ?
**R :** ChromaDB est :
- **Embedded** : un fichier SQLite (pas de service séparé à déployer)
- **Python natif** : `pip install chromadb`, pas de gRPC ni serveur
- **Suffisant** pour notre échelle (~1000 entrées max, latence ms)

Faiss serait plus rapide mais sans métadonnées intégrées. Qdrant/Pinecone : surdimensionnés (cloud, latence réseau).

### Q7.3 Quels embeddings utilisez-vous ?
**R :** CLIP (Contrastive Language-Image Pre-training) + DINO concaténés. CLIP capture la sémantique (chat, voiture, montre), DINO capture la structure visuelle (forme, texture). Ensemble : un objet de même forme mais sémantique différente N'EST PAS un hit. Évite les faux positifs du genre "chien" matchant "loup".

### Q7.4 Comment fonctionne la recherche ?
**R :**
1. Extraction d'embedding de l'image d'entrée
2. Calcul de hash des paramètres (steps, guidance, octree_resolution, texture, face_count)
3. ChromaDB `query(embedding, where={"params_hash": h}, n_results=1)`
4. Si distance ≤ 0.05 (cosine sim ≥ 0.95) : cache hit, retourne le résultat stocké
5. Sinon : génération, puis store de l'embedding + result_json

### Q7.5 Pourquoi un hash de paramètres ?
**R :** Une même image à `steps=5, guidance=5` produit un mesh différent qu'à `steps=30, guidance=7.5`. Si on cherchait juste par similarité d'image, on rendrait le mauvais mesh. Le hash de paramètres assure qu'on compare des générations équivalentes.

### Q7.6 Que se passe-t-il à un cache miss ?
**R :** Génération normale. Puis on store l'embedding + params + résultat dans ChromaDB. La prochaine requête similaire sera un hit.

### Q7.7 Comment évolue le cache au fil du temps ?
**R :** Append-only par défaut. Endpoint `/api/v1/cache-stats` montre la taille. Endpoint `/api/v1/cache-clear` pour nettoyer. Si on veut éviction LRU : il faudrait ajouter un timestamp + tâche périodique. Pour l'instant pas nécessaire (~1000 entrées max).

---

## 8. Parsing documents & LLM

### Q8.1 Comment parsez-vous les PDF ?
**R :** Bibliothèque **unstructured** (`partition_pdf`). Extrait le texte par blocs (titres, paragraphes, listes, tableaux). Préserve la structure document. Plus robuste que PyPDF2 pour les PDF complexes avec colonnes ou tableaux.

### Q8.2 Et les emails (.eml) ?
**R :** `partition_email`. Extrait sujet, corps, attachments. Gère les emails multipart (texte + HTML). Les attachments sont parsés récursivement si supportés.

### Q8.3 Quel LLM utilisez-vous ?
**R :** Llama-3 8B Instruct via **llama-cpp-python** (binding Python pour llama.cpp). Modèle quantifié Q4_K_M (~5GB). Local, pas d'API tierce, pas de coût par requête.

### Q8.4 Pourquoi local et pas OpenAI/Anthropic ?
**R :**
- **Confidentialité** : les documents PDF peuvent être sensibles
- **Coût** : 0 € par requête (vs ~$0.01-0.10)
- **Latence** : pas de round-trip réseau
- **Reproductibilité** : pas de drift de version API
- **Souveraineté** : tourne entièrement on-premise

### Q8.5 Comment garantissez-vous une sortie JSON valide du LLM ?
**R :** Trois étapes :
1. **Prompt engineering** : exemples few-shot + instruction stricte "Return ONLY valid JSON"
2. **Extraction regex** : `llm.extract_json_from_text()` qui isole le bloc JSON via regex
3. **Validation Pydantic** : `ObjectSpec(**parsed_json)` lève si schéma invalide

Si la validation échoue, le routeur LangGraph relance le LLM (max 3 tentatives) puis bascule sur fallback hand-crafted.

### Q8.6 Quel est le schéma `ObjectSpec` ?
**R :** Pydantic model avec :
- `name: str`
- `shape: Literal["CUBE", "SPHERE", "CYLINDER", "CUSTOM", ...]`
- `dimensions: Dimensions` (length/width/height + unit)
- `material: Material` (type + color)
- `description: Optional[str]`

Strict typing pour rejeter du JSON malformé.

### Q8.7 Comment fonctionne le fallback ?
**R :** Si 3 retries LLM échouent, `build_fallback_spec_node` :
- Prend les premiers caractères du texte comme `name`
- Shape = "CUSTOM"
- Dimensions = 100×100×100 mm (défaut neutre)
- Material = Plastic / Matte Black

Spec valide garantie qui permet à la génération mesh de se lancer même sur un document illisible.

### Q8.8 Comment le prompt est-il construit ?
**R :** `PromptEngineer.create_extraction_prompt(raw_text, "document_analysis")` retourne un prompt structuré :
- Rôle : "You are a 3D object extraction assistant"
- Contexte : description du schéma JSON attendu
- Few-shot : 1-2 exemples de [texte → JSON]
- Tâche : "Extract from the following: {raw_text[:2000]}"

Le texte est tronqué à 2000 chars pour rester dans la fenêtre de contexte du Llama-3 8B (8k tokens).

---

## 9. Frontend React 19 & TypeScript

### Q9.1 Pourquoi React 19 ?
**R :** Hooks modernes (useTransition, useDeferredValue), Server Components (pas utilisés ici car SPA), Concurrent Rendering. Surtout : compatible avec model-viewer Web Component pour le viewer 3D.

### Q9.2 Pourquoi Vite et pas Create React App ?
**R :** CRA est déprécié (2023). Vite utilise esbuild pour le dev (HMR < 1s) et Rollup pour le build (tree-shaking agressif). Build plus rapide, bundle plus petit. Configuration minimale.

### Q9.3 Pourquoi TypeScript ?
**R :** Sécurité de types au compile-time : on attrape les mismatches d'API (frontend appelle un endpoint avec mauvais shape). `types.ts` partage les enums et interfaces entre composants. Refactoring sûr (renommage, signatures).

### Q9.4 Quelle gestion d'état utilisez-vous ?
**R :** Pas de Redux/Zustand. État local par composant via `useState`. Données partagées (gallery, theme) via `useContext`. Polling de tâches via custom hook `useTaskPolling`. Choix : simplicité, pas de surcouche.

### Q9.5 Décrivez le hook `useTaskPolling`.
**R :** Custom hook qui prend un `taskId`, poll `/api/v1/task/{id}` avec backoff exponentiel (1.5s → 3s → 5s → 8s selon l'elapsed). Expose `{status, meta, result, error, elapsedMs}`. Cleanup au unmount via `clearTimeout`. Réagit aux changements de `current_node` dans le meta pour montrer la progression.

### Q9.6 Comment communiquez-vous avec le backend ?
**R :** `fetch()` natif (pas axios). `API_BASE` constant configurable. Endpoints en suffixe : `${API_BASE}/api/v1/...`. WebSocket via `new WebSocket(wsBase + '/api/v1/ws/generation/' + uid)`. Polling via `useTaskPolling`.

### Q9.7 Pourquoi Tailwind CSS ?
**R :** Utility-first : classes atomiques (`flex`, `p-4`, `bg-red-500`) plutôt que stylesheets séparés. Avantages : pas de CSS orphelin, design system cohérent, build optimisé (PurgeCSS retire les classes non utilisées). Apprentissage initial mais productivité élevée.

### Q9.8 Décrivez le composant ModelViewer3D.
**R :** Wrapper React autour du Web Component `<model-viewer>` (Google). Charge un GLB URL, gère camera-orbit, lighting, exposure. Props : `src`, `cameraOrbit`, `className`. Aucune dépendance React lourde (three.js, react-three-fiber) — model-viewer fait tout.

### Q9.9 Comment gérez-vous le thème clair/sombre ?
**R :** `ThemeContext` + `ThemeToggle`. Persiste dans `localStorage`. Tailwind dark mode `'class'` : ajoute `dark` au `<html>` pour activer les variantes `dark:bg-...`. Pas de prefers-color-scheme par défaut (respect explicit user choice).

### Q9.10 Comment gérez-vous les uploads de fichiers ?
**R :** `<input type="file">` ref via `useRef`, drag-and-drop avec event handlers `onDragOver/onDrop`. Validation côté client (type MIME, taille) avant `FormData.append('file', file)` puis `fetch(..., { body: formData })`. Progress simulé via setInterval (vraie progress upload requiert XMLHttpRequest qui supporte upload events).

---

## 10. WebSocket, polling, temps réel

### Q10.1 Quand utilisez-vous WebSocket vs polling ?
**R :**
- **WebSocket** : `/ws/generation/{uid}` pour la progression des 3 modes directs (Image/Text/Multi-vues). Push à 0.5s, faible latence.
- **Polling HTTP** : `/task/{task_id}` pour le pipeline LangGraph (mode Document). Backoff exponentiel via `useTaskPolling`.

Le frontend a un fallback : si WebSocket échoue (proxy mal configuré), bascule sur polling `/generation-status/{uid}`.

### Q10.2 Pourquoi WebSocket pour la génération directe et polling pour LangGraph ?
**R :** Historique. WebSocket implémenté en premier pour Image/Text/Multi-vues. LangGraph ajouté plus tard avec polling (plus simple côté frontend). Les deux fonctionnent — le streaming LangGraph côté backend update Celery state, donc polling /task/{id} voit les transitions de nœud.

### Q10.3 Pourquoi backoff exponentiel ?
**R :** Génération = 20 min. Poller à 1s = 1200 requêtes inutiles. Backoff : 1.5s → 3s → 5s → 8s. Réactif au début (premier statut rapide), économe sur la durée. Réduit la charge FastAPI ~5x sur les tâches longues.

### Q10.4 Comment fonctionne le WebSocket côté backend ?
**R :** `@router.websocket("/ws/generation/{uid}")` accepte la connection, puis loop avec `asyncio.sleep(0.5)` :
1. Lit `celery_app.AsyncResult(uid).state` et `.info`
2. Map les états Celery (PENDING/PROCESSING/SUCCESS/FAILURE/REVOKED) vers le format `{stage, pct, ...}`
3. `await websocket.send_json(prog)`
4. Break si terminal (completed/failed/cancelled)

Lecture depuis Redis (via Celery), aucun état en mémoire FastAPI.

### Q10.5 Quel est le risque d'un proxy mal configuré ?
**R :** Nginx/AWS ALB sans `proxy_http_version 1.1` + `Upgrade` headers refuse le WebSocket upgrade. Le frontend voit `ws.onerror`, bascule sur polling. C'est pourquoi le fallback existe — sans, l'app serait cassée derrière certains proxies.

### Q10.6 Comment le frontend annule-t-il proprement ?
**R :** Bouton Cancel → `DELETE /api/v1/generation/{uid}` → backend appelle `celery_app.control.revoke(uid, terminate=True, signal='SIGTERM')`. Le WebSocket finit par recevoir state=REVOKED et envoie `{stage: 'cancelled'}`. Le frontend ferme la WS et libère l'UI.

---

## 11. Intégration Unity Editor

### Q11.1 Comment Unity reçoit-il les modèles ?
**R :** Protocole **fichiers partagés JSON**. Le backend écrit un fichier `SpawnRequests/{uid}.json` contenant `{path: "/abs/path/to.glb", position, rotation, scale}`. Unity Editor surveille ce dossier via `AssetDatabase.Refresh()` + filesystem watcher, parse le JSON, et instancie le mesh dans la scène active via glTFast.

### Q11.2 Pourquoi un protocole fichier et pas WebSocket ?
**R :**
- **Pas de dépendance réseau** : Unity Editor n'a pas besoin de tourner un serveur WebSocket
- **Pas de port à configurer** : firewall, NAT, conflits — non
- **Naturellement async** : Unity scrute quand il veut
- **Debug facile** : on voit les fichiers JSON, on peut les inspecter manuellement
- **Robuste à un crash Unity** : les requêtes restent en file sur disque

Trade-off : latence (200-500ms entre écriture et pickup) vs WebSocket (ms). Acceptable pour ce cas.

### Q11.3 Décrivez SpawnBridge.cs.
**R :** Script Editor (`[InitializeOnLoad]`) qui :
1. Au chargement : enregistre un `EditorApplication.update` callback
2. Toutes les 500ms : scan le dossier `SpawnRequests/`
3. Pour chaque `*.json` : parse via `JsonUtility.FromJson<SpawnRequest>`
4. Charge le GLB via `GltfImport.LoadFile()` (glTFast 6.14.1)
5. Instancie dans la scène à la position/rotation/scale demandée
6. Supprime le fichier JSON traité (idempotence)

### Q11.4 Pourquoi glTFast ?
**R :** Importer GLB natif Unity à runtime (vs `AssetDatabase.ImportAsset` qui ne fonctionne qu'au build time). Plus rapide qu'UnityGLTF, support PBR complet, maintenu par Unity Technologies.

### Q11.5 Qu'est-ce qu'un Assembly Definition ?
**R :** Fichier `.asmdef` qui définit un module C# isolé. SpawnBridge a son propre asmdef qui dépend de glTFast. Permet :
- Compilation incrémentale (rebuild rapide)
- Isolation des dépendances
- Future inclusion dans un package UPM

### Q11.6 Comment Unity sait quel modèle spawn ?
**R :** L'URL du modèle dans le frontend est `/api/v1/outputs/{uid}.glb`. Cliquer "Open in Unity" depuis la galerie déclenche `POST /api/v1/unity/spawn` avec l'UID. Le backend résout le path absolu sur disque et écrit `SpawnRequests/{uid}.json`. Unity le ramasse et spawn.

### Q11.7 Que se passe-t-il si Unity n'est pas ouvert ?
**R :** Les fichiers s'accumulent dans `SpawnRequests/`. Au prochain démarrage d'Unity, ils sont tous traités. Effet "file d'attente persistante" gratuite.

---

## 12. Déploiement Docker

### Q12.1 Décrivez votre architecture Docker.
**R :** `docker-compose.yml` avec services :
- **redis** (port 9501) : broker Celery + result backend
- **backend** (port 9502) : FastAPI + Celery worker (ou séparés en prod)
- **frontend** (port 9503) : Nginx servant la build Vite

Volumes : `generated/` (outputs persistants), `uploads/`, modèles Hunyuan3D (`HY3DGEN_MODELS`).

### Q12.2 Pourquoi multi-stage builds ?
**R :** Le Dockerfile backend a :
1. Stage builder : pip install des dépendances (image lourde ~5GB)
2. Stage runtime : copie juste le venv + code (image finale ~2GB)

Réduit la surface d'attaque et la taille du registry.

### Q12.3 Comment gérez-vous les modèles ML lourds ?
**R :** Pas dans l'image Docker (trop gros, 10+ GB). Téléchargés au premier démarrage via `huggingface_hub.snapshot_download` ou bind-mountés depuis l'hôte (`~/.cache/hy3dgen`). Variable d'env `HY3DGEN_MODELS` pour le chemin.

### Q12.4 Comment scaler en production ?
**R :**
- Frontend : derrière CDN (CloudFront), N replicas (stateless)
- FastAPI : N replicas, sticky session pas nécessaire (stateless)
- Redis : Sentinel ou Cluster pour HA
- Workers : 1 par GPU disponible. Auto-scaling impossible (GPU = ressource physique)

### Q12.5 Pourquoi Nginx en frontal du frontend ?
**R :** Image Vite produit du HTML/CSS/JS statique. Nginx les sert avec compression gzip, cache headers, SPA fallback (`try_files $uri /index.html`). Plus efficace qu'un serveur Node en production.

### Q12.6 Healthchecks ?
**R :** `GET /health` endpoint FastAPI retourne `{"status": "ok"}`. Docker healthcheck via `curl --fail http://localhost:8000/health`. Si fail, container restart.

---

## 13. Sécurité

### Q13.1 Quelles attaques avez-vous considérées ?
**R :**
- **SQL injection** : SQLite avec paramètres préparés (jamais de string concatenation)
- **Path traversal** : validation regex `r'[a-zA-Z0-9_\-]+'` sur tous les UIDs avant `Path()`
- **File upload abuse** : whitelist content_type, taille max 50MB
- **DoS** : Celery `task_time_limit=2400` empêche les jobs de tourner indéfiniment
- **XSS** : React échappe par défaut (pas de `dangerouslySetInnerHTML`)
- **CORS** : whitelist explicite des origins

### Q13.2 Comment gérez-vous les secrets ?
**R :** `.env` (gitignored) avec `REDIS_URL`, `HF_TOKEN`, etc. Lu par Pydantic `BaseSettings`. Jamais commité, jamais dans l'image Docker (passé en runtime via env vars). Pour production : vault (HashiCorp Vault, AWS Secrets Manager).

### Q13.3 Y a-t-il de l'authentification ?
**R :** Pas dans cette version (microservice interne). Pour production publique : OAuth2 (JWT) via FastAPI Security utilities, ou Auth0/Clerk en SaaS.

### Q13.4 Comment évitez-vous le path traversal sur les UIDs ?
**R :** Tous les endpoints qui prennent un UID vérifient :
```python
if not re.fullmatch(r'[a-zA-Z0-9_\-]+', uid):
    raise HTTPException(400, "Invalid model id")
```
Empêche `../../etc/passwd` ou les chemins absolus injectés.

### Q13.5 Que se passerait-il si un utilisateur uploadait un PDF malicieux ?
**R :** `unstructured` parse en mode read-only (pas d'exécution de scripts PDF). Le texte extrait est traité par Llama-3 puis utilisé pour construire un prompt 3D. Pas d'exécution de code arbitraire. Risque résiduel : prompt injection ("Ignore previous instructions and...") — atténué par validation Pydantic du JSON de sortie.

### Q13.6 Comment limitez-vous le rate ?
**R :** Pas de rate limiting actuellement dans le code. Pour production : `slowapi` (port de Flask-Limiter pour FastAPI) ou Nginx `limit_req`. Limiter à ~10 requêtes/min par IP pour éviter l'abus.

---

## 14. Performance & scalabilité

### Q14.1 Combien de temps prend une génération ?
**R :**
- Mode rapide (1 step, octree 64) : ~30-60s
- Mode équilibré (20 steps, octree 128, texture) : ~3-5 min
- Mode qualité (50 steps, octree 192, texture) : ~10-20 min

Sur Apple M2 Pro 16GB en MPS. Sur GPU CUDA récent : 2-4x plus rapide.

### Q14.2 Pourquoi si lent ?
**R :** Diffusion = N forward passes du DiT. Chaque pass = 1-10s sur MPS. Texture gen ajoute autant (multi-vues = 6 passes du modèle paint). Compromis qualité/temps : `num_inference_steps` contrôle.

### Q14.3 Comment accélérer ?
**R :**
- GPU plus puissant (RTX 4090 vs M2 Pro = 10x)
- Quantization (FP16 vs FP32 = 2x mémoire, 1.5x vitesse, perte qualité minime)
- TensorRT / ONNX Runtime
- Distillation : modèle plus petit entraîné sur le gros
- Cache vectoriel pour les régénérations

### Q14.4 Comment gérez-vous plusieurs requêtes simultanées ?
**R :** Celery `worker_prefetch_multiplier=1` : un worker = une tâche à la fois. Si 5 requêtes arrivent ensemble, 4 attendent en file Redis. Pour parallèlisme : N workers (N GPUs). Une seule GPU = pas de parallèle utile (saturation mémoire).

### Q14.5 Bottleneck principal ?
**R :** L'inférence ML. Le frontend (~10kb gzip), FastAPI (~ms par requête), Celery (~ms d'overhead par tâche), Redis (~μs par op) sont tous négligeables devant les 5-20 min d'inférence.

### Q14.6 Pourquoi le cache vectoriel ?
**R :** Réduit la latence perçue à 0 pour les régénérations exactes ou similaires. Si l'utilisateur ré-upload la même image après refresh : cache hit, résultat en ms au lieu de min.

### Q14.7 Comment monitor en production ?
**R :** Prévu : Prometheus exporter (`prometheus-fastapi-instrumentator`), Grafana pour les dashboards. Métriques clés : taux de cache hit, temps moyen par mode, queue depth Celery, GPU utilization. Pas implémenté dans cette version.

---

## 15. Tests & qualité de code

### Q15.1 Avez-vous des tests ?
**R :** Oui : `Backend/tests/smoke_test.py` (260 lignes). Smoke test auto-contenu, sans pytest, sans Redis, sans GPU. Utilise Celery `task_always_eager=True` pour exécuter les tâches inline. Mock du service Hunyuan3D.

### Q15.2 Que valide le smoke test ?
**R :** 23 vérifications :
- 4 endpoints `/async` retournent 202 + uid
- `/generation-status/{uid}` retourne le bon shape
- `/run-pipeline` accepte et dispatch
- `DELETE /generation/{uid}` revoke proprement
- Toutes les tâches Celery enregistrées (6 au total)
- Config production appliquée (acks_late, prefetch, time_limit)
- Schéma `Pipeline3DState` correct
- Endpoint legacy `/upload` bien supprimé

### Q15.3 Pourquoi pas pytest ?
**R :** Pytest pas installé dans la venv (et pas critique). Le smoke test utilise `unittest` builtin + `httpx.TestClient`. Exit code 0/1, lisible en CI. Pour des tests plus poussés (paramétrés, fixtures complexes), pytest serait justifié.

### Q15.4 Et la qualité de code ?
**R :**
- **Backend** : `py_compile` automatique avant commit, types via type hints Python 3.11+
- **Frontend** : `tsc --noEmit` (TypeScript strict), pas d'eslint formel (devrait être ajouté)
- **Pré-commit** : pas formellement configuré, mais workflow manuel rigoureux

### Q15.5 Coverage ?
**R :** Le smoke test couvre la surface API (~80% des endpoints) et le wiring Celery/LangGraph (compilation + registration). Ne couvre PAS la logique métier ML (mockée). Pour un coverage réel, il faudrait des tests d'intégration avec GPU.

### Q15.6 Comment vérifiez-vous l'absence de régression ?
**R :** Le smoke test détecte les régressions structurelles (endpoint disparu, signature de tâche changée, kwargs mismatch). Pour les régressions fonctionnelles (mesh distordu, texture floue) : tests visuels manuels (screenshots avant/après).

---

## 16. Choix techniques & alternatives

### Q16.1 Pourquoi Python et pas Go/Rust pour le backend ?
**R :** L'écosystème ML est exclusivement Python (PyTorch, transformers, diffusers, hy3dgen). Réécrire en Go/Rust nécessiterait des bindings vers libtorch + de la cross-compile. Pas pertinent quand 99% du temps est dans des C++/CUDA kernels appelés depuis Python.

### Q16.2 Pourquoi pas Temporal/Prefect pour l'orchestration ?
**R :** Temporal est plus puissant mais lourd (Temporal server à déployer). Prefect = SaaS payant ou self-host complexe. LangGraph est minimal (`pip install langgraph`), suffit pour notre besoin. Si on devait orchestrer 50+ workflows complexes : Temporal serait justifié.

### Q16.3 Pourquoi pas Triton Inference Server ?
**R :** Triton serait idéal pour servir le modèle en production à grande échelle (batching, multi-instance, gRPC). Mais : configuration complexe, ne supporte pas tous les modèles, MPS Apple non supporté. Pour un projet PFE/POC, l'overhead n'est pas justifié.

### Q16.4 Pourquoi SQLite et pas PostgreSQL ?
**R :** SQLite est :
- Zero-config (un fichier)
- Suffisant pour notre échelle (~10k entrées max)
- Thread-safe avec WAL mode
- Pas de service séparé à gérer

PostgreSQL serait justifié si : multi-instance backend (besoin de lock distribué), volumétrie > 100k entrées, ou besoin de fonctionnalités avancées (full-text search, JSON queries complexes).

### Q16.5 Pourquoi React et pas Vue/Svelte ?
**R :** Écosystème React le plus mature pour les Web Components (model-viewer). Hooks pattern bien maîtrisé. TypeScript intégration excellente. Vue 3 aurait été équivalent. Svelte plus performant mais écosystème plus petit.

### Q16.6 Pourquoi un cache vectoriel et pas un cache Redis simple par hash d'image ?
**R :** Cache par hash exact : seules les images strictement identiques (bit-pour-bit) seraient des hits. Notre cas : un utilisateur ré-upload la même image légèrement compressée → hash différent → cache miss inutile. L'embedding capture la similarité sémantique/visuelle, plus tolérant.

### Q16.7 Pourquoi pas un message broker comme Kafka ?
**R :** Kafka est pour le streaming d'événements à haut débit (millions/s). Notre cas : quelques tâches/heure. Redis comme broker Celery suffit largement.

---

## 17. Limitations connues

### Q17.1 Quelles sont les limitations actuelles ?

**Côté ML :**
- Génération 3D requiert GPU (ou Apple Silicon avec MPS)
- 20 min pour les modes qualité (long pour un utilisateur)
- Modèle Hunyuan3D contraint à 256 octree max (bug FlashVDM)
- Texture multi-vues IA peut être incorrecte (notre custom substitution adresse partiellement)

**Côté architecture :**
- 1 GPU = 1 tâche à la fois (pas de batching)
- LangGraph resume depuis le début du sous-graphe (pas plus granulaire)
- WebSocket peut échouer derrière certains proxies (fallback polling)
- Cancellation SIGTERM peut prendre 30s pendant l'inférence CUDA

**Côté frontend :**
- Pas d'authentification utilisateur
- Pas de versioning de modèle (un seul modèle Hunyuan)
- Pas de progress bar fine pour la diffusion (juste start/end actuellement)
- Upload de fichier sans vraie progress (juste un simulé)

**Côté tests :**
- Tests unitaires absents (juste un smoke test)
- Pas de tests E2E
- Pas de tests visuels pour valider la qualité 3D

### Q17.2 Comment adresseriez-vous ces limitations ?

| Limitation | Solution future |
|---|---|
| 20 min génération | Distillation modèle, batch inference, GPU plus puissant |
| Pas d'auth | OAuth2 (JWT) via fastapi-users |
| Progress fine ML | Hook `callback_steps` du DiT (déjà supporté par hy3dgen) |
| Pas de tests E2E | Playwright pour le frontend, pytest-asyncio pour le backend |
| Pas de monitoring | Prometheus + Grafana + Loki pour logs |
| 1 GPU max | Multi-worker Celery, queue par GPU |

---

## 18. Questions pièges & comment répondre

### Q18.1 "Vous parlez d'agent IA mais c'est juste une chaîne avec retry, non ?"
**R :** Vrai, ce n'est pas un agent au sens "LLM qui décide ses actions". C'est un **workflow à état dirigé**. Le terme "agentique" vient du framework LangGraph qui le pose ainsi. Pour un vrai agent (planificateur autonome), il faudrait un nœud LLM qui choisit la prochaine action — pas pertinent ici car le pipeline est déterministe (parse → spec → mesh). LangGraph apporte le retry/checkpointing/streaming, pas l'autonomie décisionnelle.

### Q18.2 "Pourquoi utiliser Celery ET LangGraph ? C'est redondant."
**R :** Non, ils répondent à des préoccupations orthogonales (cf. tableau Celery↔LangGraph dans le rapport).
- **Celery** : infrastructure (queue, isolation processus, distribution worker)
- **LangGraph** : workflow (graphe, état, retry, checkpoint)

Sans Celery, FastAPI exécuterait LangGraph en process → un OOM tue tout. Sans LangGraph, Celery exécuterait du code Python brut avec retry manuel — plus de lignes pour moins de garanties.

### Q18.3 "Le mode multi-vues est juste un wrapper Hunyuan3D, où est votre contribution ?"
**R :** La substitution de texture utilisateur est entièrement custom (~400 LOC dans `texgen/pipelines.py`). Hunyuan3D upstream ne sait pas substituer des photos réelles dans les vues IA générées. Notre contribution :
- Re-centrage du mesh sur médiane des vertices
- Détection du boîtier via HSV + morpho closing + convex hull
- target_case_size depuis la vue IA (pas la photo utilisateur)
- Stratégies de fallback (mirror, couleur médiane)

Vérifié : tableau de comparaison dans le rapport, aucun concurrent ne fait ça.

### Q18.4 "Pourquoi pas un seul mode unifié au lieu de 4 ?"
**R :** Chaque mode est un workflow distinct techniquement :
- Image→3D : 1 pipeline (i23d_pipeline)
- Texte→3D : 2 pipelines en cascade (t2i puis i23d)
- Multi-vues→3D : pipeline différent (mv_pipeline 1.1B, pas i23d 0.6B)
- Document→3D : LangGraph + LLM + i23d via wrapper text_to_3d

Les fusionner derrière une API unique masquerait des paramètres importants (multi-vues a besoin de plusieurs images, document a besoin du PDF, etc.).

### Q18.5 "Votre système est-il production-ready ?"
**R :** Pour un usage interne ou démo : oui. Pour production publique : il manque (priorité décroissante) :
1. Authentification (OAuth2)
2. Rate limiting (slowapi/Nginx)
3. Observabilité (Prometheus, Sentry, structured logs)
4. Tests d'intégration GPU
5. CI/CD (GitHub Actions)
6. Backup automatisé (gallery, vector store, checkpoints)
7. Sécurité headers (HSTS, CSP via Nginx)

Le cœur (LangGraph + Celery + Hunyuan3D) est solide ; ce qui manque est de la "production hardening" plutôt que des changements architecturaux.

### Q18.6 "Avez-vous testé votre cancel mid-CUDA ?"
**R :** Manuellement, oui. SIGTERM est queue par Python mais ne préempte pas une c-extension en cours. Sur une diffusion de 30 steps, le SIGTERM est honoré au prochain step Python — donc retard de quelques secondes acceptable. Le worker meurt proprement, le pool en relance un (~30-60s pour recharger Hunyuan). Documenté dans le rapport comme limitation acceptable.

### Q18.7 "Comment vous différenciez-vous de Meshy ?"
**R :** Meshy est SaaS payant, cloud only, pas de PDF→3D, pas d'intégration Unity native, pas de cache vectoriel local. Nous sommes : self-hosted, gratuit, intégration Unity sans plugin, automatisation B2B (PDF → asset), confidentialité (pas de upload chez un tiers).

### Q18.8 "Le LLM Llama-3 8B est-il vraiment suffisant pour l'extraction structurée ?"
**R :** Pour des spécifications simples (nom, dimensions, matériau, couleur) : oui, le taux de validation Pydantic au premier essai est ~85-90%. Pour les 10-15% d'échec, on retry (max 3) puis fallback. Pour des extractions plus complexes (sous-composants, relations spatiales), il faudrait un modèle plus gros (70B) ou GPT-4. Architecture LangGraph permet de swap facilement.

### Q18.9 "Pourquoi le projet a pris autant de commits récents sur la même fonctionnalité ?"
**R :** La substitution multi-vues a nécessité ~15 itérations (warping per-row → shear → piecewise shear → composite → reverted). Chaque tentative semblait raisonnable a priori mais introduisait des régressions visibles (patchwork de textures). La conclusion finale (re-centrer le mesh + target_case_size depuis vue IA + ne pas warper) est contre-intuitive : il a fallu épuiser les approches naïves pour s'en convaincre. Documenté dans la mémoire projet (`project_texgen_user_view_substitution.md`).

### Q18.10 "Si vous deviez refaire le projet, qu'est-ce que vous changeriez ?"
**R :** Trois choses :
1. **Commencer par Celery dès le début** : la migration `threading.Thread` → Celery a été lourde car le code dépendait des dicts en mémoire FastAPI. Architectures avec Celery natif depuis le début = pas cette dette technique.
2. **Définir l'état LangGraph plus strictement (Pydantic au lieu de TypedDict)** : validation à l'écriture éviterait des champs morts comme `current_step`.
3. **Tester en CI avec un mock-GPU dès le début** : aurait détecté beaucoup de régressions plus tôt. Le smoke test arrive tard.

### Q18.11 "Quelles compétences avez-vous développées sur ce projet ?"
**R :**
- **ML inference engineering** : déploiement modèles diffusion, gestion mémoire GPU/MPS, debug bugs CUDA
- **Orchestration distribuée** : Celery production-grade, LangGraph state machine
- **Architecture full-stack** : FastAPI async, React 19 hooks, WebSocket bidir
- **3D pipeline** : UV mapping, multi-view bake, mesh post-processing (Trimesh)
- **DevOps** : Docker multi-stage, gestion de modèles ML lourds, observabilité
- **Debugging itératif** : approche par hypothèses + vérifications empiriques (vu sur le multiview)

### Q18.12 "Quel est le risque le plus critique de votre architecture ?"
**R :** Le **single GPU** comme single point of failure. Si la machine GPU tombe, plus aucune génération possible. En production, il faudrait : (a) pool de workers GPU sur N machines avec health-check, (b) queue Redis HA (Sentinel), (c) failover automatique. Architecture supporte mais pas déployée en redondance.

---

## Annexes utiles

### A. Glossaire à connaître

| Terme | Définition courte |
|---|---|
| DiT | Diffusion Transformer — modèle de diffusion utilisant un Transformer |
| VAE | Variational Autoencoder — compresse/décompresse en latents |
| Octree | Structure 3D hiérarchique (subdivision spatiale) |
| Marching Cubes | Algorithme d'extraction d'isosurface depuis un champ scalaire 3D |
| UV mapping | Coordonnées 2D associées aux vertices pour appliquer une texture |
| Back projection | Projection d'une image 2D sur une surface 3D via les UV |
| Reducer (LangGraph) | Fonction de combinaison d'updates parallèles |
| Checkpointer | Mécanisme de persistance d'état pour reprise |
| Subgraph | Graphe LangGraph compilé utilisé comme nœud dans un parent |
| Ack-late (Celery) | Acknowledge la tâche après succès, pas avant |
| Prefetch (Celery) | Nombre de tâches assignées d'avance à un worker |
| Revoke (Celery) | Annulation d'une tâche en file ou en cours |

### B. Commandes utiles pour démontrer

```bash
# Démarrer la stack complète
make dev-v2

# Lancer le smoke test
cd Backend && python -m tests.smoke_test

# Inspecter une tâche Celery en cours
redis-cli -p 9501
> KEYS celery-task-meta-*

# Voir les checkpoints LangGraph
sqlite3 Backend/generated/pipeline_checkpoints.db
> .tables
> SELECT thread_id, COUNT(*) FROM checkpoints GROUP BY thread_id;

# Statistiques cache vectoriel
curl http://localhost:8001/api/v1/cache-stats
```

### C. Fichiers à connaître pour pointer pendant la soutenance

| Fichier | Responsabilité |
|---|---|
| `Backend/app/pipeline/graph.py` | Topologie LangGraph + helpers |
| `Backend/app/pipeline/nodes.py` | Implémentation des nœuds + timeouts |
| `Backend/app/pipeline/state.py` | Schéma d'état typé |
| `Backend/app/tasks.py` | Tâche Celery `run_pipeline` (streaming) |
| `Backend/app/tasks_3d.py` | Tâches Celery GPU (4 modes) |
| `Backend/app/worker.py` | Config Celery production |
| `Backend/app/api/routes_3d.py` | Endpoints 3D + WebSocket |
| `Backend/app/api/routes.py` | LangGraph endpoints + task status |
| `Backend/app/services/hunyuan3d_service.py` | Wrapper Hunyuan3D |
| `Backend/hy3dgen/texgen/pipelines.py` | Substitution multi-vues custom |
| `Frontend/hooks/useTaskPolling.ts` | Hook polling avec backoff |
| `Frontend/components/TextExtractor.tsx` | UI document→3D |
| `UnityProject/Assets/Editor/SpawnBridge.cs` | Pont Unity ↔ Backend |

---

**Bonne soutenance !**

---
---

# PARTIE II — Questions complémentaires & approfondissements

## 19. Patterns d'architecture & design decisions

### Q19.1 Quels design patterns avez-vous utilisés ?
**R :**
- **Singleton** : `get_hunyuan3d()` retourne toujours la même instance du service (modèle ML chargé une fois)
- **Strategy** : `Hunyuan3DPaintPipeline.merge_method = 'fast'` (vs 'graphcut') — plug d'algorithme de bake
- **Factory** : `build_pipeline(interrupt_after=...)` retourne un graphe compilé selon les options
- **Observer/Callback** : `on_event(node_name, state)` callback dans `run_pipeline_streaming`
- **State Machine** : LangGraph est intrinsèquement un state machine pattern
- **Repository** : `gallery_db` (et le futur `pipeline_stats_db`) encapsulent l'accès SQLite
- **Adapter** : `tasks_3d.py` adapte l'interface synchrone Hunyuan vers l'interface async Celery
- **Context Manager** : `_node_timeout(seconds, label)` est un context manager pour les timeouts

### Q19.2 Avez-vous appliqué les principes SOLID ?
**R :**
- **S** (Single Responsibility) : chaque module a une responsabilité claire (`worker.py` = config Celery, `gallery_db.py` = persistance, `tasks_3d.py` = tâches GPU)
- **O** (Open/Closed) : `_AZIM_TO_VIEW` dict permet d'ajouter de nouvelles vues sans modifier le code de substitution
- **L** (Liskov) : peu de héritage dans le projet
- **I** (Interface Segregation) : Pydantic models par endpoint (`ImageTo3DRequest` ≠ `TextTo3DRequest`)
- **D** (Dependency Inversion) : services injectés via `get_*()` lazy, pas instanciés en module-top

### Q19.3 Comment gérez-vous les imports circulaires ?
**R :** Imports différés (lazy) à l'intérieur des fonctions plutôt qu'au top du module. Exemple dans `nodes.py` :
```python
def extract_spec_llm_node(state):
    from app.services.llm_service import get_llm_service  # lazy
    ...
```
Évite que l'import de `nodes.py` charge tout l'arbre de services au démarrage.

### Q19.4 Comment isolez-vous les couches ?
**R :** Architecture en 4 couches :
1. **API** (`app/api/`) — uniquement routes FastAPI, pas de logique métier
2. **Pipeline** (`app/pipeline/`) — orchestration LangGraph, pas d'accès direct aux services lourds (lazy import)
3. **Services** (`app/services/`) — logique métier (LLM, hunyuan, document parser, vector store)
4. **Models** (`app/models/`) — Pydantic models pour le typage

Les routes ne devraient jamais importer directement Hunyuan ; elles passent par les services (ou Celery tasks).

### Q19.5 Pourquoi tasks.py et tasks_3d.py séparés ?
**R :** Séparation des préoccupations :
- `tasks.py` : workflow LangGraph (pipeline document→3D)
- `tasks_3d.py` : tâches GPU directes (4 modes)

Aurait pu être dans le même fichier, mais 100+ LOC chacun = mieux séparés. Le `task_routes` Celery les inclut tous deux.

### Q19.6 Comment communiquez-vous entre workers Celery ?
**R :** Pas de communication directe entre workers. Chaque tâche est indépendante. Si une tâche A doit déclencher B, soit :
- A retourne, B est enqueue séparément (chaining via Celery `chord` ou `chain`)
- A appelle `celery_app.send_task("B", ...)` directement
- A et B partagent un état via Redis/SQLite

Dans notre cas, le pipeline LangGraph est une seule tâche Celery (`run_pipeline`) qui orchestre les sous-étapes en-process via LangGraph.

### Q19.7 Pourquoi pas une seule classe `Generator` avec une méthode `generate(mode)` ?
**R :** Considéré et rejeté. Les 4 modes ont des signatures différentes (image, text, views dict, source uid), des validations différentes, des dépendances de pipelines différentes (i23d_pipeline vs mv_pipeline). Une méthode unique avec `**kwargs` perdrait le typage et les validations. Les méthodes séparées dans `Hunyuan3DService` sont plus claires et autodocumentées.

### Q19.8 Comment ajouteriez-vous un 5e mode (ex : video-to-3D) ?
**R :**
1. Nouvelle méthode `Hunyuan3DService.video_to_3d(video_b64, ...)` (qui appellerait un nouveau pipeline ML)
2. Nouveau Pydantic model `VideoTo3DRequest`
3. Nouvelle Celery task `video_to_3d_task` dans `tasks_3d.py`
4. Nouveau endpoint `/api/v1/video-to-3d/async` dans `routes_3d.py`
5. Ajouter `"app.tasks_3d.video_to_3d_task"` au `task_routes` du `worker.py`
6. Nouveau composant frontend `VideoTo3D.tsx`

Architecture extensible : pas de refactor des autres modes.

---

## 20. Algorithmes & mathématiques

### Q20.1 Comment fonctionne mathématiquement la diffusion ?
**R :** Modèle inverse d'un processus de diffusion :
- **Forward** : on ajoute progressivement du bruit gaussien à une donnée (image latent, ici 3D latent) sur T steps. Au step T, c'est du bruit pur.
- **Reverse** : on entraîne un réseau (DiT) à prédire le bruit ajouté à chaque step. À l'inférence, on part du bruit pur et on débruit step by step.
- **Conditionnement** : on injecte un signal (image, texte) via attention cross pour guider le débruitage.
- **Classifier-Free Guidance** : `guidance_scale > 1` mélange prédiction conditionnée et non-conditionnée pour amplifier l'effet du conditionnement.

### Q20.2 Pourquoi `guidance_scale=5` par défaut ?
**R :** Compromis qualité/diversité :
- `guidance_scale=1` : pas de guidance, génération diverse mais peu fidèle au prompt
- `guidance_scale=7.5+` : très fidèle mais oversaturé, artefacts
- `5` : sweet spot pour Hunyuan3D selon le papier

### Q20.3 Combien de paramètres a votre DiT ?
**R :**
- `hunyuan3d-dit-v2-mini` : ~600M paramètres
- `hunyuan3d-dit-v2-mv` : ~1.1B paramètres (multi-vues)
- DiT paint : ~1.3B paramètres (texture)

Comparativement, Stable Diffusion 1.5 = 860M, SDXL = 2.6B.

### Q20.4 Qu'est-ce que la projection orthographique ?
**R :** Projection 3D→2D sans perspective (lignes parallèles restent parallèles). Matrice 4x4 :
```
[2/(R-L), 0,        0,         -(R+L)/(R-L)]
[0,       2/(T-B),  0,         -(T+B)/(T-B)]
[0,       0,       -2/(F-N),   -(F+N)/(F-N)]
[0,       0,        0,          1          ]
```
Avec `ortho_scale=1.2` : window [-0.6, 0.6] × [-0.6, 0.6]. Tout vertex `(x, y, z)` avec `|x|, |y| ≤ 0.6` est visible.

### Q20.5 Pourquoi orthographique et pas perspective pour le rendu multi-vues ?
**R :** Le modèle multi-vues est entraîné en orthographique (convention Hunyuan). Avantages :
- Pas de distorsion de perspective (textures plus uniformes)
- Calcul UV plus simple (mappage linéaire)
- Convention partagée par tous les modèles de génération multi-vues récents (Zero123++, MVDream)

### Q20.6 Comment fonctionne `set_mesh` ?
**R :** Trois étapes :
1. **Transformation de coordonnées** : `(x, y, z) → (-x, z, -y)` (négation X+Y, swap Y+Z). Aligne le repère hy3dgen sur celui attendu par le renderer.
2. **Auto-centering** : translate par `-bbox_center` pour mettre le centre de la bounding box à l'origine
3. **Normalization** : scale par `scale_factor / bounding_sphere_diameter` (avec `scale_factor=1.15`). Place les vertices dans un range cohérent pour le rendu.

### Q20.7 Comment marching cubes extrait le mesh ?
**R :** Le VAE 3D produit un champ d'occupation 3D (grille voxel). Marching Cubes parcourt chaque voxel, regarde les 8 coins (dedans/dehors selon un seuil), et selon les 256 configurations possibles, génère 0 à 5 triangles approximant l'isosurface dans ce voxel. Tous les triangles agrégés = le mesh.

### Q20.8 Qu'est-ce que face_reducer (quadric mesh simplification) ?
**R :** Algorithme de Garland-Heckbert. Pour chaque arête, calcule un coût de collapse (combien la forme change si on fusionne les deux vertices). Collapse les arêtes de plus faible coût itérativement jusqu'à atteindre le `face_count` cible. Préserve les détails saillants (arêtes vives).

### Q20.9 Comment fonctionne UV unwrap ?
**R :** Mapping inverse de la surface 3D vers un plan 2D. Algos : 
- **Conformal** (préserve les angles, distort les aires)
- **ARAP** (As Rigid As Possible — préserve les distances locales)
- **Smart UV Project** (auto-couture par groupes de faces co-planaires)

hy3dgen utilise `mesh_uv_wrap` qui produit un atlas UV exploitable pour la texture.

### Q20.10 Pourquoi `bake_exp=4` ?
**R :** Dans le bake multi-vues : `weighted_cos = view_weight × cos(angle_camera_normal)^bake_exp`. L'exposant amplifie la préférence pour les vues frontales (angle 0 = cos 1 = poids 1, angle 60° = cos 0.5 = poids 0.5^4 = 0.0625). Évite les blurs aux angles obliques où la précision UV est faible.

### Q20.11 Qu'est-ce que cv2.inpaint Navier-Stokes ?
**R :** Algorithme d'inpainting basé sur la PDE de Navier-Stokes (fluides). Propage les pixels environnants dans la zone à remplir en suivant les "courbes de niveau" (isolignes d'intensité). Préserve les bords et continuités. Utilisé dans `texture_inpaint` pour combler les pixels UV non visibles depuis aucune caméra.

### Q20.12 Mathématiquement, qu'est-ce que la convex hull du masque case ?
**R :** Plus petit polygone convexe contenant tous les points du masque. Algos : Quickhull O(n log n) ou Graham scan O(n log n). En OpenCV : `cv2.convexHull(contour)`. On l'utilise pour englober le boîtier (qui peut avoir des creux dus aux gravures) avec un polygone unique.

---

## 21. GPU & gestion mémoire

### Q21.1 Combien de VRAM consomme votre système ?
**R :** Approximativement :
- `hunyuan3d-dit-v2-mini` (FP32) : ~2.4 GB
- `hunyuan3d-paint-v2-0` : ~5 GB
- `hunyuan3d-mv` : ~4 GB
- `Hyper-SDXL` : ~10 GB (FP32) ou ~5 GB (FP16)
- Activations runtime : 1-3 GB selon batch + resolution

Total chargé simultanément : insurmontable sur 16 GB MPS Apple. D'où le **offload manuel CPU↔GPU**.

### Q21.2 Comment fonctionne l'offload ?
**R :** `DeviceManager._offload_to_cpu("pipeline_name")` :
1. Déplace tous les tenseurs du pipeline sur `device='cpu'`
2. Libère la VRAM (forcé via `gc.collect()` + `torch.mps.empty_cache()`)
3. Quand on en a besoin : `_move_to_device("pipeline_name")` pour réinjecter sur GPU

Cycle classique : i23d → offload(i23d) → texgen → offload(texgen) → i23d (next request).

### Q21.3 MPS vs CUDA ?
**R :**
- **MPS** (Metal Performance Shaders) : Apple Silicon (M1/M2/M3). Lent vs CUDA mais pratique en dev.
- **CUDA** : NVIDIA. Beaucoup plus rapide (Tensor Cores, plus de SMs).

Notre code détecte le device via `get_device()` qui renvoie `cuda > mps > cpu` dans cet ordre.

### Q21.4 Pourquoi `torch.inference_mode()` et pas `torch.no_grad()` ?
**R :** `inference_mode()` est plus strict :
- Désactive autograd (comme `no_grad()`)
- Marque les tenseurs créés comme "inference" — ils ne peuvent JAMAIS être utilisés en training (catch d'erreurs early)
- Évite certaines opérations de bookkeeping → ~5-10% plus rapide en pure inference

Recommandé en prod pour des workloads inference-only.

### Q21.5 Mixed precision (FP16) ?
**R :** Pas utilisé dans cette version. Hunyuan3D peut tourner en FP16 (`torch_dtype=torch.float16`) pour :
- 2x moins de VRAM
- 1.5-2x plus rapide
- Légère perte de qualité (souvent imperceptible)

Pas activé par défaut pour préserver la qualité. Activable via une option future.

### Q21.6 Comment vous gérez les fuites mémoire ?
**R :**
- `empty_cache()` après chaque génération (`torch.mps.empty_cache()` ou `torch.cuda.empty_cache()`)
- `del` explicite sur les gros tenseurs intermédiaires (mesh, outputs)
- `gc.collect()` périodique
- Sur MPS, bug connu de fragmentation : un restart périodique du worker peut être nécessaire

### Q21.7 Quel est le risque OOM ?
**R :** Si on essaie de charger 2 pipelines simultanément (i23d + texgen) sur 16 GB MPS : OOM. C'est pourquoi `_offload_to_cpu` est appelé entre les étapes. Sur GPU 24 GB : pas de problème, on peut tout garder en VRAM.

### Q21.8 Comment debug un OOM ?
**R :**
- `torch.mps.current_allocated_memory()` / `torch.cuda.memory_allocated()` pour mesurer
- `torch.cuda.memory_summary()` pour le détail par tenseur (CUDA only)
- Réduire `octree_resolution` ou `num_chunks` pour diminuer la mémoire peak
- Activer mixed precision (FP16)

---

## 22. Hunyuan3D internals (plus profond)

### Q22.1 Quelle est l'architecture exacte du DiT Hunyuan3D ?
**R :** Transformer avec :
- Patches 3D latents en entrée (compressés par le VAE)
- Multi-head self-attention + cross-attention pour le conditionnement (CLIP image features)
- Couches MLP avec activation GELU
- Time embedding (sinusoidal) pour conditionner sur le step de diffusion
- Output : prediction du bruit ajouté

Variante "Flow Matching" plutôt que diffusion classique (DDPM) → moins de steps nécessaires.

### Q22.2 Qu'est-ce que Flow Matching ?
**R :** Alternative à DDPM. Apprend un champ de vélocité qui transporte du bruit gaussien vers la distribution cible en T pas. Avantages : peut converger en 5-30 steps (vs 50-100 pour DDPM), entraînement plus stable. C'est pourquoi notre default `steps=5-30` fonctionne.

### Q22.3 Pourquoi 6 vues pour le multi-view diffusion ?
**R :** 6 vues = 4 azimuths (0°, 90°, 180°, 270°) à élévation 0 + 1 vue top (élév +90°) + 1 vue bottom (élév -90°). Couvre la sphère unitaire avec un nombre raisonnable de samples. Pourrait être plus (12, 24) mais : (a) plus lent, (b) le UV unwrap n'a pas besoin de plus pour couvrir la surface.

### Q22.4 Comment le modèle paint génère les multi-vues ?
**R :** Pipeline `Hunyuan3DPaintPipeline` :
1. Input : image delight (sans éclairage) + normal maps + position maps des 6 vues
2. Modèle de diffusion conditionné sur ces 3 inputs concatenés (via cross-attention)
3. Génère 6 images RGB en parallèle (architecture multi-vues consistente)
4. Sortie : 6 images texturées cohérentes (même éclairage, mêmes matériaux)

### Q22.5 Qu'est-ce que delight et pourquoi c'est important ?
**R :** Le modèle `hunyuan3d-delight-v2-0` retire l'éclairage de l'image source pour obtenir l'**albedo** (couleur de matériau brute). Si on ne le faisait pas, le mesh aurait l'éclairage gravé dans sa texture (un highlight blanc sur le boîtier resterait blanc sous toute lumière). Avec delight : la texture est neutre, le rendu Unity peut appliquer son propre éclairage PBR.

### Q22.6 Quel est le pipeline back_project ?
**R :** Pour chaque vue caméra :
1. Rasterize le mesh sous cette vue (rasterizer custom CUDA dans `custom_rasterizer/`)
2. Pour chaque pixel canvas du mesh, calcule la coordonnée UV correspondante
3. Lit la couleur du pixel multi-vues à cette position canvas
4. Écrit cette couleur à la coordonnée UV dans la texture map

Inverse : `(canvas_x, canvas_y) → (uv_u, uv_v) → couleur`.

### Q22.7 Pourquoi un rasterizer custom ?
**R :** Les rasterizers généralistes (nvdiffrast, PyTorch3D) ont des limitations sur MPS Apple. Le custom rasterizer (`hy3dgen/texgen/custom_rasterizer/`) est écrit en C++ avec bindings Python, fonctionne sur CPU/CUDA (et MPS via fallback CPU). Convention X/Y spécifique : `pixel_y = (0.5 + 0.5 * clip_Y) * (height-1)` (Y-flip implicite).

### Q22.8 Quel est le format GLB final ?
**R :** **glTF 2.0 Binary** (GLB) :
- Mesh : vertices, faces (triangles), normales (si disponibles), UVs
- Materials : PBR (base color, normal, metalness, roughness — pas tous remplis)
- Texture : embedded base64 PNG
- Hierarchy : scene → node → mesh
- Format binaire (vs glTF JSON+textures séparés) : un seul fichier

trimesh export gère tout : `mesh.export("file.glb")`.

---

## 23. Frontend deep-dive

### Q23.1 Quelle est la taille de votre bundle JS ?
**R :** Production build Vite : ~250-400 KB gzipped (sans model-viewer qui est lazy-loaded). Largement acceptable pour une SPA.

### Q23.2 Comment optimiseriez-vous le bundle ?
**R :**
- Code splitting par route (pas implémenté car SPA simple)
- Lazy load model-viewer (~80 KB) : `import("@google/model-viewer")` dynamiquement
- Tree-shaking automatique via Vite/Rollup
- Compression Brotli en production (Nginx config)

### Q23.3 Quelle est l'accessibilité (a11y) ?
**R :** Basique :
- Boutons avec `aria-label`
- Forms avec `<label htmlFor>`
- Couleurs avec contrast ratio ≥ 4.5:1 (vérifié)
- Pas testé avec screen reader

Pour production : audit Lighthouse a11y, tests avec NVDA/VoiceOver.

### Q23.4 SEO ?
**R :** Pas applicable (SPA derrière login, pas de contenu indexable). Pour production publique : SSR (Next.js) ou meta tags via React Helmet.

### Q23.5 Comment gérez-vous les erreurs réseau ?
**R :** `fetch()` wrappé dans try/catch, message d'erreur dans un toast. Pas de retry automatique (sauf pour le polling qui re-essaye au prochain tick). Pour production : intercepteur global avec retry exponentiel + offline detection.

### Q23.6 Why no React Query / SWR ?
**R :** Considéré. Notre cas : peu de fetch (1-2 par mode), pas de cache complexe à gérer, polling déjà custom (useTaskPolling). React Query ajouterait ~30 KB pour peu de valeur. Si l'app grossissait (50+ endpoints) : justifié.

### Q23.7 Comment fonctionne `<model-viewer>` ?
**R :** Web Component de Google. Wrapper autour de three.js + WebGL. Charge un GLB via attribut `src`, gère l'orbit camera, le PBR rendering, l'IBL (Image-Based Lighting). Auto-rotation possible. Une lib mature et performante, alternative à three.js raw qui demanderait 200+ lignes de boilerplate.

### Q23.8 Pourquoi pas Three.js direct ?
**R :** Plus de contrôle mais beaucoup plus de code. Pour un viewer simple (orbit + zoom + auto-rotate), `<model-viewer>` est déclaratif et déjà accessible. Si on voulait des features custom (annotations, mesh edit, scene complexe), Three.js direct + react-three-fiber serait justifié.

### Q23.9 Comment gérez-vous les images uploadées dans le navigateur ?
**R :** `FileReader.readAsDataURL()` pour base64 (envoyé tel quel au backend), `URL.createObjectURL()` pour preview avant upload. Cleanup via `URL.revokeObjectURL()` pour éviter les fuites mémoire DOM.

### Q23.10 Comment évitez-vous les re-renders inutiles ?
**R :**
- `useState` localisé (pas remonté au parent si pas nécessaire)
- `useCallback` pour les handlers passés à des enfants memoized
- `useMemo` pour les calculs coûteux (rarement nécessaire ici)
- Pas de React.memo systématique (over-engineering pour notre échelle)

---

## 24. Base de données & persistance

### Q24.1 Pourquoi SQLite et pas une vraie BDD ?
**R :** Cf. Q16.4. Pour récap : zero-config, suffisant en volume, threadsafe avec WAL. Trois SQLite distincts :
1. `gallery.db` — modèles 3D générés
2. `pipeline_checkpoints.db` — checkpoints LangGraph
3. ChromaDB (utilise SQLite interne) — cache vectoriel

### Q24.2 Qu'est-ce que WAL mode ?
**R :** Write-Ahead Logging. Mode SQLite qui :
- Permet plusieurs readers concurrent à un writer
- Écritures dans un journal séparé (`.db-wal`), puis batch-flush vers le fichier principal
- Réduit les conflits lock vs mode rollback journal classique

Activé via `PRAGMA journal_mode=WAL`. Indispensable pour notre cas multi-thread.

### Q24.3 Comment gérez-vous les transactions ?
**R :** Implicite via `conn.commit()`. Pour gallery_db, chaque `insert/delete/update` est sa propre transaction (autocommit-like). Pas de transactions multi-statement actuellement — pas nécessaire (opérations atomiques par UID).

### Q24.4 Risque de corruption ?
**R :** SQLite corruption rare (un des SGBDR les plus robustes). Causes possibles :
- Crash matériel pendant l'écriture (WAL atténue ce risque)
- Bug filesystem (rare sur ext4/APFS)
- Modification concurrente sans lock (impossible avec notre `threading.Lock`)

Backup recommandé en production : copie du fichier `.db` après `VACUUM`.

### Q24.5 Comment migrez-vous le schéma ?
**R :** `_MIGRATE_SQL = "ALTER TABLE models ADD COLUMN has_texture INTEGER DEFAULT 0;"` dans `_init()`. SQLite supporte `ALTER TABLE ADD COLUMN`. Pour des migrations plus complexes : Alembic (overkill ici), ou script manuel.

### Q24.6 Pourquoi `_lock` global ?
**R :** SQLite est threadsafe mais une seule écriture à la fois. Le lock évite les `database is locked` errors lors d'écritures concurrentes depuis plusieurs threads (Celery workers, WebSocket handlers). Lecture sans lock (SELECT) car concurrent OK avec WAL.

### Q24.7 Combien d'entrées dans la galerie ?
**R :** Pas de limite hard. SQLite gère facilement 1M+ rows. Notre cas réaliste : 100-10000 modèles. Index sur `created_at DESC` pour les listings rapides.

### Q24.8 Comment supprimer un modèle ?
**R :** Endpoint `DELETE /models/{uid}` :
1. Validation regex sur UID
2. `gallery_db.delete(uid)` — supprime la row
3. Suppression du fichier GLB sur disque (`Path.unlink()`)
4. Suppression de l'entrée du cache vectoriel (si présente)

---

## 25. Networking & protocoles

### Q25.1 HTTP/1.1, HTTP/2, ou HTTP/3 ?
**R :** Uvicorn supporte HTTP/1.1 par défaut. HTTP/2 via `--http h2` (uvicorn + httptools). En production derrière Nginx : Nginx parle HTTP/2 ou HTTP/3 (QUIC) côté client, HTTP/1.1 côté backend. Pas critique pour notre cas (peu de requêtes parallèles).

### Q25.2 Comment fonctionne le WebSocket upgrade ?
**R :**
1. Client envoie HTTP GET avec headers `Upgrade: websocket` + `Connection: Upgrade` + `Sec-WebSocket-Key`
2. Serveur répond `101 Switching Protocols` avec `Sec-WebSocket-Accept` (hash du key)
3. Le socket TCP passe en mode framing WebSocket
4. Échanges full-duplex de frames texte/binaire jusqu'à close

FastAPI gère ça via Starlette's `WebSocket` class.

### Q25.3 Pourquoi gzip n'est pas activé sur l'API ?
**R :** Devrait être ajouté (`GZipMiddleware`). Pour notre cas (JSON responses < 10 KB), peu d'impact. Pour gros payloads (mesh stats listing 1000+ models), gzip réduirait ~70%.

### Q25.4 Comment fonctionne l'upload multipart ?
**R :** Browser encode :
```
POST /upload HTTP/1.1
Content-Type: multipart/form-data; boundary=----WebKitFormBoundary...

------WebKitFormBoundary
Content-Disposition: form-data; name="file"; filename="doc.pdf"
Content-Type: application/pdf

<binary bytes>
------WebKitFormBoundary--
```
FastAPI parse via Starlette's multipart parser, expose `UploadFile` avec `.read()`, `.filename`, `.content_type`.

### Q25.5 Pourquoi pas de gRPC ?
**R :** Pas de cas d'usage : on a un seul client (le frontend React) qui parle HTTP/JSON. gRPC serait justifié pour : (a) micro-services communicant entre eux à haut débit, (b) clients dans plusieurs langages partageant des protobuf schemas. Notre architecture : monolithe Python backend + SPA → REST + WebSocket suffit.

### Q25.6 Comment évitez-vous les Cross-Site Request Forgery (CSRF) ?
**R :** Pas de cookies session (pas d'auth). Les endpoints prennent uniquement des inputs JSON via fetch — un attacker ne peut pas forger une requête depuis un autre site sans CORS approuvé. CORS whitelist + pas de credentials = pas de CSRF.

### Q25.7 Étapes du chargement initial du frontend ?
**R :**
1. Browser fetch `/` (Nginx sert `index.html`)
2. Parse HTML, fetch CSS + JS bundles (Vite outputs versionnés)
3. Hydrate React (mount `<App />` dans `<div id="root">`)
4. App init : load theme from localStorage, fetch gallery `/api/v1/models`
5. Render UI

Total : 100-500 ms en local, ~1s sur réseau lent.

---

## 26. Error handling & resilience

### Q26.1 Philosophie de gestion d'erreur ?
**R :** "Échouer gracieusement". Trois niveaux :
1. **Validation early** (Pydantic, regex) : rejeter les inputs invalides en 400 immédiatement
2. **Try/except autour des appels externes** (ML, HTTP, DB) : logguer, retourner une erreur structurée
3. **Fallback** : si possible, dégrader (ex : `_get_checkpointer` retourne None si SqliteSaver fail → pipeline marche sans checkpoint)

Jamais d'unhandled exception qui crash le serveur.

### Q26.2 Que se passe-t-il si rembg crash ?
**R :** Vérifié dans `multiview_to_3d` :
```python
if t_frac > 0.95 or o_frac > 0.95:
    logger.warning("MV view '%s' failed rembg ... — skipping", view_name)
    continue
```
Si rembg produit une image complètement transparente ou opaque, on saute cette vue. Si toutes les vues échouent : exception remontée au client.

### Q26.3 Et si le modèle ML est corrompu au chargement ?
**R :** `Hunyuan3DService.__init__` charge les modèles en try/except. Si échec : log, attribut `has_t2i/has_mv/has_texgen` à False. Les endpoints vérifient (`if not svc.has_t2i: raise 503`). Frontend affiche un message "feature unavailable" plutôt que crasher.

### Q26.4 Comment fonctionne la dégradation gracieuse du checkpointer ?
**R :** `_get_checkpointer()` tente `SqliteSaver(conn)`. Si :
- LangGraph pas installé : ImportError → retourne None
- SQLite path inaccessible : OSError → retourne None
- Toute autre erreur : log warning + None

Le pipeline est compilé sans checkpointer → reprise impossible mais l'exécution normale fonctionne.

### Q26.5 Que se passe-t-il si le LLM renvoie du non-JSON ?
**R :**
1. `extract_json_from_text` tente d'isoler le bloc JSON via regex
2. Si rien trouvé : `parse fail` → spec=None → spec_valid=False
3. Router LangGraph : retry (max 3) ou fallback hand-crafted
4. Le pipeline finit toujours par produire un mesh, même si avec un spec dégradé

### Q26.6 Que se passe-t-il si le mesh généré est invalide ?
**R :** `validate_mesh_node` vérifie la présence des clés (`uid`, `preview_url`). Si manquant : `mesh_valid=False`, retry (max 2). Si épuisé : pipeline progresse vers `store_result` avec ce qu'il a (errors documentés). Pas de boucle infinie.

### Q26.7 Idempotence ?
**R :** `gallery_db.insert` utilise `INSERT OR REPLACE` (upsert sur PRIMARY KEY uid). Si la même tâche tourne deux fois (Celery requeue après crash), le résultat est cohérent. Le checkpointer LangGraph est aussi idempotent par `thread_id`.

### Q26.8 Comment isolez-vous les pannes ?
**R :** Bulkhead pattern via queues Celery séparées. Si `document_processing` queue est saturée (parsing PDF lent), `3d_generation` queue continue de tourner. Workers indépendants → un crash d'un worker ne propage pas.

---

## 27. Observabilité (à implémenter)

### Q27.1 Que loggez-vous actuellement ?
**R :** `logging.info` pour les étapes clés (start/end de chaque génération, temps écoulé). `logging.warning` pour les fallbacks (rembg failed, retry). `logging.exception` pour les erreurs avec stack trace. Output stdout (capté par Docker logs).

### Q27.2 Quelles métriques ajouteriez-vous ?
**R :**
- `generation_duration_seconds` (histogram) par mode
- `cache_hit_rate` (counter)
- `queue_depth` (gauge) Celery
- `worker_active_tasks` (gauge)
- `gpu_memory_used_gb` (gauge)
- `pipeline_node_duration_seconds` (histogram) par nœud LangGraph

Format Prometheus, exposés via `/metrics` endpoint.

### Q27.3 Alertes ?
**R :** Cas alarmants :
- Queue depth > 10 pendant > 5 min (saturation)
- Aucune génération réussie en 1h (panne)
- Cache hit rate < 5% (mauvaise utilisation, ou paramètres mal hashés)
- GPU memory > 90% (risque OOM)
- Worker process restart > 3 fois en 10 min (crash loop)

### Q27.4 Tracing distribué ?
**R :** OpenTelemetry idéal. Traces : frontend request → FastAPI → Celery dispatch → worker execution. Chaque span avec timing + attributs. Visualisé dans Jaeger/Tempo. Pas implémenté actuellement.

### Q27.5 Comment debug en production sans accès SSH ?
**R :**
- Logs centralisés (ELK ou Loki) avec recherche par request_id / task_id
- Sentry pour les exceptions (stack trace + breadcrumbs)
- Métriques + dashboards Grafana
- Health checks pour détecter dégradation

### Q27.6 Audit trail ?
**R :** Chaque génération crée une row gallery_db avec timestamp + paramètres + source. Permet de retracer "qui a généré quoi quand". Pour compliance plus stricte (RGPD) : log immutable séparé.

---

## 28. Méthodologie & CRISP-DM

### Q28.1 Avez-vous suivi une méthodologie ?
**R :** Oui, CRISP-DM adapté au contexte ML/dev :
1. **Business understanding** : besoin de génération 3D automatisée depuis documents (pas de solution OSS existante)
2. **Data understanding** : analyse des PDFs/emails clients, identification des entités 3D-pertinentes
3. **Data preparation** : pipeline unstructured + nettoyage + extraction LLM
4. **Modeling** : sélection Hunyuan3D vs alternatives (TripoSR, Shap-E, Zero123++), prototypes
5. **Evaluation** : tests qualitatifs (rendus 3D), tests fonctionnels (smoke test)
6. **Deployment** : Docker compose, intégration Unity

### Q28.2 Combien de phases d'itération ?
**R :** ~3 phases majeures :
1. **POC** : prototype Image→3D simple (1 endpoint, threading)
2. **MVP** : 4 modes + WebSocket + Unity integration
3. **Production-ready** : Celery + LangGraph + tests + observabilité

### Q28.3 Comment avez-vous priorisé les features ?
**R :** Backlog informel :
- **Must** : 4 modes fonctionnels, persistence, intégration Unity
- **Should** : cache vectoriel, LangGraph pipeline, WebSocket
- **Could** : checkpointer, smoke tests, HITL
- **Won't (yet)** : authentification, rate limit, monitoring complet

### Q28.4 Outils de project management ?
**R :** Git (commits comme journal), Notion/Obsidian pour les notes architecturales, pas de Jira/Linear (équipe d'un seul dev).

### Q28.5 Code review ?
**R :** Solo. Auto-review via : (a) tests smoke avant chaque commit, (b) `py_compile` + `tsc`, (c) relecture diff avant push. Pour un projet en équipe : PR + CI obligatoires.

### Q28.6 Combien de lignes de code ?
**R :** Approximativement :
- Backend Python : ~5000 LOC (hors hy3dgen vendoré)
- Frontend TypeScript/TSX : ~4000 LOC
- Unity C# : ~500 LOC
- Tests + docs : ~1500 LOC
- **Total** : ~11000 LOC

---

## 29. Comparaisons académiques

### Q29.1 Comment Hunyuan3D se compare à Shap-E ?
**R :**
- **Shap-E** (OpenAI 2023) : génère des NeRFs, conversion en mesh moins propre
- **Hunyuan3D 2.0** (Tencent 2024) : génère directement des meshes via DiT + VAE 3D, qualité géométrique supérieure, support texture intégré

Sur les benchmarks F-Score et Chamfer Distance : Hunyuan3D > Shap-E par ~20-30%.

### Q29.2 Zero123++ vs notre approche multi-vues ?
**R :**
- **Zero123++** : génère 6 vues consistantes depuis une seule image, puis reconstruction via SDS (Score Distillation Sampling)
- **Hunyuan3D multi-view** : prend N vues réelles en input (notre cas), génère directement le mesh

Trade-off : Zero123++ est plus pratique (1 photo suffit), Hunyuan3D-mv est plus fidèle (s'adapte aux vues réelles).

### Q29.3 InstantMesh vs Hunyuan ?
**R :**
- **InstantMesh** (2024) : approche feed-forward (pas de diffusion), ~10s par génération mais qualité moindre
- **Hunyuan3D** : diffusion (plus lent), qualité géométrique supérieure

Choisi Hunyuan pour la qualité (PFE production-grade vs démo).

### Q29.4 Hunyuan3D 2.0 vs 2.1 ?
**R :** Hunyuan3D 2.1 (sortie fin 2024) ajoute :
- Modèles plus gros (3B paramètres)
- Meilleur support PBR
- Améliorations qualité géométrique

Pas adopté car : (a) requiert plus de VRAM (~16 GB minimum), (b) API breaking change. Sticking à 2.0 pour stabilité.

### Q29.5 Pourquoi pas un papier académique original ?
**R :** Le PFE est un projet d'**ingénierie** (intégration, déploiement, UX), pas de recherche fondamentale. Notre contribution : (a) architecture full-stack production-ready, (b) substitution multi-vues custom, (c) intégration Unity. Pas une nouvelle technique de génération 3D — ce serait un doctorat.

### Q29.6 Le papier Hunyuan3D mentionne quoi ?
**R :** "Hunyuan3D-2: A High-Resolution Texture-Aware 3D Generation Foundation Model" (Tencent Hunyuan, 2024). Architecture DiT 1.1B + VAE 3D. Entraîné sur ~10M de meshes 3D (mix Objaverse + datasets propriétaires). Benchmarks : FID-3D, Chamfer Distance, F-Score. Texture pipeline : delight + multi-view diffusion + UV-aware bake.

---

## 30. Roadmap & dette technique

### Q30.1 Quelles sont les prochaines features prioritaires ?
**R :**
1. Visualisation progress fine-grained (stages.py + diffusion callbacks)
2. ETA basé sur historique (pipeline_stats_db)
3. Authentification utilisateur (OAuth2 + JWT)
4. Rate limiting (slowapi)
5. Tests E2E avec Playwright
6. Monitoring (Prometheus + Grafana)

### Q30.2 Quelle dette technique avez-vous accumulée ?
**R :**
- `current_step` field LangGraph supprimé mais le code historique mentionne encore (cleanup à finir)
- Tests unitaires absents (juste smoke test)
- Pas de CI/CD GitHub Actions
- Docs API limitées au OpenAPI auto-généré (pas d'exemples curl)
- Logs structurés manquants (juste `logging.info`)
- Pas de versioning d'API (`/api/v1/` mais pas de v2 prévu)

### Q30.3 Refactor le plus urgent ?
**R :** Migrer le mode multi-vues vers `mv_pipeline` exclusivement (actuellement fallback i23d si mv fail). Code de fallback complexifie sans bénéfice clair.

### Q30.4 Comment évaluez-vous la qualité du code ?
**R :** Subjectif :
- **Lisibilité** : bonne (commentaires denses sur les choix non-évidents)
- **Modularité** : correcte (services séparés, pipeline modulaire)
- **Tests** : faible (smoke test seul)
- **Documentation** : moyenne (rapport PFE + ce Q&A, mais pas de docs API user-facing)

### Q30.5 Open-source ?
**R :** Repository public (`github.com/houcembelkhiria/3D-Generator`). License à formaliser (probablement MIT). Contributions externes : pas encore (projet personnel PFE).

---

## 31. Questions comportementales / parcours

### Q31.1 Quel a été le défi technique le plus difficile ?
**R :** La substitution multi-vues. 15 itérations sur 3-4 jours pour trouver l'approche correcte (re-center mesh sur médiane + target_case_size depuis vue IA + pas de warping). Chaque itération semblait raisonnable a priori mais introduisait des régressions visibles. Leçon : valider empiriquement, ne pas se fier à l'intuition seule sur les pipelines ML.

### Q31.2 Qu'avez-vous appris pendant ce projet ?
**R :**
- **ML inference engineering** : c'est très différent du training. Mémoire GPU, offload, MPS quirks
- **Async/distributed systems** : Celery, queue design, race conditions
- **Architecture full-stack** : équilibre entre backend/frontend, où placer la logique
- **Debugging itératif** : sur les bugs ML, valider par observation directe (visualiser, sauver des debug images)
- **Trade-offs** : aucune décision n'est "best" en absolu, dépend du contexte (latence vs qualité, simplicité vs flexibilité)

### Q31.3 Si vous deviez recommencer, que feriez-vous différemment ?
**R :**
1. **Celery dès le début** : la migration `threading.Thread`→Celery a été lourde
2. **Tests CI dès le jour 1** : aurait évité des régressions
3. **Architecture state-machine plus stricte** (Pydantic vs TypedDict) pour éviter les champs morts
4. **Choisir Apple Silicon vs CUDA dès le début** : naviguer entre les deux a doublé certains effort
5. **Documenter les "why nots"** : pourquoi pas X — souvent oublié

### Q31.4 Comment travaillez-vous quand vous êtes bloqué ?
**R :**
1. Reproduire le bug en isolation (test minimum)
2. Vérifier les logs + debug print
3. Vérifier la documentation officielle
4. Chercher GitHub issues du projet
5. Demander à un LLM (Claude/GPT) avec contexte précis
6. Si toujours bloqué : pause, retour avec œil neuf

### Q31.5 Comment gérez-vous la pression / les deadlines ?
**R :** Découpage en jalons hebdomadaires, MVP en premier, polish ensuite. Commit fréquent pour pouvoir rollback. Smoke test pour valider chaque ajout. Si vraiment en retard : couper le scope, pas la qualité.

### Q31.6 Vous travaillez en équipe ou solo ?
**R :** Ce projet : solo. Comfortable dans les deux modes. En équipe : code review, pair programming sur les bugs complexes, communication async via Slack/PR.

---

## 32. Questions très techniques (edge cases)

### Q32.1 Que se passe-t-il si l'utilisateur upload un PNG transparent en front ?
**R :** rembg ne fait rien (image déjà sans fond). Image passée telle quelle au DiT. Fonctionne bien — le DiT supporte les inputs avec alpha.

### Q32.2 Et si l'image est en niveaux de gris ?
**R :** `image.convert("RGB")` dans le service (`_decode_b64_image`). Le DiT est entraîné sur RGB, niveaux de gris convertis vers 3 canaux identiques.

### Q32.3 Que se passe-t-il si l'image est gigantesque (10000x10000) ?
**R :** Pour multi-vues : `pil.thumbnail((1024, 1024), Image.LANCZOS)` avant rembg pour éviter l'OOM. Pour image-to-3d : pas de cap explicite — pourrait OOM en RAM. À ajouter en production.

### Q32.4 Concurrency : 2 utilisateurs uploadent en même temps ?
**R :** FastAPI accepte les deux requêtes (async). Les deux tâches Celery vont en queue. Worker avec `prefetch_multiplier=1` traite séquentiellement. User #2 attend, voit "queued" puis "processing" via polling/WebSocket.

### Q32.5 Que se passe-t-il si Redis disparait pendant qu'un job tourne ?
**R :** Le worker se déconnecte du broker mais continue son inférence en cours. À la fin, il essaie d'écrire le résultat dans le backend (Redis aussi) : échec. La tâche est marquée FAILURE côté client. Le worker se reconnecte automatiquement pour les futures tâches.

### Q32.6 Que se passe-t-il si la SQLite gallery_db est corrompue ?
**R :** Au démarrage, `_init()` tente `CREATE TABLE IF NOT EXISTS`. Si corrompue : `sqlite3.DatabaseError`. Pas de récupération automatique. Solution : restaurer depuis backup ou supprimer (recrée vide).

### Q32.7 Que se passe-t-il si l'utilisateur ferme le navigateur pendant la génération ?
**R :** Aucun impact backend. Le job continue à tourner dans le worker, finit, écrit dans gallery_db. À la réouverture, l'utilisateur retrouve son modèle dans la galerie. C'est précisément pourquoi on a migré vers Celery (vs threading qui perdait tout sur refresh).

### Q32.8 Comment vérifiez-vous que le mesh généré n'est pas vide ?
**R :** Dans `image_to_3d` :
```python
if mesh is None:
    raise RuntimeError("Shape generation produced no mesh...")
```
Plus loin : `validate_mesh_node` vérifie le GLB sur disque (existe + non vide).

### Q32.9 Que se passe-t-il avec un Unicode dans le prompt (emoji, chinois) ?
**R :** Stocké tel quel (UTF-8). Llama-3 supporte multi-langue. Hunyuan3D text-to-3D passe par t2i (HunyuanDiT supporte chinois nativement, SDXL via prompt translation). Les emoji dans le prompt sont généralement filtrés par le t2i model.

### Q32.10 Comment gérez-vous les caractères spéciaux dans les noms de fichier ?
**R :** Tous les fichiers générés sont nommés `{uid}.glb` où uid est un UUID v4 (`xxxxxxxx-xxxx-...`). Aucun caractère spécial. Les filenames uploadés par l'utilisateur ne sont jamais utilisés tels quels — on les renomme en UUID.

### Q32.11 Race condition possible dans gallery_db ?
**R :** Protégée par `threading.Lock`. Une seule écriture à la fois. Lectures concurrent OK (avec WAL). Pas de read-modify-write multi-statement (le `INSERT OR REPLACE` est atomique).

### Q32.12 Memory leak potentiel ?
**R :** Surveillé :
- Pipelines ML : offload + `gc.collect()` après chaque génération
- Worker process recyclable (Celery `worker_max_tasks_per_child=N` pour recycler après N tâches — pas activé par défaut, à considérer)
- WebSocket : connection close handlers nettoient les ressources

Si vraiment problématique : restart périodique du worker via cron / supervisor.

---

## 33. Démo live — comment réagir

### Q33.1 "Faites une démo de la génération multi-vues"
**R :** Préparer 3-4 photos d'un objet (avant la soutenance) :
1. Démarrer `make dev-v2`
2. Ouvrir frontend, mode "Multi-vues"
3. Upload les 4 photos (front, back, left, right)
4. Choisir preset "Balanced"
5. Lancer génération
6. Pendant que ça tourne, expliquer le pipeline (subgraphes, Celery, etc.)
7. Montrer le résultat dans le viewer 3D
8. Bonus : ouvrir dans Unity Editor

### Q33.2 "Et si la démo plante ?"
**R :** Prévoir un screenshot du résultat attendu en backup. Expliquer ce qui aurait dû se passer. Pas de panique — montrer la stack-trace, expliquer rapidement. Les profs apprécient la capacité à debugger en live.

### Q33.3 "Montrez-moi le code de X"
**R :** Avoir VS Code ouvert avec les fichiers clés :
- `Backend/app/pipeline/graph.py`
- `Backend/app/tasks_3d.py`
- `Backend/hy3dgen/texgen/pipelines.py` (substitution multi-vues)
- `Frontend/hooks/useTaskPolling.ts`
- `UnityProject/Assets/Editor/SpawnBridge.cs`

Naviguer rapidement via Cmd+P (file search).

### Q33.4 "Pourquoi l'écran clignote-t-il / pourquoi c'est lent ?"
**R :** Honest : "Le rendu 3D dans le navigateur (model-viewer) consomme du GPU. Sur ma machine en mode dev avec plusieurs onglets ouverts, ça peut ralentir. En production avec un GPU dédié, c'est fluide."

### Q33.5 "Modifiez X et montrez le résultat"
**R :** Garder l'environnement chaud : pas de redémarrage backend. Modifier le frontend (HMR via Vite = instantané). Pour le backend, expliquer qu'un redémarrage serait nécessaire pour les changements de pipeline LangGraph (mais Celery reload n'est pas instantané).

### Q33.6 "Combien de temps ça prend de relancer après une modification ?"
**R :** Frontend : <1s (Vite HMR). Backend FastAPI : ~5s (uvicorn reload). Celery worker : ~30-60s (reload modèles Hunyuan). C'est pourquoi le dev est itératif : tests unitaires + smoke test plutôt que restart complet à chaque modif.

---

## 34. Pour finir : phrases-clés à mémoriser

### En cas de doute, dites :

> "L'architecture combine **Celery pour l'infrastructure de queue/isolation** et **LangGraph pour l'orchestration de workflow**. Les deux sont complémentaires : Celery décide *où et quand* exécuter, LangGraph décide *quoi faire*."

> "Le choix de re-centrer le mesh sur la **médiane des vertices** (vs bbox center) résout le problème de la case offset pour les montres à bracelet long. C'est la médiane qui est dominée par la région dense (le boîtier), pas la moyenne."

> "Le target_case_size est dérivé de la **vue IA du modèle multi-vues** (qui rend la vraie géométrie du mesh), pas de la photo utilisateur. Cela évite l'overflow de texture sur le bracelet."

> "Le pipeline LangGraph utilise **5 nœuds top-level + 2 sous-graphes compilés** pour modulariser le retry/fallback. Le checkpointer SqliteSaver permet la reprise après crash worker."

> "Le smoke test (260 LOC, sans pytest) valide en 5 secondes l'API + le wiring Celery, sans nécessiter Redis ni GPU."

### En cas de critique sur un choix :

> "C'est un trade-off conscient entre [simplicité / vitesse / qualité]. Pour notre échelle [N utilisateurs / 1 GPU / X générations par jour], ce choix est justifié. Pour scaler à 100x, il faudrait [migration vers Triton / multi-GPU / etc.]."

### Si on vous accuse de "wrapper d'API" :

> "Le wrapper représente ~30% du code. Les 70% restants : substitution multi-vues custom (geometrically non-trivial), intégration Unity Editor (protocole custom), cache vectoriel (CLIP+DINO design), orchestration LangGraph (5 nœuds + 2 sous-graphes), et toute la stack production-grade (Celery prod-config, smoke tests, observabilité). Ce n'est pas juste un appel à un modèle externe."

---

**Encore une fois : bonne soutenance !**
