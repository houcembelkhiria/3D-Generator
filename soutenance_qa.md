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
