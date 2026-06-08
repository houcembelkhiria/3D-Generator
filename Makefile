.PHONY: \
  setup setup-backend setup-frontend \
  backend frontend dev \
  setup-v2 setup-backend-v2 setup-frontend-v2 \
  backend-v2 frontend-v2 dev-v2 \
  docker docker-down docker-v2 docker-v2-down \
  clean clean-v2

# ──────────────────────────────────────────────
#  v1  (original)  ports: Frontend 3000 · Backend 8000
# ──────────────────────────────────────────────

setup: setup-backend setup-frontend
	@echo "\n=== v1 ready. Run 'make dev' to start. ==="

setup-backend:
	@cd Backend && bash setup.sh

setup-frontend:
	@cd Frontend && npm install

Backend/venv:
	@cd Backend && bash setup.sh

Frontend/node_modules:
	@cd Frontend && npm install

backend: Backend/venv
	@cd Backend && source venv/bin/activate && uvicorn app.main:app --host 0.0.0.0 --port 8000

frontend: Frontend/node_modules
	@cd Frontend && npm run dev

dev: Backend/venv Frontend/node_modules
	@trap 'kill 0' EXIT; \
	(cd Backend && source venv/bin/activate && uvicorn app.main:app --host 0.0.0.0 --port 8000) & \
	(cd Frontend && npm run dev) & \
	wait

# ──────────────────────────────────────────────
#  v2  (new iteration)  ports: Frontend 3001 · Backend 8001
# ──────────────────────────────────────────────

setup-v2: setup-backend-v2 setup-frontend-v2
	@echo "\n=== v2 ready. Run 'make dev-v2' to start. ==="

setup-backend-v2:
	@cd Backend-v2 && bash setup.sh

setup-frontend-v2:
	@cd Frontend-v2 && npm install

Backend-v2/venv:
	@cd Backend-v2 && bash setup.sh

Frontend-v2/node_modules:
	@cd Frontend-v2 && npm install

backend-v2: Backend-v2/venv
	@cd Backend-v2 && source venv/bin/activate && uvicorn app.main:app --host 0.0.0.0 --port 8001

frontend-v2: Frontend-v2/node_modules
	@cd Frontend-v2 && npm run dev

dev-v2: Backend-v2/venv Frontend-v2/node_modules
	@trap 'kill 0' EXIT; \
	(cd Backend-v2 && source venv/bin/activate && uvicorn app.main:app --host 0.0.0.0 --port 8001) & \
	(cd Frontend-v2 && npm run dev) & \
	wait

# ──────────────────────────────────────────────
#  Docker
# ──────────────────────────────────────────────

docker:
	docker-compose up --build

docker-down:
	docker-compose down

docker-v2:
	docker-compose -f docker-compose-v2.yml up --build

docker-v2-down:
	docker-compose -f docker-compose-v2.yml down

# ──────────────────────────────────────────────
#  Cleanup
# ──────────────────────────────────────────────

clean:
	rm -rf Backend/venv Backend/*.egg-info Backend/build
	rm -rf Frontend/node_modules Frontend/dist

clean-v2:
	rm -rf Backend-v2/venv Backend-v2/*.egg-info Backend-v2/build
	rm -rf Frontend-v2/node_modules Frontend-v2/dist
