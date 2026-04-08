.PHONY: setup setup-backend setup-frontend backend frontend dev docker clean

# One-command full setup
setup: setup-backend setup-frontend
	@echo "\n=== All set! Run 'make dev' to start. ==="

setup-backend:
	@cd Backend && bash setup.sh

setup-frontend:
	@cd Frontend && npm install

# Run servers (auto-setup if needed)
backend: Backend/venv
	@cd Backend && source venv/bin/activate && uvicorn app.main:app --host 0.0.0.0 --port 8000

frontend: Frontend/node_modules
	@cd Frontend && npm run dev

# Auto-create venv if missing
Backend/venv:
	@cd Backend && bash setup.sh

Frontend/node_modules:
	@cd Frontend && npm install

# Run both in parallel (auto-setup if needed)
dev: Backend/venv Frontend/node_modules
	@trap 'kill 0' EXIT; \
	(cd Backend && source venv/bin/activate && uvicorn app.main:app --host 0.0.0.0 --port 8000) & \
	(cd Frontend && npm run dev) & \
	wait

# Docker
docker:
	docker-compose up --build

docker-down:
	docker-compose down

# Cleanup
clean:
	rm -rf Backend/venv Backend/*.egg-info Backend/build
	rm -rf Frontend/node_modules Frontend/dist
