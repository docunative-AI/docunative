.PHONY: install test server-global server-earth demo check-models models

install:
	uv venv --python 3.11
	uv sync
	@$(MAKE) check-models
	@echo "✅ Environment ready. Run 'source .venv/bin/activate' to activate."

check-models:
	@echo "🔍 Checking for model files..."
	@if [ ! -f models/weights/tiny-aya-global-q4_k_m.gguf ] || [ ! -f models/weights/tiny-aya-earth-q4_k_m.gguf ]; then \
		echo "⚠️  Model files not found. Downloading (~4.2 GB, this may take 5-10 minutes)..."; \
		uv run python models/pull_models.py; \
		if [ $$? -eq 0 ]; then \
			echo "✅ Models downloaded successfully."; \
		else \
			echo "❌ Model download failed. Please try manually:"; \
			echo "   source .venv/bin/activate && python models/pull_models.py"; \
			exit 1; \
		fi \
	else \
		echo "✅ Model files already present."; \
	fi

models:
	@echo "📥 Downloading model files (~4.2 GB total)..."
	@echo "This may take 5-10 minutes depending on your internet speed."
	uv run python models/pull_models.py

test:
	uv run pytest -q

server-global:
	bash models/start_server.sh global

server-earth:
	bash models/start_server.sh earth

demo:
	uv run python ui/app.py