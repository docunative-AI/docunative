.PHONY: install test server-global server-earth server-global-q3 server-global-iq2 check-metal build-metal demo freeze check-models models

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
	llama-server -m models/weights/tiny-aya-global-q4_k_m.gguf \
		-ngl 99 --flash-attn --cache-prompt \
		-c 4096 --port 8080

server-earth:
	llama-server -m models/weights/tiny-aya-earth-q4_k_m.gguf \
		-ngl 99 --flash-attn --cache-prompt \
		-c 4096 --port 8080

# CPU users — lower quantization for survivable latency
server-global-q3:
	llama-server -m models/weights/tiny-aya-global-q3_k_m.gguf \
		-ngl 0 --cache-prompt -c 2048 \
		-t $(nproc 2>/dev/null || sysctl -n hw.physicalcpu) \
		--port 8080

server-global-iq2:
	llama-server -m models/weights/tiny-aya-global-iq2_xxs.gguf \
		-ngl 0 -c 2048 \
		-t $(nproc 2>/dev/null || sysctl -n hw.physicalcpu) \
		--port 8080

# Metal verification — run this if generation feels slow on Mac
check-metal:
	@echo "Checking Metal support in llama-server..."
	@llama-server --version 2>&1 | grep -i "metal\|mps\|gpu" \
		|| echo "WARNING: Metal not detected — llama.cpp may not be compiled with Metal support."
	@echo "If Metal is missing, recompile: cmake .. -DLLAMA_METAL=ON && make -j$(sysctl -n hw.physicalcpu)"

build-metal:
	@echo "Recompiling llama.cpp with Metal support..."
	cmake .. -DLLAMA_METAL=ON -DCMAKE_BUILD_TYPE=Release
	make -j$(sysctl -n hw.physicalcpu)

demo:
	uv run python -m ui.app

# requirements.txt removed — uv.lock + pyproject.toml are the source of truth
# Use 'uv sync' to install dependencies
# Use 'uv add <package>' to add new packages (updates pyproject.toml + uv.lock)