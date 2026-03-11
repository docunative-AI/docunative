.PHONY: install test server-global server-earth demo

install:
	python -m venv venv
	. venv/bin/activate && pip install --upgrade pip && pip install -r requirements.txt
	@echo "✅ Environment ready. Run 'source venv/bin/activate' to activate."

test:
	. venv/bin/activate && pytest -q

server-global:
	bash models/start_server.sh global

server-earth:
	bash models/start_server.sh earth

demo:
	. venv/bin/activate && python ui/app.py