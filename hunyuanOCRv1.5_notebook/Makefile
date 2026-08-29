.PHONY: check run status stop

check:
	@for script in scripts/*.sh; do bash -n "$$script"; done
	python3 -m py_compile src/*.py scripts/*.py
	python3 scripts/validate_notebook.py hunyuan_ocr_demo.ipynb

run:
	./scripts/run.sh

status:
	./scripts/status.sh

stop:
	./scripts/stop.sh
