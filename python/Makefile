# Test runner for the k26rl package. Each gate is one process and
# owns its skip conditions: it exits 77 with a printed reason when
# the built compiler, a stack archive, or the gymnasium package is
# absent, so `make test` completes everywhere and reports what it
# could not check. PYTHON selects the interpreter (for instance a
# virtual environment's); the default is python3.
#
# Gate numbers 8 and 9 belong to the telemetry transport suite and
# land with that component; the numbering here stays aligned with
# the full suite.

PYTHON ?= python3

GATES = tests/gate01_cross_driver.py \
        tests/gate02_spec.py \
        tests/gate03_fault.py \
        tests/gate04_status_decode.py \
        tests/gate05_autoreset.py \
        tests/gate06_seeding.py \
        tests/gate07_api_conformance.py \
        tests/gate10_marshalling.py \
        tests/gate11_loading.py

.PHONY: test
test:
	@for t in $(GATES); do \
	    $(PYTHON) $$t; rc=$$?; \
	    if [ $$rc -eq 77 ]; then \
	        echo "$$(basename $$t): SKIP"; \
	    elif [ $$rc -ne 0 ]; then \
	        echo "$$(basename $$t): FAIL (rc=$$rc)"; \
	        exit $$rc; \
	    fi; \
	done
