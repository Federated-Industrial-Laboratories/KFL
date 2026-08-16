# Test runner for the k26rl package. Each gate is one process and
# owns its skip conditions: it exits 77 with a printed reason when
# the built compiler, a stack archive, or a package it needs is
# absent, so `make test` completes everywhere and reports what it
# could not check. The multi-agent gate needs pettingzoo, which is an
# optional dependency of the package, so it is the one gate a
# consumer of the single-agent shapes alone will see skip.
#
# Every gate runs; the summary line counts passes,
# skips, and failures, the exit status is nonzero exactly when a
# gate failed, and a run in which every gate skipped fails, because
# an all-skip exit 0 enforces nothing and a green suite that checked
# nothing is worse than a red one. A machine that cannot install the
# shim's dependencies stays usable by passing ALLOW_ALL_SKIP=1, which
# makes the same run pass and says on the line above it that nothing
# was enforced; the point is that the exemption is typed by someone
# rather than assumed. PYTHON selects the interpreter (for instance a
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
        tests/gate11_loading.py \
        tests/gate12_environments.py \
        tests/gate13_parallel.py

.PHONY: test
test:
	@passed=0; skipped=0; failed=0; \
	for t in $(GATES); do \
	    $(PYTHON) $$t; rc=$$?; \
	    if [ $$rc -eq 77 ]; then \
	        echo "$$(basename $$t): SKIP"; \
	        skipped=$$((skipped + 1)); \
	    elif [ $$rc -ne 0 ]; then \
	        echo "$$(basename $$t): FAIL (rc=$$rc)"; \
	        failed=$$((failed + 1)); \
	    else \
	        passed=$$((passed + 1)); \
	    fi; \
	done; \
	echo "gates: $$passed passed, $$skipped skipped, $$failed failed"; \
	if [ $$passed -eq 0 ] && [ $$failed -eq 0 ]; then \
	    echo "all gates skipped: no enforcement this run"; \
	    if [ -z "$(ALLOW_ALL_SKIP)" ]; then \
	        echo "failing on that: pass ALLOW_ALL_SKIP=1 to accept it"; \
	        exit 1; \
	    fi; \
	fi; \
	[ $$failed -eq 0 ]
