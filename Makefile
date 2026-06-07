# Convenience wrapper — targets live in dev/scripts/Makefile
.PHONY: install dev test lint typecheck demo tutorial doctor

install dev test lint typecheck demo tutorial doctor:
	@$(MAKE) -f dev/scripts/Makefile $@
