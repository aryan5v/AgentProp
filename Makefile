# Convenience wrapper — targets live in scripts/Makefile
.PHONY: install dev test lint typecheck demo tutorial doctor

install dev test lint typecheck demo tutorial doctor:
	@$(MAKE) -f scripts/Makefile $@
