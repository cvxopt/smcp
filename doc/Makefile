# Makefile for SMCP website

html:
	sphinx-build  -b html source build/html	
	cp source/benchmarks/*.html build/html/benchmarks/

linkcheck:
	mkdir -p build/linkcheck 
	sphinx-build -b linkcheck  source build/linkcheck
	@echo
	@echo "Link check complete; look for any errors in the above output " \
	      "or in build/linkcheck/output.txt."
