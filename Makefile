PETSC_ROOT = $(HOME)/petsc-install
CC = mpicc

all: petsc_exp
.PHONY: all

petsc_exp: petsc_exp.c
	$(CC) $< -I $(PETSC_ROOT)/include -L$(PETSC_ROOT)/lib -lpetsc -Wl,-rpath,$(PETSC_ROOT)/lib -o $@

clean:
	rm -rf petsc_exp
.PHONY: clean
