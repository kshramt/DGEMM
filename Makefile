# Configurations
.SUFFIXES:
.DELETE_ON_ERROR:
.SECONDARY:
.ONESHELL:
export SHELL := /bin/bash
export SHELLOPTS := pipefail:errexit:nounset:noclobber


# Tasks
.PHONY: all
.DEFAULT_GOAL := all
all:

# Specific to this project

TIMEOUT_SEC := 120

CXX := clang++
CXX_FLAGS := -std=c++17 -march=native -Wall -O3 -fopenmp
CXX_DEP_FLAGS := -MM

all_files := $(shell git ls-files)
cc_files := $(filter %.cc,$(all_files))


all: $(cc_files:%.cc=%.exe.done)


%.exe.done: %.exe
	mkdir -p $(@D)
	timeout --kill-after=$(TIMEOUT_SEC) $(TIMEOUT_SEC) nice -n19 $(abspath $<)
	touch $@


%.exe: %.o
	mkdir -p $(@D)
	$(CXX) $(CXX_FLAGS) -o $@ $^


%.o: %.cc
	mkdir -p $(@D)
	$(CXX) $(CXX_FLAGS) -c -o $@ $<


%.d: %.cc
	mkdir -p $(@D)
	$(CXX) $(CXX_FLAGS) -MM -MT $(@:%.d=%.o) -MF $@ $^


%.d: %.cc
	mkdir -p $(@D)
	$(CXX) $(CXX_FLAGS) -MM -MT $(@:%.d=%.o) -MF $@ $^


-include $(cc_files:%.cc=%.d)
