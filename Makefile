# tierlm - build with plain make and a C11 compiler.
#
#   make                    portable build (scalar + AVX2 + AVX-512 kernels, runtime dispatch)
#   make PROFILE=wrx90      Threadripper PRO 9000 / Zen 5 tuned build (profiles/wrx90.mk)
#   make URING=1            add the io_uring backend (needs liburing-devel)
#   make test               build and run unit + end-to-end tests
#   make asan / tsan        sanitizer builds of the test suite
#   make oracle MODEL=DIR MANIFEST=S.tlmm TOKENS=1,2,3   P-121 teacher-forced oracle vs HF
#   make clean
#
# BUILD=dir puts every object, library, binary and test under dir (default build/),
# so several configurations can coexist, e.g. make BUILD=build-debug OPTFLAGS=-O0.

CC      ?= cc
AR      ?= ar
PROFILE ?= generic
URING   ?= 0
BUILD   ?= build

OPTFLAGS  ?= -O2
CSTD       = -std=c11
WARN       = -Wall -Wextra -Wno-unused-parameter -Wshadow -Wpointer-arith
CPPFLAGS  += -D_GNU_SOURCE -Isrc/lib
CFLAGS    += $(CSTD) $(WARN) $(OPTFLAGS) -g -pthread -fno-math-errno -ffp-contract=off
LDLIBS    += -lm -pthread

-include profiles/$(PROFILE).mk

ifeq ($(URING),1)
CPPFLAGS += -DHAVE_LIBURING=1
LDLIBS   += -luring
endif

ARCH := $(shell uname -m)
ifeq ($(ARCH),x86_64)
KERN_AVX2_FLAGS   = -mavx2 -mfma -mf16c
KERN_AVX512_FLAGS = -mavx512f -mavx512bw -mavx512vl -mavx512vnni -mavx512dq -mfma -mf16c
else
KERN_AVX2_FLAGS   =
KERN_AVX512_FLAGS =
endif

LIB_SRC  = $(wildcard src/lib/*.c)
LIB_OBJ  = $(patsubst src/%.c,$(BUILD)/%.o,$(LIB_SRC))
BIN_NAMES = tierlm tlmc tierlm-setup
BINS     = $(addprefix $(BUILD)/bin/,$(BIN_NAMES))
TEST_SRC = $(wildcard tests/test_*.c)
TESTS    = $(patsubst tests/%.c,$(BUILD)/tests/%,$(TEST_SRC))

.PHONY: all lib bins test oracle asan tsan clean fmt-check

all: lib bins

lib: $(BUILD)/libtlm.a
bins: $(BINS)

$(BUILD)/lib/tlm_kern_avx2.o: src/lib/tlm_kern_avx2.c src/lib/*.h
	@mkdir -p $(dir $@)
	$(CC) $(CPPFLAGS) $(CFLAGS) $(KERN_AVX2_FLAGS) -c $< -o $@

$(BUILD)/lib/tlm_kern_avx512.o: src/lib/tlm_kern_avx512.c src/lib/*.h
	@mkdir -p $(dir $@)
	$(CC) $(CPPFLAGS) $(CFLAGS) $(KERN_AVX512_FLAGS) -c $< -o $@

$(BUILD)/%.o: src/%.c src/lib/*.h
	@mkdir -p $(dir $@)
	$(CC) $(CPPFLAGS) $(CFLAGS) -c $< -o $@

$(BUILD)/libtlm.a: $(LIB_OBJ)
	$(AR) rcs $@ $^

$(BUILD)/bin/%: src/bin/%.c $(BUILD)/libtlm.a
	@mkdir -p $(dir $@)
	$(CC) $(CPPFLAGS) $(CFLAGS) $< -o $@ $(BUILD)/libtlm.a $(LDLIBS)

$(BUILD)/tests/%: tests/%.c tests/tlm_test.h $(BUILD)/libtlm.a
	@mkdir -p $(dir $@)
	$(CC) $(CPPFLAGS) -Itests $(CFLAGS) $< -o $@ $(BUILD)/libtlm.a $(LDLIBS)

test: $(TESTS) $(BINS)
	@set -e; for t in $(TESTS); do echo "== $$t"; $$t; done
	@echo "== tests/setup.sh"; BUILD=$(BUILD) sh tests/setup.sh
	@echo "== tests/e2e.sh"; BUILD=$(BUILD) sh tests/e2e.sh
	@echo "ALL TESTS PASSED"

# P-121 teacher-forced oracle vs the HF reference (needs the torch venv + model):
#   make oracle MODEL=~/models/Qwen3-30B-A3B MANIFEST=path.tlmm TOKENS=1,2,3 [TOPK=10] [PY=PY]
oracle:
	@bash tools/oracle.sh "$(MODEL)" "$(MANIFEST)" "$(TOKENS)" "$(TOPK)" "$(PY)"

asan:
	$(MAKE) BUILD=build-asan OPTFLAGS="-O1 -fsanitize=address,undefined -fno-omit-frame-pointer" \
	        LDLIBS="-lm -pthread -fsanitize=address,undefined" test

tsan:
	$(MAKE) BUILD=build-tsan OPTFLAGS="-O1 -fsanitize=thread" LDLIBS="-lm -pthread -fsanitize=thread" test

clean:
	rm -rf $(BUILD) build build-asan build-tsan build-debug build-wrx90
