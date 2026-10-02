# Threadripper PRO 9965WX (Zen 5), RHEL 10.2. See research/02-wrx90-build-profile.md.
# GCC >= 14 knows znver5; older compilers fall back to znver4 (same AVX-512 subset we use).
ZNVER := $(shell $(CC) -march=znver5 -E -x c /dev/null >/dev/null 2>&1 && echo znver5 || echo znver4)
OPTFLAGS  = -O3 -march=$(ZNVER) -mtune=$(ZNVER) -mprefer-vector-width=512 \
            -fno-math-errno -fno-trapping-math -fno-semantic-interposition
# No -ffast-math: reassociation breaks the bitwise invariance tests.
URING    ?= 1
