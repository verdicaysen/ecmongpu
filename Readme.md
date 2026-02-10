# co-ecm (HIP)

This is an AMD HIP port of co-ecm, a GPU-accelerated implementation of
Lenstra's Elliptic Curve Method (ECM) for integer factorization.

It operates on "a = -1" twisted Edwards curves with extended projective
coordinates, uses w-NAF point multiplication during stage one and a
baby-step/giant-step approach during stage two.

All GPU multi-precision operations are performed in Montgomery arithmetic
using a custom fixed-bitlength implementation. Stages one and two execute
entirely on the GPU (GCD computation excluded).


## HIP Port

Ported to AMD GPUs (HIP/ROCm) by Verdic Aysen.


# Paper

> *Jonas Wloka, Jan Richter-Brockmann, Colin Stahlke, Thorsten Kleinjung,
> Christine Priplata, Tim Güneysu*:
> **Revisiting ECM on GPUs**.
> 19th International Conference on Cryptology and Network Security (CANS 2020),
> December 14-16, 2020


## Building

### Dependencies

 - [ROCm](https://rocm.docs.amd.com/) (tested with ROCm 6.x/7.x)
 - [GMP](https://gmplib.org/) (GNU Multi Precision Arithmetic Library)
 - CMake >= 3.21
 - A C/C++ compiler (gcc/g++)
 - Python 3 (for benchmarks only)


### Compilation

`BITWIDTH` is the most important build parameter. It must be **at least 32
bits larger** than the numbers you want to factor. It must be a multiple of 32.
Higher values use more GPU registers per thread, which reduces occupancy and
throughput.

| Input size         | Bits needed | BITWIDTH to set |
|--------------------|-------------|-----------------|
| 58 digits (192-bit)| ~192        | 224 (default)   |
| 100 digits         | ~333        | 384             |
| 150 digits         | ~498        | 544             |
| 215 digits         | ~714        | 768             |

Other build-time options (set via environment variables before running cmake):

| Variable         | Default | Description                              |
|------------------|---------|------------------------------------------|
| `BITWIDTH`       | 192     | Max input bit size (must be multiple of 32, at least 32 more than input) |
| `BATCH_JOB_SIZE` | 32768   | Curves per GPU batch. Reduce if you hit OOM at large BITWIDTH |
| `WINDOW_SIZE`    | 4       | w-NAF window size for point multiplication |
| `MON_PROD`       | FIPS    | Montgomery product algorithm (CIOS, FIPS, FIOS, CIOS_XMAD) |
| `GPU_ARCH`       | auto    | GPU architecture target (e.g. gfx1201 for RX 9070 XT) |

Build for a 215-digit number:

```
cd /path/to/source
mkdir build && cd build
BITWIDTH=768 cmake ..
make -j2 cuda-ecm
```

Use `-j2` instead of full parallelism to avoid running out of RAM during
compilation (each target recompiles the full GPU codebase).

### Testing

```
make -j2 all
ctest
```

### GPU architecture auto-detection

The build system runs `rocminfo` to detect your GPU. If that fails, it
defaults to `gfx1201`. Override with:

```
GPU_ARCH=gfx1100 BITWIDTH=768 cmake ..
```


## Usage

```
./bin/cuda-ecm -c config.ini
```

### Command-line options

| Flag                | Description                       |
|---------------------|-----------------------------------|
| `-c, --config FILE` | Config file (required)            |
| `--b1 N`            | Override stage 1 bound            |
| `--b2 N`            | Override stage 2 bound            |
| `-e, --effort N`    | Override effort (curves per number)|
| `-f, --file`        | Force file input mode             |
| `-l, --listen`      | Force server input mode           |
| `-p, --port N`      | Port for server mode              |
| `-s, --silent`      | Suppress all but fatal output     |
| `--log N`           | Log level (1=verbose .. 7=none)   |

### Configuration file

All options go in an INI-format config file. See `example/config.ini` for a
fully commented example. The key sections:

**[general]**
- `mode` - `file` or `server`
- `loglevel` - 1 (verbose) through 7 (none), default 3 (info)
- `random` - `true`/`false` for RNG seed

**[file]**
- `input` - path to input file
- `output` - path to output file

**[cuda]** (name kept for compatibility)
- `streams` - concurrent GPU streams (default: 2)
- `threads_per_block` - threads per block, or `auto` (default: auto)

**[ecm]**
- `b1` - stage 1 smoothness bound
- `b2` - stage 2 smoothness bound
- `effort` - max curves per input number
- `curve_gen` - curve generation method: 0=Naive, 1=GKL2016_j1, 2=GKL2016_j4
- `powersmooth` - `true` for lcm(2..b1), `false` for primorial
- `find_all_factors` - `true` to keep going after first factor
- `stage2.enabled` - `true`/`false`
- `stage2.window_size` - baby-step/giant-step window (default: 2310)


### Input file format

One number per line: `<id> <number>`

```
1 12345678901234567890123456789012345678901234567890
```

Lines not starting with a digit are ignored (comments).

### Output format

```
<id> <factor1>,<factor2>, # effort: <curves_used>
```


## Factoring a 215-digit number

A 215-digit composite is ~714 bits. You need a build with `BITWIDTH=768`.

### 1. Rebuild

```
cd /path/to/source/build
rm -rf *
BITWIDTH=768 cmake ..
make -j2 cuda-ecm
```

### 2. Create your input file

Create a file (e.g. `input215.txt`) with your number:

```
1 <your 215-digit number here>
```

### 3. Create a config file

Create `config215.ini`:

```ini
[general]
mode = file
loglevel = 3
random = true

[file]
input = /path/to/input215.txt
output = /path/to/output215.txt

[cuda]
streams = 2
threads_per_block = auto

[ecm]
; ECM is best at finding "small" factors relative to the input.
; It does NOT brute-force all 215 digits — it finds a factor of
; D digits with probability that increases with effort and B1/B2.
;
; Recommended B1/B2 by factor size you're hoping to find:
;   ~25 digits:  b1 = 50000,       b2 = 5000000
;   ~30 digits:  b1 = 250000,      b2 = 25000000
;   ~35 digits:  b1 = 1000000,     b2 = 100000000
;   ~40 digits:  b1 = 3000000,     b2 = 300000000
;   ~45 digits:  b1 = 11000000,    b2 = 1100000000
;   ~50 digits:  b1 = 43000000,    b2 = 4300000000
;   ~55 digits:  b1 = 110000000,   b2 = 11000000000
;   ~60 digits:  b1 = 260000000,   b2 = 26000000000
;
; Start with smaller bounds and increase if no factor is found.
; Each B1 level needs roughly 2-3x more curves than the previous.

b1 = 50000
b2 = 5000000
effort = 5000
curve_gen = 2
powersmooth = true
find_all_factors = false
stage2.enabled = true
stage2.window_size = 2310
```

### 4. Run

```
./bin/cuda-ecm -c config215.ini
```

### 5. Strategy

ECM finds small factors efficiently. It does not crack the full 215-digit
semiprime directly — it searches for a factor of a given size. The approach:

1. Start with `b1=50000` and `effort=5000`. This quickly searches for factors
   up to ~25 digits.
2. If nothing is found, increase to `b1=250000, b2=25000000, effort=3000`.
3. Keep doubling B1 and effort, working your way up the table above.
4. If the composite has no factor below ~60 digits, ECM is the wrong tool
   and you would need GNFS/SNFS instead.

The program reports throughput as curves/second. At larger BITWIDTH the
throughput drops significantly because each curve uses more GPU registers.
