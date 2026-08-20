## Memory investigation — two ensembles on Streamlit Community Cloud

Measured 2026-08-19 in a Linux container, ONNX Runtime 1.25.0, Python 3.11,
using the 20-member v1.0 ONNX ensemble with the app's own SessionOptions
(`inter_op_num_threads = 1`, `intra_op_num_threads = 1`, CPU provider).
RSS from `psutil`; peak from `/proc/self/status` `VmHWM`.

**Platform limit:** Streamlit Community Cloud gives 690 MB minimum, 2.7 GB
maximum per app. Design to the 690 MB floor, not the ceiling.

### The finding: the binding constraint is not the second ensemble

ONNX Runtime's CPU memory arena is on by default, grows to the high-water mark
of any single inference call, and never returns memory for the life of the
session. Measured with the configuration that ships today — `CHUNK = 500`,
arena at its default:

| batch | RSS |
|---|---:|
| 20 sessions loaded, no inference | 284 MB |
| one 500-soil chunk × 19 potentials (today's default) | **888 MB** |
| same, user enters 50 potentials | **2,541 MB** |
| same, v1.1 canonical grid of 151 points | **5,060 MB** |

The first row of real work already exceeds the 690 MB floor. This is a live
property of the deployed app, not a consequence of adding a second model.

### Fix: `enable_cpu_mem_arena = False`

Same work, arena disabled:

| state | RSS | peak |
|---|---:|---:|
| one ensemble resident | 276 MB | 277 MB |
| + 100-soil × 151 batch | 278 MB | 334 MB |
| + 500-soil × 151 batch | 278 MB | 544 MB |
| **both ensembles resident** | **511 MB** | 544 MB |
| both resident, 100-soil × 151 on each | 503 MB | 560 MB |

A 500-soil × 151-point batch costs 41 MB of transient allocation instead of
4.5 GB, and it is also faster: a 100-soil x 151-point ensemble pass takes
2,165 ms with the arena off against 3,229 ms with it on; the single-soil path is
52.9 vs 55.1 ms. Weights are not affected — they load once at session
construction; the arena holds only intermediate tensors.

### Eviction works, but needs an explicit trim

`del` + `gc.collect()` releases the sessions to the allocator; glibc does not
return the pages to the OS on its own, and RSS is what gets a container killed.

| step | RSS |
|---|---:|
| two ensembles + a large batch, arena on | 4,327 MB |
| `del` + `gc.collect()` | 704 MB |
| + `malloc_trim(0)` | 292 MB |
| both released + trim | 69 MB |

So `st.cache_resource(max_entries=1)` does free the ONNX sessions — the
Python-level eviction is real — but without `ctypes.CDLL("libc.so.6").malloc_trim(0)`
several hundred MB stay resident.

### Recommendation

1. **`opts.enable_cpu_mem_arena = False`** in the app and in `habit-ptf`. This
   is the whole answer; everything else is margin.
2. **`CHUNK = 100`** rather than 500. With the arena off this matters much less,
   but it also caps the numpy-side `(20, chunk, n_wp)` array.
3. **Keep `max_entries=1`** and call `malloc_trim(0)` immediately after
   obtaining an ensemble. One resident ensemble is ~276 MB, leaving real
   headroom under the 690 MB floor once Streamlit's own runtime is counted.
   Both resident is *possible* at 511 MB but leaves only ~60–90 MB of margin,
   which is not enough to be comfortable.

Two open implementation details: Streamlit's eviction ordering relative to the
loader (trimming after retrieval rather than inside the loader is
order-independent, so this does not need resolving), and inference throughput
with the arena off, which was not measured.

### Correction to an earlier figure

Per-member **resident** cost is 12.0 MB, against 7.4 MB on disk. The "148 MB
ensemble" figure is disk size; resident is ~240 MB.
