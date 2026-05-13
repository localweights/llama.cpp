# Tap-eval-cb async patches — drafts for next capture cycle

Two patches for `tools/server/server-context.cpp` to relieve the synchronous
`ggml_backend_tensor_get` + write bottleneck on the main eval thread.

Apply order: **#2 first** (low risk, ~5-10% gain). If more needed, layer **#1**
on top (medium-high risk, additional 5-15%).

Test both with capture validator + sample-row spot check before full sweep.

---

## #2 — Dedicated writer thread + bounded queue (low risk)

**Idea:** callback still synchronously copies device→host, but `write()` +
`fsync()` move to a worker thread on a different core. Main thread blocks
only on the unavoidable cudaMemcpy (~5-8ms), not on disk write (~5-10ms +
syscall overhead). Net win: disk path runs in parallel with next forward.

### State (members of `server_context_impl`)

```cpp
// --- async tap writer ---
struct TapPayload {
    std::vector<ggml_fp16_t> data;   // owned f16 row block (4095 × n_embd)
};
std::mutex                    tap_q_mtx;
std::condition_variable       tap_q_cv;
std::queue<TapPayload>        tap_q;
std::thread                   tap_writer_thread;
std::atomic<bool>             tap_writer_stop{false};
static constexpr size_t       TAP_Q_MAX = 4;   // backpressure bound
```

### Writer thread body

```cpp
void tap_writer_loop() {
    while (true) {
        TapPayload p;
        {
            std::unique_lock<std::mutex> lk(tap_q_mtx);
            tap_q_cv.wait(lk, [&]{ return !tap_q.empty() || tap_writer_stop.load(); });
            if (tap_writer_stop.load() && tap_q.empty()) return;
            p = std::move(tap_q.front()); tap_q.pop();
        }
        tap_q_cv.notify_one();   // wake any blocked producer
        if (tap_h_pre_norm_file.is_open() && !p.data.empty()) {
            tap_h_pre_norm_file.write((const char *)p.data.data(),
                                       p.data.size() * sizeof(ggml_fp16_t));
        }
    }
}
```

### Producer (modified callback hot path)

Replace the existing `tap_h_pre_norm_file.write(...)` line with:

```cpp
TapPayload payload;
payload.data.assign(half_buf_begin, half_buf_begin + n_elem);   // copy out
{
    std::unique_lock<std::mutex> lk(self->tap_q_mtx);
    self->tap_q_cv.wait(lk, [&]{ return self->tap_q.size() < TAP_Q_MAX; });   // backpressure
    self->tap_q.push(std::move(payload));
}
self->tap_q_cv.notify_one();
```

Where `half_buf_begin` is either `self->tap_host_buf.data()` (F16 fast path)
or `self->tap_half_buf.data()` (F32/BF16 converted path) per the existing
batch patch.

### Lifecycle

- Start writer thread when `tap_h_pre_norm_file` is first opened (in
  `init_tap_layers` / first request).
- On server shutdown: set `tap_writer_stop=true`, `notify_all()`, `join()`.
  Then close file.

### Invariants

- Queue order preserves window order (FIFO). Single producer (main thread)
  + single consumer = no reorder.
- Backpressure: producer blocks if queue full → bounds memory at `4 ×
  16.7MB = 67MB`. No data loss, no GPU bubble unless writer falls
  more than 4 windows behind (sustained NFS stall).
- Crash safety: same as before (file size + window offset). Queue draining
  loss on hard kill = ≤4 windows = ~16k tokens, recoverable via resume.

### Pin writer to a non-hot CCD0 core

```cpp
cpu_set_t set; CPU_ZERO(&set);
for (int c : {1,2,3,5,6,7,17,18,19,21,22,23}) CPU_SET(c, &set);
pthread_setaffinity_np(tap_writer_thread.native_handle(), sizeof(set), &set);
```

Avoids contention with main forward thread that's pinned around core 0/16.

### Expected gain

5-10%. Frees main thread from disk syscall latency; copy still inline.

### Risk

Medium. Threading bugs hard to test (queue/cv races). Recommend smoke run on
HCL slug for 5min, validate via existing `validate_capture.py` + manual
sample-row spot check at 10 random offsets before full sweep.

---

## #1 — cudaMemcpyAsync + pinned host memory (medium-high risk)

**Idea:** allocate page-locked (pinned) host buffer via `cudaHostAlloc`.
Submit `cudaMemcpyAsync` on a dedicated CUDA stream. Callback returns
immediately. GPU overlaps the device→host transfer with next kernel
launches on the main compute stream. Writer thread (from #2) blocks on
`cudaStreamSynchronize` then writes.

**Critical caveat:** llama.cpp's eval-callback is invoked AFTER the kernel
producing the tensor has been scheduled on the compute stream. The callback
itself runs on the calling thread (main eval thread). So the ASYNC copy
must happen on a SEPARATE CUDA stream, with an explicit cross-stream
dependency: wait for main compute stream's event before starting copy.

### State

```cpp
// Pinned host buffer (registered with CUDA)
void *      tap_pinned_host = nullptr;     // cudaHostAlloc'd
size_t      tap_pinned_size = 0;
cudaStream_t tap_copy_stream = nullptr;     // dedicated copy stream
cudaEvent_t  tap_copy_done   = nullptr;     // signaled when copy lands

// Writer thread + queue (from #2 above)
struct TapPayload {
    cudaEvent_t copy_done;        // ref to event (NOT owning)
    size_t      n_bytes;          // bytes valid in pinned region
    size_t      pinned_off;       // offset within tap_pinned_host (ring buffer)
};
std::queue<TapPayload> tap_q;
// ... mtx, cv, thread, stop flag as in #2
```

### One-time init (when tap_h_pre_norm_file opens)

```cpp
// Allocate 4× window worth of pinned mem as a ring buffer
size_t row_bytes_max = expected_n_embd * sizeof(ggml_fp16_t);
size_t window_bytes  = row_bytes_max * 4096;
tap_pinned_size      = window_bytes * 4;   // 4 windows ring
cudaHostAlloc(&tap_pinned_host, tap_pinned_size, cudaHostAllocPortable);
cudaStreamCreateWithFlags(&tap_copy_stream, cudaStreamNonBlocking);
cudaEventCreate(&tap_copy_done);
```

### Producer (callback)

```cpp
if (std::strcmp(t->name, "h_pre_norm") == 0) {
    if (ask) return true;
    if (!self->tap_h_pre_norm_file.is_open()) return true;

    const size_t nbytes = ggml_nbytes(t);
    if (t->type != GGML_TYPE_F16) {
        // Fall back to sync path for F32/BF16 conversion (uncommon for h_pre_norm)
        // ... existing sync code ...
        return true;
    }

    // Slot in ring buffer
    size_t slot = self->tap_ring_idx.fetch_add(1) % 4;
    void * dst  = (uint8_t *) self->tap_pinned_host + slot * (self->tap_pinned_size / 4);

    // Wait for compute stream to finish producing t (cross-stream sync via event)
    cudaEvent_t produced;
    cudaEventCreate(&produced);
    cudaEventRecord(produced, /* compute stream from ggml-cuda backend */ 0);
    cudaStreamWaitEvent(self->tap_copy_stream, produced, 0);

    // Async D->H into pinned slot
    cudaMemcpyAsync(dst, t->data, nbytes, cudaMemcpyDeviceToHost,
                    self->tap_copy_stream);

    // Record event for writer thread to wait on
    cudaEvent_t done;
    cudaEventCreate(&done);
    cudaEventRecord(done, self->tap_copy_stream);

    // Enqueue payload pointing at the pinned slot + the event
    TapPayload p{done, nbytes, slot * (self->tap_pinned_size / 4)};
    {
        std::unique_lock<std::mutex> lk(self->tap_q_mtx);
        self->tap_q_cv.wait(lk, [&]{ return self->tap_q.size() < TAP_Q_MAX; });
        self->tap_q.push(p);
    }
    self->tap_q_cv.notify_one();
    cudaEventDestroy(produced);   // not needed after wait
    return true;   // RETURN IMMEDIATELY — copy continues async
}
```

### Writer thread body

```cpp
void tap_writer_loop() {
    while (true) {
        TapPayload p;
        {
            std::unique_lock<std::mutex> lk(tap_q_mtx);
            tap_q_cv.wait(lk, [&]{ return !tap_q.empty() || tap_writer_stop.load(); });
            if (tap_writer_stop.load() && tap_q.empty()) return;
            p = tap_q.front(); tap_q.pop();
        }
        tap_q_cv.notify_one();

        cudaEventSynchronize(p.copy_done);                  // wait for D->H landing
        cudaEventDestroy(p.copy_done);

        const char * src = (const char *) tap_pinned_host + p.pinned_off;
        if (tap_h_pre_norm_file.is_open()) {
            tap_h_pre_norm_file.write(src, p.n_bytes);
        }
    }
}
```

### Lifecycle (additions)

- Shutdown: stop writer, drain queue (wait for last events), free pinned mem
  + destroy stream + event.
- Resize: if model n_embd or window changes, realloc pinned buffer.

### Invariants

- Ring buffer prevents stomp: 4 slots × max-window-size, queue bounded ≤4
  ensures producer never reuses a slot that writer hasn't drained.
- Order: single producer + single consumer + FIFO = preserved.
- Cross-stream dependency: `cudaStreamWaitEvent` ensures copy waits for
  compute kernel producing t. Without it, copy reads stale device memory.

### Risks

- **CUDA stream interaction**: ggml-cuda backend may reuse t->data buffer
  AFTER callback returns (the original code's GGML_BACKEND_BUFFER_COPY_INSITU
  pattern). If so, copy submitted async could read corrupted data. Need to
  verify that h_pre_norm tensor's backing buffer is preserved at least until
  the next forward pass, OR use ggml backend's own event-record API instead
  of raw cudaEventRecord on stream 0.
- **Pinned memory limits**: cudaHostAlloc reserves non-pageable RAM. 67MB
  for ring is fine.
- **Multi-GPU / multi-stream**: callback can fire from non-default stream
  in some backends. Need to record event on the correct stream.

### Mitigation

Smoke run on HCL slug, capture 1M tokens, validate via:
```bash
python3 validate_capture.py --strict --sample-rows 5000
```
Compare h_pre_norm sample row distributions to a reference run with sync
path. If means/L2/std diverge >0.1%, copy timing bug → revert.

### Expected gain

10-25% additional on top of #2. Mostly from overlap: while writer is
syncing+writing window N's copy, GPU is already running window N+1's
forward pass. Pipeline depth = 2-3 windows in flight.

### Combined #1+#2 gain estimate

15-30% over current. Hits the practical ceiling of "what software can do"
without GPU swap.

---

## Out-of-scope (rejected)

- **#4 lower precision tap**: rejected (training quality risk)
- **#5 GPU swap**: rejected ($$, time)

## Files to touch

- `tools/server/server-context.cpp` — class members + callback + writer loop
- `tools/server/server-context.h` — forward decls if any
- Build: same `cmake --build build-cuda --target llama-server -j 16` inside
  container; <3 min incremental.

## Test plan

1. Apply #2 only. Smoke HCL 5min. Validator clean. Bench tps before/after.
2. If #2 gains as expected (5-10%), apply #1. Smoke + validator + sample-row
   distribution check vs sync baseline.
3. If both pass → next full sweep uses combined.

## Rollback

Each patch is self-contained additions to server-context.cpp. Revert =
`git diff` then `git checkout`. Build artifact at `build-cuda/bin/llama-server`
swappable instantly.
