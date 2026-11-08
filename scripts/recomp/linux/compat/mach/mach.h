// Linux replacement for the small part of the Mach API that scripts/recomp/runtime (kernel.c, main.c) uses.
// Own code, only for the Linux build (docs/recomp-linux.md): makes it possible to compile the runtime unchanged on Linux
// (`-I scripts/recomp/linux/compat`) without touching its files. Not a replacement for the Mach semantics on the
// Mac: suspending via signal (SIGUSR2), memory usage from /proc, regions via mincore.
#pragma once
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include <errno.h>
#include <pthread.h>
#include <signal.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <sys/mman.h>
#include <time.h>
#include <unistd.h>

typedef int kern_return_t;
extern int pthread_getattr_np(pthread_t, pthread_attr_t *);
typedef uint32_t mach_port_t;
typedef uint32_t mach_msg_type_number_t;
typedef int *thread_state_t;
#define KERN_SUCCESS 0
#define KERN_FAILURE 5

// Host page size: 4 KiB on x86-64, 16 KiB or 64 KiB on some AArch64 kernels (Asahi), so never a constant.
static inline uint64_t compat_page_size(void) {
    static uint64_t size;
    if (!size) { long v = sysconf(_SC_PAGESIZE); size = v > 0 ? (uint64_t)v : 4096; }
    return size;
}

// ---- Time (mach_time.h)
typedef struct { uint32_t numer, denom; } mach_timebase_info_data_t;
static inline kern_return_t mach_timebase_info(mach_timebase_info_data_t *i) { i->numer = i->denom = 1; return KERN_SUCCESS; }
static inline uint64_t mach_absolute_time(void) {
    struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t);
    return (uint64_t)t.tv_sec * 1000000000ull + (uint64_t)t.tv_nsec;
}

// ---- Suspend threads: a signal handler blocks the target thread until thread_resume releases it
#define COMPAT_MAX_THREADS 256
static pthread_t compat_threads[COMPAT_MAX_THREADS];
static volatile int compat_stopped[COMPAT_MAX_THREADS], compat_go[COMPAT_MAX_THREADS];
static int compat_nthreads;
static pthread_mutex_t compat_lock = PTHREAD_MUTEX_INITIALIZER;

// Spin-wait hint of the host CPU (x86 pause, AArch64 yield); other architectures only get the compiler barrier.
static inline void compat_cpu_relax(void) {
#if defined(__x86_64__) || defined(__i386__)
    __asm__ volatile("pause" ::: "memory");
#elif defined(__aarch64__)
    __asm__ volatile("yield" ::: "memory");
#else
    __asm__ volatile("" ::: "memory");
#endif
}

// Mach semantics the GC relies on (SetThreadActivity, kernel.c): thread_suspend returns only once the thread is stopped, and its
// register state (Cpu) is visible to the suspender then; thread_resume returns once the thread has left the handler, so a
// suspend right after it cannot be lost. Before, thread_suspend returned at once, and a quick suspend/resume/suspend/resume left
// the thread stopped for good: the second signal stayed pending, and its handler ran after go had been reset (reproduced on the
// Mac with a copy of these functions, stuck within 10 sequences; none in 2000 with this version). The flags use acquire/release
// atomics: volatile alone orders nothing on AArch64.
static inline void compat_on_stop(int sig) {
    (void)sig;
    pthread_t me = pthread_self();
    int s = -1, n = __atomic_load_n(&compat_nthreads, __ATOMIC_ACQUIRE);
    for (int i = 0; i < n; i++) if (pthread_equal(compat_threads[i], me)) { s = i; break; }
    if (s < 0) return;
    __atomic_store_n(&compat_stopped[s], 1, __ATOMIC_RELEASE);
    while (!__atomic_load_n(&compat_go[s], __ATOMIC_ACQUIRE)) compat_cpu_relax();
    __atomic_store_n(&compat_go[s], 0, __ATOMIC_RELAXED);
    __atomic_store_n(&compat_stopped[s], 0, __ATOMIC_RELEASE);
}
static inline mach_port_t pthread_mach_thread_np(pthread_t t) {
    pthread_mutex_lock(&compat_lock);
    int i;
    for (i = 0; i < compat_nthreads; i++) if (pthread_equal(compat_threads[i], t)) break;
    if (i == compat_nthreads && i < COMPAT_MAX_THREADS) {
        compat_threads[i] = t; __atomic_store_n(&compat_nthreads, compat_nthreads + 1, __ATOMIC_RELEASE);
        struct sigaction sa; memset(&sa, 0, sizeof sa);
        sa.sa_handler = compat_on_stop;
        sa.sa_flags = SA_RESTART;   // a suspended thread's read/write/... resume instead of failing with EINTR
        sigaction(SIGUSR2, &sa, NULL);
    }
    pthread_mutex_unlock(&compat_lock);
    return (mach_port_t)(i + 1);
}
static inline kern_return_t thread_suspend(mach_port_t p) {
    int s = (int)p - 1;
    if (s < 0 || s >= compat_nthreads) return KERN_FAILURE;
    __atomic_store_n(&compat_go[s], 0, __ATOMIC_RELAXED);
    if (pthread_kill(compat_threads[s], SIGUSR2)) return KERN_FAILURE;
    // Bounded: a thread that exited meanwhile (glibc >= 2.34 reports success for it) or blocks the signal never stops.
    uint64_t give_up = mach_absolute_time() + 2000000000ull;
    for (unsigned spins = 0; !__atomic_load_n(&compat_stopped[s], __ATOMIC_ACQUIRE); spins++) {
        compat_cpu_relax();
        if (!(spins & 4095) && mach_absolute_time() > give_up) return KERN_FAILURE;
    }
    return KERN_SUCCESS;
}
static inline kern_return_t thread_resume(mach_port_t p) {
    int s = (int)p - 1;
    if (s < 0 || s >= compat_nthreads) return KERN_FAILURE;
    __atomic_store_n(&compat_go[s], 1, __ATOMIC_RELEASE);
    while (__atomic_load_n(&compat_stopped[s], __ATOMIC_ACQUIRE)) compat_cpu_relax();
    return KERN_SUCCESS;
}

// ---- Stack size/address (Darwin calls; stackaddr = upper bound as on macOS)
static inline size_t pthread_get_stacksize_np(pthread_t t) {
    pthread_attr_t a; void *lo = NULL; size_t sz = 0;
    if (pthread_getattr_np(t, &a) != 0) return 0;
    pthread_attr_getstack(&a, &lo, &sz); pthread_attr_destroy(&a);
    return sz;
}
static inline void *pthread_get_stackaddr_np(pthread_t t) {
    pthread_attr_t a; void *lo = NULL; size_t sz = 0;
    if (pthread_getattr_np(t, &a) != 0) return NULL;
    pthread_attr_getstack(&a, &lo, &sz); pthread_attr_destroy(&a);
    return (char *)lo + sz;
}

// ---- Prozessinformation (task_info)
#define MACH_TASK_BASIC_INFO 20
#define MACH_TASK_BASIC_INFO_COUNT 12
typedef struct { uint64_t resident_size, virtual_size, resident_size_max; } mach_task_basic_info_data_t;
typedef int *task_info_t;
static inline mach_port_t mach_task_self(void) { return 1; }
// TASK_VM_INFO: only the two fields the runtime reads. phys_footprint is the resident set (what Linux and Android's low-memory killer
// count), internal the anonymous part of it (RssAnon in /proc/self/status, the Mach "internal" pages).
#define TASK_VM_INFO 22
#define TASK_VM_INFO_COUNT 2
typedef struct { uint64_t phys_footprint, internal; } task_vm_info_data_t;
static inline kern_return_t task_info(mach_port_t t, int flavor, task_info_t out, mach_msg_type_number_t *cnt) {
    (void)t; (void)cnt;
    FILE *f = fopen("/proc/self/statm", "r");
    if (!f) return KERN_FAILURE;
    unsigned long vm = 0, rss = 0;
    if (fscanf(f, "%lu %lu", &vm, &rss) != 2) { fclose(f); return KERN_FAILURE; }
    fclose(f);
    if (flavor == TASK_VM_INFO) {
        task_vm_info_data_t *v = (task_vm_info_data_t *)out;
        v->phys_footprint = rss * compat_page_size();
        v->internal = 0;
        if ((f = fopen("/proc/self/status", "r"))) {
            char line[128]; unsigned long kb;
            while (fgets(line, sizeof line, f))
                if (sscanf(line, "RssAnon: %lu kB", &kb) == 1) { v->internal = (uint64_t)kb << 10; break; }
            fclose(f);
        }
        return KERN_SUCCESS;
    }
    if (flavor != MACH_TASK_BASIC_INFO) return KERN_FAILURE;
    mach_task_basic_info_data_t *i = (mach_task_basic_info_data_t *)out;
    i->virtual_size = vm * compat_page_size(); i->resident_size = rss * compat_page_size(); i->resident_size_max = i->resident_size;
    return KERN_SUCCESS;
}
