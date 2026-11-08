"""Linux Mach compat layer (scripts/recomp/linux/compat/mach/mach.h): thread suspension with Mach semantics and task_info."""
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
COMPAT = ROOT / "scripts/recomp/linux/compat"

# The GC in the game stops threads through SetThreadActivity (kernel.c): thread_suspend, read the registers, thread_resume,
# several hundred times a session and often back to back. Mach's thread_suspend returns once the thread is stopped.
PROGRAM = r"""
#include <mach/mach.h>
#include <stdlib.h>

static volatile uint64_t counter;
static void *worker(void *arg) { (void)arg; for (;;) { counter++; __asm__ volatile("" ::: "memory"); } return NULL; }
static uint64_t now_ns(void) { return mach_absolute_time(); }

int main(void) {
    pthread_t t;
    if (pthread_create(&t, NULL, worker, NULL)) return 10;
    mach_port_t port = pthread_mach_thread_np(t);
    while (counter < 1000) {}
    for (int i = 0; i < 500; i++) {   /* stopped when thread_suspend returns, and it stays stopped */
        if (thread_suspend(port) != KERN_SUCCESS) return 11;
        uint64_t before = counter, until = now_ns() + 200000;
        while (now_ns() < until) {}
        if (counter != before) { printf("thread ran after thread_suspend returned (round %d)\n", i); return 1; }
        if (thread_resume(port) != KERN_SUCCESS) return 12;
    }
    for (int i = 0; i < 2000; i++) {   /* back-to-back pairs must not leave the thread stopped */
        thread_suspend(port); thread_resume(port); thread_suspend(port); thread_resume(port);
        uint64_t before = counter, until = now_ns() + 2000000000ull;
        while (counter == before && now_ns() < until) {}
        if (counter == before) { printf("thread stuck after a suspend/resume sequence (round %d)\n", i); return 2; }
    }
    task_vm_info_data_t vm;
    mach_msg_type_number_t count = TASK_VM_INFO_COUNT;
    if (task_info(mach_task_self(), TASK_VM_INFO, (task_info_t)&vm, &count) != KERN_SUCCESS || !vm.phys_footprint) {
        printf("TASK_VM_INFO failed\n");
        return 3;
    }
    printf("ok\n");
    return 0;
}
"""


@unittest.skipUnless(sys.platform.startswith("linux"), "the compat layer replaces Mach only on Linux")
class LinuxCompatMach(unittest.TestCase):
    def test_suspend_resume_and_task_vm_info(self):
        cc = os.environ.get("CC") or shutil.which("cc") or shutil.which("gcc") or shutil.which("clang")
        if not cc:
            self.skipTest("no C compiler")
        with tempfile.TemporaryDirectory() as folder:
            source, binary = Path(folder) / "compat.c", Path(folder) / "compat"
            source.write_text(PROGRAM)
            subprocess.run([cc, "-O2", "-pthread", "-I", str(COMPAT), str(source), "-o", str(binary)], check=True)
            run = subprocess.run([str(binary)], capture_output=True, text=True, timeout=120)
            self.assertEqual(run.returncode, 0, run.stdout + run.stderr)


if __name__ == "__main__":
    unittest.main()
