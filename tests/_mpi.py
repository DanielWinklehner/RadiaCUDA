# tests/_mpi.py
#
# MPI-safety helper for RadiaCUDA test scripts. Import and use this in every test.
#
# Why it's needed: under `mpiexec -n N`, Radia's GPU field-eval and solve run on
# rank 0 and broadcast only success/failure to the other ranks (radgpu_fld.cpp,
# radintrc.cpp). Two consequences a test MUST respect:
#   1. Every rank has to issue the SAME sequence of rad.Fld / rad.Solve calls, so
#      those internal MPI collectives stay balanced -- otherwise they deadlock.
#      => run all field/solve calls on every rank, unconditionally.
#   2. Only rank 0 holds valid results, so only rank 0 may assert / decide pass-fail.
#      => guard every assertion and the exit code with the helper below.
#
# Single-process (`python test.py`) is a no-op: rank 0, MPI never initialized.
# Under `mpiexec` it initializes Radia MPI, tags rank 0 as the reporter, and
# finalizes on all ranks.
#
# Usage:
#     from _mpi import MpiTest
#     mpi = MpiTest()                                  # before any rad.Fld/Solve
#     ...                                              # run ALL calls on every rank
#     mpi.check("name", ok, "detail")                 # asserts on rank 0 only
#     mpi.say("message")                              # prints on rank 0 only
#     return mpi.finish()                             # finalizes MPI; returns exit code

import os
import radia as rad


def _under_mpiexec():
    # Process launchers export these: Intel MPI ('impi') and MS-MPI -> PMI_*,
    # Open MPI -> OMPI_*. Present on every rank, so the check is rank-consistent.
    keys = ("PMI_RANK", "PMI_SIZE", "PMI_LOCAL_RANK",
            "OMPI_COMM_WORLD_RANK", "OMPI_COMM_WORLD_SIZE", "MPI_LOCALRANKID")
    return any(k in os.environ for k in keys)


class MpiTest:
    def __init__(self):
        self.under_mpi = _under_mpiexec()
        # rad.UtiMPI('on') initializes MPI and returns this process's rank. Only call
        # it when actually launched under mpiexec so a plain `python test.py` run stays
        # a normal single process. If it fails (e.g. non-MPI build), degrade to serial:
        # every process then runs the whole test independently, which is safe (no
        # collectives are issued) if redundant.
        self.rank = 0
        if self.under_mpi:
            try:
                self.rank = int(rad.UtiMPI('on'))
            except Exception:
                self.under_mpi = False
                self.rank = 0
        self.is_root = (self.rank <= 0)
        self._failures = []

    def check(self, name, ok, detail=""):
        """Record + print one assertion, on rank 0 only (other ranks hold no valid data)."""
        if not self.is_root:
            return
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" :: {detail}" if detail else ""),
              flush=True)
        if not ok:
            self._failures.append(name)

    def say(self, *args):
        """Print on rank 0 only."""
        if self.is_root:
            print(*args, flush=True)

    @property
    def failures(self):
        return list(self._failures)

    def finish(self, extra_code=0):
        """Finalize MPI on ALL ranks (collective) and return the process exit code.
        Non-root ranks return 0; rank 0 returns 1 iff it recorded a failure."""
        if self.under_mpi:
            try:
                rad.UtiMPI('off')
            except Exception:
                pass
        if not self.is_root:
            return 0
        return 1 if self._failures else extra_code
