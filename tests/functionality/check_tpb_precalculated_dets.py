#!/usr/bin/env python3
"""Check a UHF TPB diag binary against exact diagonal energies in both determinant modes.

Example (inside a GPU allocation, using a UHF binary):
  python3 check_tpb_precalculated_dets.py --diag /path/to/diag --work-dir /project/test \
    --launcher srun --mpi=pmix --nodes=1 --ntasks=1

The launcher is optional for MPI singleton execution. Only synthetic inputs are used.
"""

import argparse
import itertools
import json
import math
import os
import re
import subprocess
from pathlib import Path

CASES = {
    "single-word": (6, 20),
    "odd-full-width": (6, 4),
    "fe4s4-width": (36, 20),
    "three-half-words": (36, 12),
}


def make_inputs(directory, norb, alpha, beta):
    for name, determinants in [("AlphaDets.bin", alpha), ("BetaDets.bin", beta)]:
        (directory / name).write_bytes(
            b"".join(value.to_bytes((norb + 7) // 8, "big") for value in determinants)
        )
    lines = [f" &FCI NORB={norb},NELEC=4,MS2=0,", " &END"]
    for orbital in range(norb):
        for spin in [0, 1]:
            index = 2 * orbital + spin + 1
            energy = (orbital + 1) if spin == 0 else (10 + orbital)
            lines.append(f" {energy:.16e} {index} {index} 0 0")
    lines.append(" 2.5000000000000000e-1 0 0 0 0")
    (directory / "fcidump.txt").write_text("\n".join(lines) + "\n")


def expected_energy(alpha, beta, norb):
    return (
        0.25
        + sum(orbital + 1 for orbital in range(norb) if alpha >> orbital & 1)
        + sum(10 + orbital for orbital in range(norb) if beta >> orbital & 1)
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--diag", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--cases", nargs="+", choices=CASES, default=list(CASES))
    parser.add_argument("--modes", nargs="+", type=int, choices=[0, 1], default=[1, 0])
    parser.add_argument("--timeout", type=float, default=120)
    parser.add_argument("--launcher", nargs=argparse.REMAINDER, default=[])
    options = parser.parse_args()
    options.work_dir.mkdir(parents=True, exist_ok=True)
    binary = options.diag.resolve()
    environment = os.environ.copy()
    environment.setdefault("OMP_NUM_THREADS", "2")
    records = []
    for case in options.cases:
        norb, bit_length = CASES[case]
        alpha = [3, (1 << 1) | (1 << (norb - 2)), (1 << 2) | (1 << (norb - 1))]
        beta = [3, 1 | (1 << (norb - 2)), (1 << 2) | (1 << (norb - 1))]
        for (ia, ib), mode in itertools.product([(0, 2), (2, 0), (2, 2)], options.modes):
            directory = (options.work_dir / f"{case}-a{ia}-b{ib}-mode{mode}").resolve()
            directory.mkdir()  # Never replace evidence from an earlier run.
            make_inputs(directory, norb, alpha, beta)
            command = options.launcher + [
                str(binary),
                "--adetfile",
                "AlphaDets.bin",
                "--bdetfile",
                "BetaDets.bin",
                "--task_comm_size",
                "1",
                "--adet_comm_size",
                "1",
                "--bdet_comm_size",
                "1",
                "--block",
                "4",
                "--iteration",
                "2",
                "--tolerance",
                "1e-10",
                "--carryover_type",
                "0",
                "--rdm",
                "0",
                "--bit_length",
                str(bit_length),
                "--shuffle",
                "1",
                "--initial_adeterminant_bitstring",
                format(alpha[ia], f"0{norb}b"),
                "--initial_bdeterminant_bitstring",
                format(beta[ib], f"0{norb}b"),
                "--use_precalculated_dets",
                str(mode),
            ]
            (directory / "command.json").write_text(json.dumps(command, indent=2) + "\n")
            result = subprocess.run(
                command,
                cwd=directory,
                env=environment,
                text=True,
                capture_output=True,
                timeout=options.timeout,
            )
            (directory / "stdout.log").write_text(result.stdout)
            (directory / "stderr.log").write_text(result.stderr)
            energies = re.findall(
                r"Sample-based diagonalization: Energy =\s*([-+\d.eE]+)", result.stdout
            )
            residuals = re.findall(r"Davidson iteration .*?\(tol=([^\)]+)\)", result.stdout)
            energy = float(energies[-1]) if energies else None
            residual = float(residuals[-1]) if residuals else None
            expected = expected_energy(alpha[ia], beta[ib], norb)
            passed = (
                result.returncode == 0
                and energy is not None
                and math.isfinite(energy)
                and abs(energy - expected) < 1e-9
                and residual is not None
                and math.isfinite(residual)
                and residual < 1e-10
            )
            record = dict(
                case=case,
                mode=mode,
                alpha_index=ia,
                beta_index=ib,
                expected=expected,
                observed=energy,
                residual=residual,
                exit_code=result.returncode,
                passed=passed,
            )
            records.append(record)
            (options.work_dir / "results.json").write_text(json.dumps(records, indent=2) + "\n")
            print(json.dumps(record), flush=True)
            if not passed:
                raise SystemExit(1)
    print(f"PASS: {len(records)} exact-diagonal checks", flush=True)


if __name__ == "__main__":
    main()
