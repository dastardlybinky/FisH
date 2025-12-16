import os
import re
import sys
import subprocess
from openmm import Platform, Vec3, LangevinIntegrator
from openmm.app import AmberPrmtopFile, AmberInpcrdFile, Simulation, PME, HBonds, StateDataReporter, PDBFile, Modeller
from openmm.unit import kelvin, picosecond, femtoseconds, nanometer
import mdtraj as md
import numpy as np
from mdtraj.reporters import DCDReporter as MDTrajDCDReporter
from mdtraj.formats import DCDTrajectoryFile


STEPS_PER_TRAJ = 5_000_000
DCD_REPORT_INTERVAL = 25_000
EXPECTED_FRAMES = STEPS_PER_TRAJ // DCD_REPORT_INTERVAL


def _traj_dcd_path(base_traj_prefix: str, traj_idx: int) -> str:
    if traj_idx == 0:
        return base_traj_prefix + ".dcd"
    return f"{base_traj_prefix}{traj_idx}.dcd"


def _traj_chk_path(base_traj_prefix: str, traj_idx: int) -> str:
    return f"{base_traj_prefix}{traj_idx}.chk"


def _count_dcd_frames(dcd_path: str) -> int:
    try:
        with DCDTrajectoryFile(dcd_path) as f:
            return int(f.n_frames)
    except Exception:
        return 0


def _is_complete_traj(dcd_path: str) -> bool:
    if not os.path.exists(dcd_path):
        return False
    return _count_dcd_frames(dcd_path) >= EXPECTED_FRAMES


def _delete_if_exists(path: str, label: str = ""):
    if os.path.exists(path):
        try:
            os.remove(path)
            if label:
                print(f"{label}{path}")
        except OSError:
            pass


def _max_traj_index_in_dir(output_dir: str) -> int:
    max_idx = -1
    for fn in os.listdir(output_dir):
        if not fn.endswith(".dcd"):
            continue
        if fn.endswith("_traj.dcd"):
            max_idx = max(max_idx, 0)
        else:
            m = re.search(r"_traj(\d+)\.dcd$", fn)
            if m:
                max_idx = max(max_idx, int(m.group(1)))
    return max_idx


def _base_names_in_dir(output_dir: str) -> set[str]:
    bases = set()
    for fn in os.listdir(output_dir):
        if fn.endswith(".chk"):
            bases.add(fn[:-4])
    return bases


def simulate_one_target(prmtop_file, inpcrd_file, device_mode, gpu_id, output_dir, target_traj: int, legacy_checkpoint: bool):
    prmtop = AmberPrmtopFile(prmtop_file)
    inpcrd = AmberInpcrdFile(inpcrd_file)
    topology = prmtop.topology
    positions = inpcrd.positions

    solvent_resnames = {"WAT", "HOH", "SOL", "TIP3", "TIP4P"}
    solute_atom_indices = [a.index for a in topology.atoms() if a.residue.name not in solvent_resnames]
    water_atom_indices = [a.index for a in topology.atoms() if a.residue.name in solvent_resnames]

    system = prmtop.createSystem(nonbondedMethod=PME, nonbondedCutoff=1 * nanometer, constraints=HBonds)

    if inpcrd.boxVectors is not None:
        system.setDefaultPeriodicBoxVectors(*inpcrd.boxVectors)
        box_vectors = inpcrd.boxVectors
    else:
        box = Vec3(3.0, 3.0, 3.0) * nanometer
        system.setDefaultPeriodicBoxVectors(box, box, box)
        box_vectors = (box, box, box)

    integrator = LangevinIntegrator(300 * kelvin, 1 / picosecond, 2 * femtoseconds)

    if device_mode == "cpu":
        platform = Platform.getPlatformByName("CPU")
        properties = {}
        print("Running on CPU")
    else:
        platform = Platform.getPlatformByName("CUDA")
        properties = {"CudaDeviceIndex": str(gpu_id), "CudaPrecision": "mixed"}
        print(f"Running on GPU {gpu_id}")

    simulation = Simulation(topology, system, integrator, platform, properties)

    base = os.path.basename(prmtop_file).replace(".prmtop", "")
    os.makedirs(output_dir, exist_ok=True)

    base_traj_prefix = os.path.join(output_dir, f"{base}_traj")
    latest_chk = os.path.join(output_dir, f"{base}.chk")

    dcd_out = _traj_dcd_path(base_traj_prefix, target_traj)
    seg_chk = _traj_chk_path(base_traj_prefix, target_traj)

    if _is_complete_traj(dcd_out):
        print(f"[{base}] traj{target_traj} already complete (>= {EXPECTED_FRAMES} frames).")
        return

    if os.path.exists(dcd_out):
        _delete_if_exists(dcd_out, label=f"[{base}] Removed incomplete/corrupt ")

    if target_traj == 0:
        simulation.context.setPositions(positions)
        if box_vectors is not None:
            simulation.context.setPeriodicBoxVectors(*box_vectors)
        simulation.minimizeEnergy()
    else:
        ck_to_load = None
        if legacy_checkpoint:
            if os.path.exists(latest_chk):
                ck_to_load = latest_chk
        else:
            prev_chk = _traj_chk_path(base_traj_prefix, target_traj - 1)
            if os.path.exists(prev_chk):
                ck_to_load = prev_chk
            elif os.path.exists(latest_chk):
                ck_to_load = latest_chk

        if ck_to_load is None:
            print(f"[{base}] ERROR: missing checkpoint to run traj{target_traj}. Needed prev segment checkpoint or {base}.chk.")
            return

        print(f"[{base}] Loading checkpoint for traj{target_traj}: {ck_to_load}")
        simulation.loadCheckpoint(ck_to_load)

    simulation.reporters = []
    simulation.reporters.append(StateDataReporter(sys.stdout, 100000, step=True, potentialEnergy=True, temperature=True, speed=True))
    simulation.reporters.append(MDTrajDCDReporter(dcd_out, DCD_REPORT_INTERVAL, atomSubset=solute_atom_indices))

    print(f"[{base}] Running traj{target_traj} -> {os.path.basename(dcd_out)}")
    simulation.step(STEPS_PER_TRAJ)

    if legacy_checkpoint:
        simulation.saveCheckpoint(latest_chk)
    else:
        simulation.saveCheckpoint(seg_chk)
        simulation.saveCheckpoint(latest_chk)

    n_frames = _count_dcd_frames(dcd_out)
    if n_frames < EXPECTED_FRAMES:
        print(f"[{base}] ERROR: traj{target_traj} wrote only {n_frames}/{EXPECTED_FRAMES} frames.")
        _delete_if_exists(dcd_out, label=f"[{base}] Removed incomplete ")
        if not legacy_checkpoint:
            _delete_if_exists(seg_chk, label=f"[{base}] Removed ")
        return

    pdb_out = os.path.join(output_dir, f"{base}_final.pdb")
    pdb_out_solv = os.path.join(output_dir, f"{base}_final_solv.pdb")
    water_dcd_out = os.path.join(output_dir, f"{base}_water_lastframe.dcd")

    state = simulation.context.getState(getPositions=True, enforcePeriodicBox=True)
    all_positions = state.getPositions()

    with open(pdb_out_solv, "w") as f:
        PDBFile.writeFile(simulation.topology, all_positions, f)

    modeller = Modeller(simulation.topology, all_positions)
    modeller.deleteWater()

    with open(pdb_out, "w") as f:
        PDBFile.writeFile(modeller.topology, modeller.positions, f)

    if len(water_atom_indices) > 0:
        xyz = np.array(all_positions.value_in_unit(nanometer))
        if xyz.ndim == 2:
            xyz = xyz[np.newaxis, :, :]

        md_top = md.Topology.from_openmm(topology)
        traj = md.Trajectory(xyz=xyz, topology=md_top)
        water_traj = traj.atom_slice(water_atom_indices)

        try:
            box = state.getPeriodicBoxVectors()
            if box is not None:
                def _to_nm(component):
                    if hasattr(component, "value_in_unit"):
                        return float(component.value_in_unit(nanometer))
                    return float(component)

                box_nm = np.array(
                    [
                        [_to_nm(box[0].x), _to_nm(box[0].y), _to_nm(box[0].z)],
                        [_to_nm(box[1].x), _to_nm(box[1].y), _to_nm(box[1].z)],
                        [_to_nm(box[2].x), _to_nm(box[2].y), _to_nm(box[2].z)],
                    ],
                    dtype=float,
                )
                water_traj.unitcell_vectors = box_nm.reshape(1, 3, 3)
        except Exception:
            pass

        water_traj.save_dcd(water_dcd_out)

    print(f"[{base}] Done traj{target_traj} ({n_frames} frames).")


def run_all(system_type, device_mode, continue_mode=False, num_gpus=1, legacy_checkpoint=False):
    base_dir = {"solo": "systems/solo_sys", "combo": "systems/combo_sys"}.get(system_type)
    if base_dir is None:
        print("ERROR: choose 'solo' or 'combo'")
        sys.exit(1)

    output_dir = os.path.join("sim_output", os.path.basename(base_dir))
    os.makedirs(output_dir, exist_ok=True)

    prmtops = sorted([f for f in os.listdir(base_dir) if f.endswith(".prmtop")])

    if device_mode == "cpu":
        num_gpus = 1

    target_traj = None
    if continue_mode:
        target_traj = _max_traj_index_in_dir(output_dir)
        if target_traj < 0:
            print("[BATCH] --continue: no existing traj files found; nothing to repair.")
            return
        print(f"[BATCH] --continue: global highest traj index = {target_traj}; repairing/rerunning only traj{target_traj}")

    processes = []
    for i, prmt in enumerate(prmtops):
        inp = prmt.replace(".prmtop", ".inpcrd")
        prmt_path = os.path.join(base_dir, prmt)
        inp_path = os.path.join(base_dir, inp)
        gpu_id = i % num_gpus

        cmd = [sys.executable, "simulate_systems.py", prmt_path, inp_path, device_mode, str(gpu_id), output_dir]

        if continue_mode:
            cmd += ["--continue", "--target-traj", str(target_traj)]

        if legacy_checkpoint:
            cmd.append("--legacy-checkpoint")

        p = subprocess.Popen(cmd)
        processes.append(p)

        if len(processes) >= num_gpus:
            for p in processes:
                p.wait()
            processes = []

    for p in processes:
        p.wait()


def _parse_int_arg(flag: str, argv: list[str], default=None):
    if flag not in argv:
        return default
    i = argv.index(flag)
    if i + 1 >= len(argv):
        return default
    try:
        return int(argv[i + 1])
    except ValueError:
        return default


if __name__ == "__main__":
    if len(sys.argv) >= 6 and sys.argv[1].endswith(".prmtop"):
        prmtop_file = sys.argv[1]
        inpcrd_file = sys.argv[2]
        device_mode = sys.argv[3]
        gpu_id = int(sys.argv[4])
        output_dir = sys.argv[5]
        extra = sys.argv[6:]

        continue_mode = "--continue" in extra
        legacy_checkpoint = "--legacy-checkpoint" in extra
        target_traj = _parse_int_arg("--target-traj", extra, default=None)

        if continue_mode:
            if target_traj is None:
                target_traj = _max_traj_index_in_dir(output_dir)
                if target_traj < 0:
                    print("No existing traj files found; nothing to repair.")
                    sys.exit(0)
            base = os.path.basename(prmtop_file).replace(".prmtop", "")
            base_traj_prefix = os.path.join(output_dir, f"{base}_traj")
            dcd_out = _traj_dcd_path(base_traj_prefix, target_traj)
            if not _is_complete_traj(dcd_out):
                simulate_one_target(prmtop_file, inpcrd_file, device_mode, gpu_id, output_dir, target_traj, legacy_checkpoint)
            else:
                print(f"[{base}] traj{target_traj} already complete (>= {EXPECTED_FRAMES} frames).")
        else:
            print("This script is configured for --continue repair of the global highest traj index. Use batch mode with --continue.")
    else:
        if len(sys.argv) < 2:
            print("Usage:")
            print("  python simulate_systems.py solo [cpu|gpu [NUM_GPUS]] --continue [--legacy-checkpoint]")
            print("  python simulate_systems.py combo [cpu|gpu [NUM_GPUS]] --continue [--legacy-checkpoint]")
            sys.exit(1)

        system_type = sys.argv[1]
        device_mode = "gpu"
        num_gpus = 1
        continue_mode = False
        legacy_checkpoint = False

        args = sys.argv[2:]
        i = 0
        while i < len(args):
            arg = args[i]
            if arg == "--continue":
                continue_mode = True
                i += 1
            elif arg == "--legacy-checkpoint":
                legacy_checkpoint = True
                i += 1
            elif arg.lower() in ("cpu", "gpu"):
                device_mode = arg.lower()
                i += 1
                if device_mode == "gpu" and i < len(args) and args[i] not in ("--continue", "--legacy-checkpoint"):
                    try:
                        num_gpus = int(args[i])
                        i += 1
                    except ValueError:
                        i += 1
            else:
                i += 1

        if not continue_mode:
            print("ERROR: batch mode requires --continue for this script.")
            sys.exit(1)

        run_all(system_type, device_mode, continue_mode, num_gpus, legacy_checkpoint)
