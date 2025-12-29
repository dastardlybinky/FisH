import os
import sys
import subprocess
from openmm import *
from openmm.app import *
from openmm.unit import *
import mdtraj as md
import numpy as np
from mdtraj.reporters import DCDReporter as MDTrajDCDReporter

def simulate(prmtop_file, inpcrd_file, device_mode, gpu_id, output_dir, continue_mode=False):
    prmtop = AmberPrmtopFile(prmtop_file)
    inpcrd = AmberInpcrdFile(inpcrd_file)
    topology = prmtop.topology
    positions = inpcrd.positions

    solvent_resnames = {"WAT", "HOH", "SOL", "TIP3", "TIP4P"}
    solute_atom_indices = [
        atom.index
        for atom in topology.atoms()
        if atom.residue.name not in solvent_resnames
    ]
    water_atom_indices = [
        atom.index
        for atom in topology.atoms()
        if atom.residue.name in solvent_resnames
    ]

    system = prmtop.createSystem(
        nonbondedMethod=PME,
        nonbondedCutoff=1 * nanometer,
        constraints=HBonds,
    )

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
    base_traj_prefix = os.path.join(output_dir, f"{base}_traj")
    if continue_mode:
        first_candidate = base_traj_prefix + ".dcd"
        if not os.path.exists(first_candidate):
            dcd_out = first_candidate
        else:
            i = 1
            while True:
                candidate = f"{base_traj_prefix}{i}.dcd"
                if not os.path.exists(candidate):
                    dcd_out = candidate
                    break
                i += 1
    else:
        dcd_out = base_traj_prefix + ".dcd"

    pdb_out = os.path.join(output_dir, f"{base}_final.pdb")
    pdb_out_solv = os.path.join(output_dir, f"{base}_final_solv.pdb")
    checkpoint_file = os.path.join(output_dir, f"{base}.chk")
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

        water_traj.save_dcd(water_dcd_out)
        print(f"[{base}] Wrote single-frame water DCD: {water_dcd_out}")
    else:
        print(f"[{base}] No water atoms detected; skipping water DCD output.")

def run_all(system_type, device_mode, continue_mode=False, num_gpus=1):
    base_dir = {"solo": "systems/solo_sys", "combo": "systems/combo_sys"}.get(system_type)
    if base_dir is None:
        print("ERROR: choose 'solo' or 'combo'")
        sys.exit(1)

    output_dir = os.path.join("sim_output", os.path.basename(base_dir))
    os.makedirs(output_dir, exist_ok=True)

    prmtops = sorted([f for f in os.listdir(base_dir) if f.endswith(".prmtop")])

    if device_mode == "cpu":
        num_gpus = 1

    processes = []

    for i, prmt in enumerate(prmtops):
        inp = prmt.replace(".prmtop", ".inpcrd")
        prmt_path = os.path.join(base_dir, prmt)
        inp_path = os.path.join(base_dir, inp)
        gpu_id = i % num_gpus

        base = os.path.basename(prmt).replace(".prmtop", "")
        dcd_out = os.path.join(output_dir, f"{base}_traj.dcd")
        pdb_out = os.path.join(output_dir, f"{base}_final.pdb")
        checkpoint_file = os.path.join(output_dir, f"{base}.chk")

        has_pdb = os.path.exists(pdb_out)
        has_dcd = os.path.exists(dcd_out)
        has_chk = os.path.exists(checkpoint_file)

        if continue_mode:
            if not has_chk:
                print(f"[{base}] --continue specified but no checkpoint found; skipping this system.")
                continue
            if has_pdb:
                print(f"[{base}] --continue: final PDB present, continuing from checkpoint {checkpoint_file}.")
            else:
                print(f"[{base}] --continue: continuing from checkpoint {checkpoint_file} (no final PDB yet).")
        else:
            if has_pdb:
                print(f"[{base}] Final PDB found; assuming simulation is complete. Skipping.")
                continue

            if has_dcd and not has_pdb:
                print(
                    f"[{base}] DCD exists but final PDB is missing. "
                    "Deleting DCD and any checkpoint and restarting from scratch."
                )
                try:
                    os.remove(dcd_out)
                except OSError:
                    print(f"[{base}] Warning: failed to remove DCD {dcd_out}")
                if has_chk:
                    try:
                        os.remove(checkpoint_file)
                    except OSError:
                        print(f"[{base}] Warning: failed to remove checkpoint {checkpoint_file}")

        cmd = [
            sys.executable,
            "simulate_systems.py",
            prmt_path,
            inp_path,
            device_mode,
            str(gpu_id),
            output_dir,
        ]
        if continue_mode:
            cmd.append("--continue")

        p = subprocess.Popen(cmd)
        processes.append(p)

        if len(processes) >= num_gpus:
            for p in processes:
                p.wait()
            processes = []

    for p in processes:
        p.wait()


if __name__ == "__main__":
    if len(sys.argv) >= 6 and sys.argv[1].endswith(".prmtop"):
        prmtop_file = sys.argv[1]
        inpcrd_file = sys.argv[2]
        device_mode = sys.argv[3]
        gpu_id = int(sys.argv[4])
        output_dir = sys.argv[5]
        continue_mode = "--continue" in sys.argv[6:]

        simulate(
            prmtop_file=prmtop_file,
            inpcrd_file=inpcrd_file,
            device_mode=device_mode,
            gpu_id=gpu_id,
            output_dir=output_dir,
            continue_mode=continue_mode,
        )

    else:
        if len(sys.argv) < 2:
            print("Usage:")
            print("  python simulate_systems.py solo [cpu|gpu [NUM_GPUS]] [--continue]")
            print("  python simulate_systems.py combo [cpu|gpu [NUM_GPUS]] [--continue]")
            sys.exit(1)

        system_type = sys.argv[1]

        device_mode = "gpu"
        num_gpus = 1
        continue_mode = False

        args = sys.argv[2:]
        i = 0
        while i < len(args):
            arg = args[i]
            if arg == "--continue":
                continue_mode = True
                i += 1
            elif arg.lower() in ("cpu", "gpu"):
                device_mode = arg.lower()
                i += 1
                if device_mode == "gpu" and i < len(args) and args[i] != "--continue":
                    try:
                        num_gpus = int(args[i])
                        i += 1
                    except ValueError:
                        print(f"Warning: unrecognized argument '{args[i]}', ignoring.")
                        i += 1
            else:
                print(f"Warning: unrecognized argument '{arg}', ignoring.")
                i += 1

        if device_mode not in ("cpu", "gpu"):
            print("ERROR: device_mode must be 'cpu' or 'gpu'")
            sys.exit(1)

        run_all(system_type, device_mode, continue_mode, num_gpus)
