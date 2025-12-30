import mdtraj as md
import numpy as np
import os
import re
import subprocess
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
from mdtraj.core.topology import Topology

polar_amino_acids = ["SER", "THR"]
nonpolar_amino_acids = ["VAL"]
fluorinated_amino_acids = ["V6G", "V3R", "V3S", "I3G", "E3G"]


def get_chainlength(filename):
    return len(filename.replace(".prmtop", "").split('_'))


def get_p_np(filename):
    for amino_acid in polar_amino_acids:
        if amino_acid in filename:
            return 'P'
    return 'NP'


def get_makeup(filename):
    makeup = ""
    for segment in filename.replace(".prmtop", "").split('_'):
        if any(flu for flu in fluorinated_amino_acids if flu in segment):
            makeup += "F"
        else:
            makeup += "H"
    return makeup


def generate_output_basename(prmtop_filename, num_replicas, system_number):
    return f"{get_chainlength(prmtop_filename)}C_{get_p_np(prmtop_filename)}_{get_makeup(prmtop_filename)}_{num_replicas}_sys{system_number}"


def load_files(prmtop_file, inpcrd_file):
    topology = md.load_prmtop(prmtop_file)
    coords = md.load(inpcrd_file, top=topology)
    return topology, coords


def random_rotation_matrix():
    u1, u2, u3 = np.random.rand(3)
    q = np.array([
        np.sqrt(1 - u1) * np.sin(2 * np.pi * u2),
        np.sqrt(1 - u1) * np.cos(2 * np.pi * u2),
        np.sqrt(u1) * np.sin(2 * np.pi * u3),
        np.sqrt(u1) * np.cos(2 * np.pi * u3)
    ])
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]
    ])


def transform_coords(xyz, box_size):
    rot = random_rotation_matrix()
    trans = np.random.rand(3) * box_size
    return xyz.dot(rot.T) + trans


def clone_topology(top):
    new_top = Topology()
    chain_map = {}
    res_map = {}
    atom_map = {}

    for chain in top.chains:
        new_chain = new_top.add_chain()
        chain_map[chain.index] = new_chain

    for res in top.residues:
        chain = chain_map[res.chain.index]
        new_res = new_top.add_residue(
            name=res.name,
            chain=chain,
            resSeq=res.resSeq
        )
        res_map[res.index] = new_res

    for atom in top.atoms:
        new_atom = new_top.add_atom(
            name=atom.name,
            element=atom.element,
            residue=res_map[atom.residue.index]
        )
        atom_map[atom.index] = new_atom

    for bond in top.bonds:
        new_top.add_bond(
            atom_map[bond.atom1.index],
            atom_map[bond.atom2.index]
        )

    return new_top


def compute_bounding_sphere(xyz):
    center = xyz.mean(axis=0)
    radius = np.linalg.norm(xyz - center, axis=1).max()
    return center, radius


def place_without_overlap(base_xyz, placed_spheres, box_size, max_attempts=1000, min_gap=0.2):
    for attempt in range(max_attempts):
        trial_xyz = transform_coords(base_xyz, box_size)
        center, radius = compute_bounding_sphere(trial_xyz)

        ok = True
        for c_old, r_old in placed_spheres:
            dist = np.linalg.norm(center - c_old)
            if dist < (radius + r_old + min_gap):
                ok = False
                break

        if ok:
            placed_spheres.append((center, radius))
            return trial_xyz

    raise RuntimeError(
        f"Failed to place non-overlapping replica after {max_attempts} attempts. "
        f"Try increasing box_size or reducing num_replicas."
    )


def replicate_and_randomize(topology, coords, num_replicas, box_size):
    xyz_list = []
    top_list = []
    base_xyz = coords.xyz[0]
    placed_spheres = []

    for _ in range(num_replicas):
        new_xyz = place_without_overlap(base_xyz, placed_spheres, box_size)
        xyz_list.append(new_xyz)
        top_list.append(clone_topology(topology))

    big_xyz = np.concatenate([x.reshape(1, x.shape[0], 3) for x in xyz_list], axis=1)

    big_top = Topology()
    for t in top_list:
        big_top = big_top.join(t)

    traj = md.Trajectory(big_xyz, big_top)
    traj.unitcell_lengths = np.array([[box_size, box_size, box_size]])
    traj.unitcell_angles = np.array([[90.0, 90.0, 90.0]])
    return traj


def replicate_and_randomize_combo(top1, coords1, top2, coords2, num_replicas, box_size):
    xyz_list = []
    top_list = []
    half = num_replicas // 2

    base_xyz1 = coords1.xyz[0]
    base_xyz2 = coords2.xyz[0]

    placed_spheres = []

    for _ in range(half):
        xyz1 = place_without_overlap(base_xyz1, placed_spheres, box_size)
        xyz_list.append(xyz1)
        top_list.append(clone_topology(top1))

        xyz2 = place_without_overlap(base_xyz2, placed_spheres, box_size)
        xyz_list.append(xyz2)
        top_list.append(clone_topology(top2))

    big_xyz = np.concatenate([x.reshape(1, x.shape[0], 3) for x in xyz_list], axis=1)

    big_top = Topology()
    for t in top_list:
        big_top = big_top.join(t)

    traj = md.Trajectory(big_xyz, big_top)
    traj.unitcell_lengths = np.array([[box_size, box_size, box_size]])
    traj.unitcell_angles = np.array([[90.0, 90.0, 90.0]])
    return traj


def write_pdb(traj, outfile):
    os.makedirs(os.path.dirname(outfile), exist_ok=True)
    traj.save(outfile)


def write_leap_input(pdb_path, prmtop_out, inpcrd_out, ff_dir="ff_files"):
    ff_dir_abs = os.path.abspath(ff_dir)
    leap_input_path = os.path.splitext(pdb_path)[0] + "_leap.in"

    off_files = []
    frcmods = []
    if os.path.isdir(ff_dir_abs):
        for fname in os.listdir(ff_dir_abs):
            path = os.path.join(ff_dir_abs, fname)
            if fname.lower().endswith(".off"):
                off_files.append(path)
            elif fname.lower().endswith(".frcmod"):
                frcmods.append(path)

    solv_pdb_out = os.path.splitext(pdb_path)[0] + "_solvated.pdb"

    with open(leap_input_path, "w") as f:
        f.write(f"""source leaprc.water.tip3p
loadAmberParams parm10.dat
loadAmberParams frcmod.ff14SB
loadOff amino12.lib
""")
        for off in sorted(off_files):
            f.write(f'loadOff "{off}"\n')
        for frc in sorted(frcmods):
            f.write(f'loadAmberParams "{frc}"\n')

        f.write(f'mol = loadPdb "{os.path.abspath(pdb_path)}"\n')
        f.write('check mol\n')
        f.write('solvateBox mol TIP3PBOX 10.0\n')
        f.write(f'saveAmberParm mol "{os.path.abspath(prmtop_out)}" "{os.path.abspath(inpcrd_out)}"\n')
        f.write(f'savePdb mol "{os.path.abspath(solv_pdb_out)}"\n')
        f.write("quit\n")

    return leap_input_path


def run_tleap_on_pdb(pdb_path, ff_dir="ff_files"):
    base = os.path.splitext(pdb_path)[0]
    prmtop_out = base + ".prmtop"
    inpcrd_out = base + ".inpcrd"

    leap_input = write_leap_input(pdb_path, prmtop_out, inpcrd_out, ff_dir=ff_dir)

    try:
        result = subprocess.run(
            ["tleap", "-f", leap_input],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            check=False
        )
        log_path = base + "_leap.log"
        with open(log_path, "w") as logf:
            logf.write(result.stdout)

        if result.returncode != 0 or (not os.path.exists(prmtop_out)) or (not os.path.exists(inpcrd_out)):
            print(f"Error running tleap on {pdb_path}: return code {result.returncode}")
            print(f"  (see log: {log_path})")
            return None, None

        return prmtop_out, inpcrd_out

    except Exception as e:
        print(f"Exception running tleap on {pdb_path}: {e}")
        return None, None


def create_single_system(prmtop, inpcrd, num_replicas, box_size, pdb_out, ff_dir):
    top, coords = load_files(prmtop, inpcrd)
    combined = replicate_and_randomize(top, coords, num_replicas, box_size)
    write_pdb(combined, pdb_out)
    prmtop_path, inpcrd_path = run_tleap_on_pdb(pdb_out, ff_dir=ff_dir)
    return pdb_out, prmtop_path, inpcrd_path


def create_combo_system(pr1, ic1, pr2, ic2, num_replicas, box_size, pdb_out, ff_dir):
    top1, coords1 = load_files(pr1, ic1)
    top2, coords2 = load_files(pr2, ic2)
    combined = replicate_and_randomize_combo(top1, coords1, top2, coords2, num_replicas, box_size)
    write_pdb(combined, pdb_out)
    prmtop_path, inpcrd_path = run_tleap_on_pdb(pdb_out, ff_dir=ff_dir)
    return pdb_out, prmtop_path, inpcrd_path


def find_max_sys_index(output_dir, prmtop_filename, num_replicas):
    """
    Scan output_dir for files matching the pattern:
      {chain}C_{P/NP}_{makeup}_{num_replicas}_sysN.pdb
    and return the maximum N found (0 if none).
    """
    if not os.path.isdir(output_dir):
        return 0

    base_prefix = f"{get_chainlength(prmtop_filename)}C_{get_p_np(prmtop_filename)}_{get_makeup(prmtop_filename)}_{num_replicas}_sys"
    pattern = re.compile(re.escape(base_prefix) + r"(\d+)\.pdb$")

    max_idx = 0
    for fname in os.listdir(output_dir):
        m = pattern.match(fname)
        if m:
            idx = int(m.group(1))
            if idx > max_idx:
                max_idx = idx
    return max_idx


def create_systems(
    input_dir,
    num_systems,
    num_replicas,
    box_size,
    solo_dir,
    combo_dir,
    ff_dir="ff_files",
    max_workers=None
):
    os.makedirs(solo_dir, exist_ok=True)
    os.makedirs(combo_dir, exist_ok=True)

    files = os.listdir(input_dir)
    prmtops = [f for f in files if f.endswith(".prmtop")]

    fluorinated = [f for f in prmtops if any(fluo in f for fluo in fluorinated_amino_acids)]

    jobs = []

    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        for prmt in prmtops:
            inp = prmt.replace(".prmtop", ".inpcrd")
            if inp not in files:
                continue

            start_sys = find_max_sys_index(solo_dir, prmt, num_replicas)

            for i in range(start_sys + 1, start_sys + num_systems + 1):
                base = generate_output_basename(prmt, num_replicas, i)
                pdb_out = os.path.join(solo_dir, base + ".pdb")

                jobs.append(
                    ex.submit(
                        create_single_system,
                        os.path.join(input_dir, prmt),
                        os.path.join(input_dir, inp),
                        num_replicas,
                        box_size,
                        pdb_out,
                        ff_dir,
                    )
                )

        for flu in fluorinated:  # achoo
            for aa in nonpolar_amino_acids:
                base_name, ext = os.path.splitext(flu)
                nf_base = base_name
                for fluo in fluorinated_amino_acids:
                    nf_base = nf_base.replace(fluo, aa)
                nf = nf_base + ext

                if nf not in prmtops:
                    continue

                start_sys = find_max_sys_index(combo_dir, flu, num_replicas)

                for i in range(start_sys + 1, start_sys + num_systems + 1):
                    base = generate_output_basename(flu, num_replicas, i)
                    pdb_out = os.path.join(combo_dir, base + ".pdb")

                    jobs.append(
                        ex.submit(
                            create_combo_system,
                            os.path.join(input_dir, flu),
                            os.path.join(input_dir, flu.replace(".prmtop", ".inpcrd")),
                            os.path.join(input_dir, nf),
                            os.path.join(input_dir, nf.replace(".prmtop", ".inpcrd")),
                            num_replicas,
                            box_size,
                            pdb_out,
                            ff_dir,
                        )
                    )

        for task in tqdm(jobs):
            try:
                pdb_path, prmtop_path, inpcrd_path = task.result()
                if prmtop_path is None or inpcrd_path is None:
                    print(f"Created PDB but failed tleap: {pdb_path}")
                else:
                    print(f"Created: {prmtop_path} and {inpcrd_path}")
            except Exception as e:
                print("Error:", e)


if __name__ == "__main__":
    input_dir = "input_files"
    num_systems = 5
    num_replicas = 25
    box_size = 3
    output_dir = "systems"
    solo_dir = os.path.join(output_dir, "solo_sys")
    combo_dir = os.path.join(output_dir, "combo_sys")
    ff_dir = "ff_files"

    create_systems(input_dir, num_systems, num_replicas, box_size, solo_dir, combo_dir, ff_dir=ff_dir)