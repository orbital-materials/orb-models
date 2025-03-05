import numpy as np
import torch
from collections import deque
from ase.io import read, write
from orb_models.forcefield import pretrained
from orb_models.forcefield.calculator import ORBCalculator
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.md.langevin import Langevin
from ase import units
from ase.md import MDLogger
import os

def setup_device() -> torch.device:
    """Set up and return the appropriate compute device."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")
    return device

class BondingLogger:
    """Logger for bond changes and reactions during MD simulation."""
    def __init__(self, dynamics, atoms, filename):
        self.dyn = dynamics
        self.atoms = atoms
        self.filename = filename
        # Write initial file and bonding info
        with open(self.filename, 'w') as f:
            f.write("# Bonding Log\n")
            self._print_initial_info(f)
    
    def write(self):
        """Write current bonding information (required by ASE's dynamics system)."""
        # This method will be called periodically by the dynamics
        # For now, we don't need to write anything here since reactions
        # are logged separately via log_reaction
        pass
    
    def log_reaction(self, reaction_str: str):
        """Log a detected reaction."""
        current_time = self.dyn.get_number_of_steps() * self.dyn.dt / units.fs
        with open(self.filename, 'a') as f:
            f.write(f"Reaction at {current_time:.1f} fs: {reaction_str}\n")
    
    def _print_initial_info(self, f):
        """Print initial bonding information."""
        # Trigger energy calculation to update bonding results
        self.atoms.get_potential_energy()
        calc = self.atoms.calc

        f.write("\nBond definitions:\n")
        vdw_multiplier = getattr(calc, "vdw_multiplier", None)
        if vdw_multiplier is not None:
            f.write(" - A bond is defined if the actual distance between two atoms is lower than:\n")
            f.write("   cutoff = vdw_multiplier * (vdW radius of atom1 + vdW radius of atom2)\n")
            f.write(f"   (with vdw_multiplier = {vdw_multiplier})\n")
        else:
            f.write(" - Bond cutoff threshold not available from calculator.\n")
        f.write(" - Note: H-H bonds are never considered bonded.\n\n")

        # Get bonding information from calculator
        bonding_graph = calc.results["bonding_graph"]
        pair_senders = calc.results.get("pair_senders")
        pair_receivers = calc.results.get("pair_receivers")
        pair_bond_lengths = calc.results.get("pair_bond_lengths")
        pair_vdw_cutoffs = calc.results.get("pair_vdw_cutoffs")

        # Count bonds and store details
        bond_counts = {}
        bonds_details = []
        cutoff_by_type = {}  # Store cutoff for each bond type
        
        if pair_senders is not None:
            for s, r, d, cutoff in zip(pair_senders, pair_receivers, pair_bond_lengths, pair_vdw_cutoffs):
                if s < r:  # Only process each pair once
                    symbol1 = self.atoms[s].symbol
                    symbol2 = self.atoms[r].symbol
                    bond_type = tuple(sorted([symbol1, symbol2]))
                    cutoff_by_type[bond_type] = cutoff
                    if d < cutoff:
                        bonds_details.append((s, symbol1, r, symbol2, d, cutoff))
                        bond_counts[bond_type] = bond_counts.get(bond_type, 0) + 1

        # Write bond summary
        f.write("Bond summary:\n")
        species = sorted(set(atom.symbol for atom in self.atoms))
        for i, species1 in enumerate(species):
            for species2 in species[i:]:
                bond_type = tuple(sorted([species1, species2]))
                count = bond_counts.get(bond_type, 0)
                cutoff = cutoff_by_type.get(bond_type, None)
                if cutoff is not None:
                    f.write(f" - {'-'.join(bond_type)} bonds: {count}, cutoff = {cutoff:.3f} Å\n")
                else:
                    f.write(f" - {'-'.join(bond_type)} bonds: {count}\n")

        # Write detailed bond information
        f.write("\nDetailed bond values (only unique bonds are listed):\n")
        if bonds_details:
            for s, sym1, r, sym2, distance, cutoff in bonds_details:
                f.write(f"Atom {s} ({sym1}) -- Atom {r} ({sym2}): distance = {distance:.3f} Å, cutoff = {cutoff:.3f} Å\n")
        else:
            f.write(" No bonds detected.\n")

class ReactionTrajectoryManager:
    def __init__(self, time_window: float, timestep: float, reaction_traj_interval: int, 
                 frame_interval_fs: float, bonding_logger: BondingLogger):
        """
        Reaction trajectory manager that captures frames around reaction events.
        
        Args:
            time_window: Total time window to capture around reaction events (in fs)
            timestep: MD timestep (in internal ASE units)
            reaction_traj_interval: MD steps between saved frames
            frame_interval_fs: Time between captured frames (in fs)
            bonding_logger: Logger instance for recording reaction events
        """
        self.frame_interval = frame_interval_fs
        self.timestep = timestep
        self.reaction_traj_interval = reaction_traj_interval
        self.bonding_logger = bonding_logger

        # Calculate number of frames based on time window
        frames_per_side = round(time_window / (2 * self.frame_interval))
        self.reaction_capture_size = 2 * frames_per_side + 1
        self.reaction_pre_frames = frames_per_side

        # Initialize buffers and counters
        self.frame_counter = 0
        self.pre_reaction_buffer = deque(maxlen=self.reaction_pre_frames)
        self.pending_reactions = []
        self.last_reaction_frame = -float('inf')
    
    def add_frame(self, atoms, step: int):
        """Updates the pre-reaction buffer and pending reaction captures."""
        current_frame = self.frame_counter
        abs_time = current_frame * self.frame_interval * (self.timestep/units.fs)
        self.frame_counter += 1

        frame_data = (atoms.copy(), abs_time)
        self.pre_reaction_buffer.append(frame_data)

        # Update each pending reaction event
        for event in self.pending_reactions[:]:
            event["capture_buffer"].append(frame_data)
            if len(event["capture_buffer"]) >= event["required_total"]:
                self._write_reaction_trajectory(event)
                self.last_reaction_frame = event["reaction_frame"]
                self.pending_reactions.remove(event)
    
    def on_reaction_detected(self, step: int, reaction_str: str):
        """Called when a reaction is detected."""
        capture_buffer = list(self.pre_reaction_buffer)
        # Calculate time based on actual MD step with proper fs conversion
        reaction_time = step * self.timestep / units.fs
        
        event = {
            "reaction_frame": self.frame_counter,
            "reaction_time": reaction_time,  # This is the time of the actual reaction
            "reaction_str": reaction_str,
            "capture_buffer": capture_buffer,
            "required_total": self.reaction_capture_size
        }
        self.pending_reactions.append(event)
        
        # Log the reaction
        self.bonding_logger.log_reaction(reaction_str)
        print(f"Reaction detected at MD step {step} (time {reaction_time:.1f} fs): {reaction_str}")
        print(f"Queued reaction event. Waiting until capture buffer reaches {self.reaction_capture_size} frames.")
    
    def _write_reaction_trajectory(self, event):
        """Writes the captured reaction trajectory to file."""
        # Replace long phrases with abbreviations
        short_reaction_str = event["reaction_str"].replace("Bond Formed", "BF").replace("Bond Broken", "BB")
        safe_reaction_str = short_reaction_str.replace(" ", "_").replace("(", "").replace(")", "").replace(",", "_")
        
        output_file = os.path.join("output", f"reaction_traj_{event['reaction_time']:.1f}fs_{safe_reaction_str}.xyz")
        MAX_FILENAME_LEN = 150
        
        if len(output_file) > MAX_FILENAME_LEN:
            output_file = os.path.join("output", f"reaction_traj_{event['reaction_time']:.1f}fs.xyz")
            print(f"Warning: Reaction file name too long, using shortened name: {output_file}")
        
        print(f"Writing reaction trajectory for reaction at frame {event['reaction_frame']} "
              f"(reaction time {event['reaction_time']:.1f} fs) to {output_file}")
        
        with open(output_file, 'w') as f:
            for atoms, abs_time in event["capture_buffer"]:
                relative_time = abs_time - event["reaction_time"]
                cell_info = f"Cell: {atoms.get_cell()}"
                comment = (f"Time: {abs_time:.1f} fs, "
                          f"Time relative to reaction: {relative_time:.1f} fs, "
                          f"{event['reaction_str']}, {cell_info}")
                write(f, atoms, format='xyz', comment=comment)

def get_bonding_graph(atoms: "atoms") -> np.ndarray:
    """Get the bonding graph from the atoms object."""
    try:
        if "bonding_graph" not in atoms.calc.results:
            atoms.get_potential_energy()
        return atoms.calc.results["bonding_graph"].copy()
    except Exception as e:
        raise RuntimeError(f"Failed to compute bonding graph: {e}")

def update_bonds(atoms, ema_alpha: float, ema_bonding_graph: np.ndarray, 
                baseline_bonding_graph: np.ndarray, traj_manager, 
                dyn, reaction_threshold: float) -> tuple[np.ndarray, np.ndarray]:
    """Update the EMA bonding graph and check for reactions."""
    current_graph = get_bonding_graph(atoms)
    if ema_bonding_graph is None or baseline_bonding_graph is None:
        ema_bonding_graph = current_graph.astype(float)
        baseline_bonding_graph = current_graph.copy()
        return ema_bonding_graph, baseline_bonding_graph

    ema_bonding_graph = ema_alpha * current_graph + (1 - ema_alpha) * ema_bonding_graph

    # Get upper triangle indices just once
    i_upper, j_upper = np.triu_indices(len(atoms), k=1)
    diff = np.abs(ema_bonding_graph[i_upper, j_upper] - baseline_bonding_graph[i_upper, j_upper])
    
    # Only do the expensive processing if there's actually a reaction
    if np.any(diff >= reaction_threshold):
        reaction_mask = diff >= reaction_threshold
        i_react = i_upper[reaction_mask]
        j_react = j_upper[reaction_mask]
        
        is_forming = ema_bonding_graph[i_react, j_react] > baseline_bonding_graph[i_react, j_react]
        
        new_baseline = baseline_bonding_graph.copy()
        new_baseline[i_react, j_react] = is_forming.astype(int)
        new_baseline[j_react, i_react] = is_forming.astype(int)
        
        reaction_details = [
            f"{atoms[i].symbol}{i}-{atoms[j].symbol}{j} ({'Bond Formed' if forming else 'Bond Broken'})"
            for i, j, forming in zip(i_react, j_react, is_forming)
        ]
        
        reaction_str = ', '.join(reaction_details)
        current_step = dyn.get_number_of_steps()
        traj_manager.on_reaction_detected(current_step, reaction_str)
        baseline_bonding_graph = new_baseline

    return ema_bonding_graph, baseline_bonding_graph

def print_bonding_info(atoms, file=None):
    """Print bonding information including actual bond distances and cutoffs from the current calculation."""
    # Helper function to write only to file if provided
    def write_output(message):
        if file:
            print(message, file=file)

    # Trigger energy calculation to update bonding results.
    atoms.get_potential_energy()
    calc = atoms.calc

    write_output("Bond definitions:")
    vdw_multiplier = getattr(calc, "vdw_multiplier", None)
    if vdw_multiplier is not None:
        write_output(" - A bond is defined if the actual distance between two atoms is lower than:")
        write_output("   cutoff = vdw_multiplier * (vdW radius of atom1 + vdW radius of atom2)")
        write_output(f"   (with vdw_multiplier = {vdw_multiplier})")
    else:
        write_output(" - Bond cutoff threshold not available from calculator.")
    write_output(" - Note: H-H bonds are never considered bonded.\n")

    bonding_graph = calc.results["bonding_graph"]
    pair_senders = calc.results.get("pair_senders")
    pair_receivers = calc.results.get("pair_receivers")
    pair_bond_lengths = calc.results.get("pair_bond_lengths")
    pair_vdw_cutoffs = calc.results.get("pair_vdw_cutoffs")

    bond_counts = {}
    bonds_details = []
    cutoff_by_type = {}  # Store cutoff for each bond type
    
    if pair_senders is not None:
        for s, r, d, cutoff in zip(pair_senders, pair_receivers, pair_bond_lengths, pair_vdw_cutoffs):
            if s < r:  # Only process each pair once
                symbol1 = atoms[s].symbol
                symbol2 = atoms[r].symbol
                bond_type = tuple(sorted([symbol1, symbol2]))
                cutoff_by_type[bond_type] = cutoff  # Store cutoff for this bond type
                if d < cutoff:
                    bonds_details.append((s, symbol1, r, symbol2, d, cutoff))
                    bond_counts[bond_type] = bond_counts.get(bond_type, 0) + 1

    write_output("Bond summary:")
    # Get all unique species
    species = sorted(set(atom.symbol for atom in atoms))
    # Print all possible combinations
    for i, species1 in enumerate(species):
        for species2 in species[i:]:
            bond_type = tuple(sorted([species1, species2]))
            count = bond_counts.get(bond_type, 0)
            cutoff = cutoff_by_type.get(bond_type, None)
            if cutoff is not None:
                write_output(f" - {'-'.join(bond_type)} bonds: {count}, cutoff = {cutoff:.3f} Å")
            else:
                write_output(f" - {'-'.join(bond_type)} bonds: {count}")

    write_output("\nDetailed bond values (only unique bonds are listed):")
    if bonds_details:
        for s, sym1, r, sym2, distance, cutoff in bonds_details:
            write_output(f"Atom {s} ({sym1}) -- Atom {r} ({sym2}): distance = {distance:.3f} Å, cutoff = {cutoff:.3f} Å")
    else:
        write_output(" No bonds detected.")

def run_md_simulation_with_reaction_capture(
    input_file: str = "HClWater.xyz",
    output_dir: str = "output",
    output_file: str = None,
    cell_size: float = 25.25,
    temperature_K: float = 300,
    timestep: float = 0.5 * units.fs,          # 0.5 fs timestep
    friction: float = 0.01 / units.fs,
    log_interval_fs: float = 10.0,              # Time between log entries (fs)
    total_time: float = 1000.0,                # Total simulation time in fs
    md_traj_interval_fs: float = 500.0,        # Time between main trajectory frames (fs)
    bond_check_interval_fs: float = 10.0,       # Time between bond checks (fs)
    ema_alpha: float = 0.1,                     # Smoothing factor for the EMA of the bonding graph
    reaction_threshold: float = 0.6,            # Threshold for detecting reactions
    vdw_multiplier: float = 0.5,                # Multiplier for van der Waals radii in bonding calculations
    reaction_time_window: float = None,         # Total time window to capture around reaction events (fs)
    reaction_traj_interval_fs: float = None,    # Time between captured frames (fs)
):
    """Run MD simulation with reaction event capture using an EMA-based bonding graph scheme."""
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Calculate default values if not provided
    if reaction_time_window is None:
        reaction_time_window = bond_check_interval_fs / ema_alpha * 4.0
    if reaction_traj_interval_fs is None:
        reaction_traj_interval_fs = reaction_time_window / 100.0
    
    # Convert time intervals to steps
    md_traj_interval_steps = int(md_traj_interval_fs / (timestep/units.fs))
    reaction_traj_interval_steps = int(reaction_traj_interval_fs / (timestep/units.fs))
    bond_check_interval_steps = int(bond_check_interval_fs / (timestep/units.fs))
    log_interval_steps = int(log_interval_fs / (timestep/units.fs))
    total_steps = int(total_time / (timestep/units.fs))
    
    # Set output_file based on input_file if not provided
    if output_file is None:
        base, ext = os.path.splitext(os.path.basename(input_file))
        output_file = f"{base}_md{ext}"
    
    # Build full output file paths
    output_file_path = os.path.join(output_dir, output_file)
    md_log_path = os.path.join(output_dir, "md.log")
    bonding_log_path = os.path.join(output_dir, "bonding.log")
    
    # Setup computation device
    device = setup_device()
    
    # Read and setup atomic system
    print(f"Reading structure from {input_file}")
    try:
        atoms = read(input_file)
    except Exception as e:
        raise RuntimeError(f"Failed to read input file {input_file}: {e}")
    
    atoms.set_cell([cell_size] * 3)
    atoms.set_pbc([True] * 3)
    
    # Setup calculator
    calc = ORBCalculator(
        model=pretrained.orb_d3_v2(),
        device=device,
        return_bonding_graph=True,
        vdw_multiplier=vdw_multiplier
    )
    atoms.calc = calc
    
    # Setup dynamics
    MaxwellBoltzmannDistribution(atoms, temperature_K=temperature_K)
    dyn = Langevin(atoms, timestep, temperature_K=temperature_K, friction=friction)
    
    # Initialize loggers
    bonding_logger = BondingLogger(dyn, atoms, bonding_log_path)
    
    # Initialize trajectory manager
    traj_manager = ReactionTrajectoryManager(
        time_window=reaction_time_window,
        timestep=timestep,
        reaction_traj_interval=reaction_traj_interval_steps,
        frame_interval_fs=reaction_traj_interval_fs,
        bonding_logger=bonding_logger
    )
    
    # Initialize bond tracking variables
    ema_bonding_graph = None
    baseline_bonding_graph = None
    
    def update_bonds_wrapper():
        nonlocal ema_bonding_graph, baseline_bonding_graph
        ema_bonding_graph, baseline_bonding_graph = update_bonds(
            atoms, ema_alpha, ema_bonding_graph, baseline_bonding_graph, 
            traj_manager, dyn, reaction_threshold
        )
    
    # Attach observers
    dyn.attach(
        lambda: write(output_file_path, atoms, append=True, format='xyz',
                 comment=f"Time: {dyn.get_number_of_steps() * (timestep/units.fs):.1f} fs, Cell: {atoms.get_cell()}"),
        interval=md_traj_interval_steps
    )
    dyn.attach(lambda: traj_manager.add_frame(atoms, dyn.get_number_of_steps()), 
              interval=reaction_traj_interval_steps)
    dyn.attach(update_bonds_wrapper, interval=bond_check_interval_steps)
    dyn.attach(bonding_logger, interval=log_interval_steps)
    dyn.attach(MDLogger(dyn, atoms, md_log_path), interval=log_interval_steps)
    
    # Run simulation
    print("\nStarting MD simulation...")
    try:
        dyn.run(steps=total_steps)
    except Exception as e:
        print(f"\nMD simulation failed: {e}")
        raise
    finally:
        print("\nMD simulation completed or interrupted!")

if __name__ == "__main__":
    run_md_simulation_with_reaction_capture() 
