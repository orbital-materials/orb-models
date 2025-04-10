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

class ReactionTrajectoryManager:
    def __init__(self, time_window: float, timestep: float, reaction_traj_interval: int, frame_interval_fs: float):
        """
        Reaction trajectory manager that captures frames around reaction events.
        
        Args:
            time_window: Total time window to capture around reaction events (in fs)
            timestep: MD timestep (in internal ASE units)
            reaction_traj_interval: MD steps between saved frames
            frame_interval_fs: Time between captured frames (in fs)
        """
        self.frame_interval = frame_interval_fs
        self.timestep = timestep
        self.reaction_traj_interval = reaction_traj_interval

        # Calculate number of frames based on time window
        frames_per_side = round(time_window / (2 * self.frame_interval))
        self.reaction_capture_size = 2 * frames_per_side + 1
        self.reaction_pre_frames = frames_per_side
        
        # Initialize buffers and counters
        self.frame_counter = 0
        self.pre_reaction_buffer = deque(maxlen=self.reaction_pre_frames)
        self.pending_reactions = []
        
        print("ReactionTrajectoryManager initialized with:")
        print(f"  Time window: {time_window} fs")
        print(f"  MD timestep: {timestep/units.fs} fs")
        print(f"  Reaction trajectory interval: {reaction_traj_interval} steps")
        print(f"  Frame interval for reaction capture: {self.frame_interval} fs")
        print(f"  Total frames to capture: {self.reaction_capture_size}")
        print(f"  Pre-reaction frames: {self.reaction_pre_frames}")
        print(f"  Frame range: -{self.reaction_pre_frames} to +{self.reaction_pre_frames}")

    def add_frame(self, atoms, step: int):
        """
        Called on every MD step (or at the same frequency as the MD observer).
        Updates the pre-reaction buffer and appends frames to pending reaction events.
        """
        current_frame = self.frame_counter
        abs_time = current_frame * self.frame_interval
        self.frame_counter += 1

        # Create a copy of the current state and store with its absolute time
        frame_data = (atoms.copy(), abs_time)
        # Update the pre-reaction buffer (used for new reaction events)
        self.pre_reaction_buffer.append(frame_data)
        print(f"Added frame {current_frame} (time {abs_time:.1f} fs) to pre-reaction buffer (size: {len(self.pre_reaction_buffer)})")

        # Update each pending reaction event
        for event in self.pending_reactions[:]:
            event["capture_buffer"].append(frame_data)
            if len(event["capture_buffer"]) >= event["required_total"]:
                self._write_reaction_trajectory(event)
                self.pending_reactions.remove(event)
    
    def on_reaction_detected(self, step: int, reaction_str: str, bonding_file=None):
        """
        Called when a reaction is detected.
        Copies the contents of the pre-reaction buffer into a dedicated capture buffer for this event,
        along with recording the reaction frame and time.
        """
        # Initialize the capture buffer for the reaction event with the current pre-reaction buffer.
        capture_buffer = list(self.pre_reaction_buffer)
        # The reaction time is taken as the current frame time
        reaction_time = self.frame_counter * self.frame_interval

        event = {
            "reaction_frame": self.frame_counter,  # current frame is the reaction frame
            "reaction_time": reaction_time,
            "reaction_str": reaction_str,
            "bonding_file": bonding_file,
            "capture_buffer": capture_buffer,
            "required_total": self.reaction_capture_size
        }
        self.pending_reactions.append(event)
        print(f"Reaction detected at frame {event['reaction_frame']} (time {reaction_time:.1f} fs): {reaction_str}")
        print(f"Queued reaction event. Waiting until capture buffer reaches {self.reaction_capture_size} frames.")
    
    def _write_reaction_trajectory(self, event):
        """
        Writes the captured reaction trajectory to file.
        Each frame is written with its absolute time and time relative to the reaction.
        """
        safe_reaction_str = event["reaction_str"].replace(" ", "_").replace("(", "").replace(")", "").replace(",", "_")
        output_file = os.path.join("output", f"reaction_traj_{event['reaction_time']:.1f}fs_{safe_reaction_str}.xyz")
        print(f"Writing reaction trajectory for reaction at frame {event['reaction_frame']} (reaction time {event['reaction_time']:.1f} fs) to {output_file}")
        
        with open(output_file, 'w') as f:
            for atoms, abs_time in event["capture_buffer"]:
                relative_time = abs_time - event["reaction_time"]
                cell_info = f"Cell: {atoms.get_cell()}"
                comment = (f"Time: {abs_time:.1f} fs, "
                           f"Time relative to reaction: {relative_time:.1f} fs, "
                           f"{event['reaction_str']}, {cell_info}")
                write(f, atoms, format='xyz', comment=comment)
        
        if event["bonding_file"]:
            print(f"\nReaction at time {event['reaction_time']:.1f} fs: {event['reaction_str']}\nTrajectory written to {output_file}",
                  file=event["bonding_file"])

def get_bonding_graph(atoms: "atoms") -> np.ndarray:
    """Get the bonding graph (binary) from the atoms object."""
    atoms.get_potential_energy()  # Triggers calculation
    return atoms.calc.results["bonding_graph"].copy()

def check_bonds(atoms, dyn, avg_bonding_graph, traj_manager, bonding_file=None):
    """Update the averaged bonding graph and check for reactions."""
    current_graph = get_bonding_graph(atoms)
    old_avg = avg_bonding_graph.copy()
    
    # Update and clip the bonding graph
    avg_bonding_graph = np.clip(avg_bonding_graph + 0.1 * (2 * current_graph - 1), 0, 1)

    # Check only upper triangle for transitions
    i_upper, j_upper = np.triu_indices(len(atoms), k=1)
    transitions = np.isclose(avg_bonding_graph[i_upper, j_upper], 0.5) & \
                 (~np.isclose(old_avg[i_upper, j_upper], 0.5))
    
    # Process transitions
    current_step = dyn.get_number_of_steps()
    if np.any(transitions):
        # Collect all transitions for this reaction event
        reaction_details = []
        for idx, (i, j) in enumerate(zip(i_upper[transitions], j_upper[transitions])):
            event_type = "Bond Formed" if old_avg[i, j] < 0.5 else "Bond Broken"
            reaction_details.append(f"{atoms[i].symbol}{i}-{atoms[j].symbol}{j} ({event_type})")
        
        reaction_str = ', '.join(reaction_details)
        traj_manager.on_reaction_detected(current_step, reaction_str, bonding_file)
    
    return avg_bonding_graph

def print_bonding_info(atoms, file=None):
    """Print bonding information including actual bond distances and cutoffs."""
    def write_output(message):
        if file:
            print(message, file=file)

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
    cutoff_by_type = {}
    
    if pair_senders is not None:
        for s, r, d, cutoff in zip(pair_senders, pair_receivers, pair_bond_lengths, pair_vdw_cutoffs):
            if s < r:
                symbol1 = atoms[s].symbol
                symbol2 = atoms[r].symbol
                bond_type = tuple(sorted([symbol1, symbol2]))
                cutoff_by_type[bond_type] = cutoff
                if d < cutoff:
                    bonds_details.append((s, symbol1, r, symbol2, d, cutoff))
                    bond_counts[bond_type] = bond_counts.get(bond_type, 0) + 1

    write_output("Bond summary:")
    species = sorted(set(atom.symbol for atom in atoms))
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
    input_file: str = "SN2-noH.xyz",
    output_dir: str = "output",
    output_file: str = "SN2-noHMD.xyz",
    bonding_info_file: str = "bondinginfo.txt",
    cell_size: float = 29.42,
    temperature_K: float = 300,
    timestep: float = 0.5 * units.fs,          # 0.5 fs timestep
    friction: float = 0.01 / units.fs,
    total_steps: int = 1000000,                # Increased to 1 million steps
    reaction_traj_interval_fs: float = 1.0,     # Time between reaction capture frames (fs)
    md_traj_interval_fs: float = 1000.0,       # Time between main trajectory frames (fs)
    log_interval_fs: float = 10.0,             # Time between log entries (fs)
    bond_check_interval_fs: float = 100.0,     # Time between bond checks (fs)
    reaction_time_window: float = 1000.0,      # Total capture window in fs around reaction
):
    """Run MD simulation with reaction event capture and separate MD trajectory output spacing.
    All time intervals are specified in femtoseconds and converted to MD steps internally."""
    
    # Convert time intervals to number of MD steps
    timestep_fs = timestep / units.fs
    reaction_traj_steps = max(1, round(reaction_traj_interval_fs / timestep_fs))
    md_traj_steps = max(1, round(md_traj_interval_fs / timestep_fs))
    log_steps = max(1, round(log_interval_fs / timestep_fs))
    bond_check_steps = max(1, round(bond_check_interval_fs / timestep_fs))
    
    device = setup_device()
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Update file paths to use the output directory
    output_file_path = os.path.join(output_dir, output_file)
    bonding_info_file_path = os.path.join(output_dir, bonding_info_file)
    md_log_path = os.path.join(output_dir, "md.log")
    
    try:
        with open(bonding_info_file_path, 'w') as bonding_file:
            print(f"Reading structure from {input_file}")
            try:
                atoms = read(input_file)
            except Exception as e:
                raise RuntimeError(f"Failed to read input file {input_file}: {e}")
            
            atoms.set_cell([cell_size] * 3)
            atoms.set_pbc([True] * 3)
            
            calc = ORBCalculator(
                model=pretrained.orb_d3_v2(),
                device=device,
                return_bonding_graph=True,
                vdw_multiplier=0.58  # Changed to 0.55 for slightly larger bond cutoffs
            )
            atoms.calc = calc
            
            MaxwellBoltzmannDistribution(atoms, temperature_K=temperature_K)
            dyn = Langevin(atoms, timestep, temperature_K=temperature_K, friction=friction)
            
            print_bonding_info(atoms, bonding_file)
            
            # Initialize reaction capture manager
            traj_manager = ReactionTrajectoryManager(
                time_window=reaction_time_window,
                timestep=timestep,
                reaction_traj_interval=reaction_traj_steps,
                frame_interval_fs=reaction_traj_interval_fs
            )
            
            avg_bonding_graph = get_bonding_graph(atoms).astype(float)
            
            def update_bonds():
                nonlocal avg_bonding_graph
                avg_bonding_graph = check_bonds(atoms, dyn, avg_bonding_graph, traj_manager, bonding_file)
            
            # Attach observers
            # Write the main MD trajectory at specified intervals
            dyn.attach(
                lambda: write(output_file_path, atoms, append=True, format='xyz',
                         comment=f"Time: {dyn.get_number_of_steps() * (timestep/units.fs):.1f} fs, Cell: {atoms.get_cell()}"),
                interval=md_traj_steps
            )
            # Pass frames to reaction capture manager
            dyn.attach(lambda: traj_manager.add_frame(atoms, dyn.get_number_of_steps()), 
                      interval=reaction_traj_steps)
            dyn.attach(update_bonds, interval=bond_check_steps)
            dyn.attach(MDLogger(dyn, atoms, md_log_path), interval=log_steps)
            
            print("\nStarting MD simulation...")
            
            try:
                dyn.run(steps=total_steps)
            except Exception as e:
                print(f"\nMD simulation failed: {e}")
                raise
            finally:
                print("\nMD simulation completed or interrupted!")
                
    except IOError as e:
        print(f"Error handling bonding info file {bonding_info_file_path}: {e}")
        raise

if __name__ == "__main__":
    run_md_simulation_with_reaction_capture() 
