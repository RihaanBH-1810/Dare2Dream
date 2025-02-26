import json
import os
import numpy as np
import pandas as pd
from enum import Enum
import multiprocessing as mp
from functools import partial
from tqdm import tqdm
from motorlib.motor import Motor
from motorlib.grains.bates import BatesGrain
# from motorlib.grains.finocyl import FinocylGrain
# from motorlib.grains.rodTube import RodTubeGrain
from motorlib.propellant import Propellant

class FailureMode(Enum):
    # Grain Failures
    GRAIN_CRACKING = "Grain Cracking"
    GRAIN_BLOWOUT = "Grain Blowout" 
    DELAMINATION = "Delamination of Grain Layers"
    BURN_RATE_IRREGULARITY = "Grain Burn Rate Irregularities"
    PARTICLE_CRACKING = "Particle Cracking"
    INCOMPLETE_COMBUSTION = "Incomplete Combustion"
    GRAIN_DEFORMATION = "Grain Shrinkage/Swelling"
    
    # Nozzle Failures
    NOZZLE_EROSION = "Nozzle Erosion"
    NOZZLE_THROAT_CLOGGING = "Nozzle Throat Choking"
    NOZZLE_DEFORMATION = "Nozzle Deformation"
    
    # Combustion Instabilities
    PRESSURE_OSCILLATION = "Pressure Oscillations"
    COMBUSTION_INSTABILITY = "Combustion Instability"
    HOT_SPOTS = "Hot Spots"
    
    # Ignition Failures
    PREMATURE_IGNITION = "Premature Ignition"
    DELAYED_IGNITION = "Delayed Ignition"
    
    # Thermal Failures
    OVERHEATING = "Overheating"
    THERMAL_SHOCK = "Thermal Shock"
    THERMAL_MISMATCH = "Thermal Mismatch"

def run_simulation(motor_path):
    """Run simulation for a single motor configuration"""
    try:
        # Get failure mode from path
        parent_dir = os.path.basename(os.path.dirname(motor_path))
        
        # Get the root output directory (generated_data)
        root_dir = os.path.dirname(os.path.dirname(os.path.dirname(motor_path)))
        
        # Construct csv path correctly
        if parent_dir != "normal":
            csv_dir = os.path.join(root_dir, "csv", parent_dir)
        else:
            csv_dir = os.path.join(root_dir, "csv", "normal")
            
        # Create output directory if it doesn't exist
        os.makedirs(csv_dir, exist_ok=True)
        
        # Set output path
        base_name = os.path.basename(motor_path)
        output_csv = os.path.join(csv_dir, base_name.replace('.ric', '.csv'))

        # Load and simulate the motor directly
        with open(motor_path, 'r') as f:
            motor_dict = json.load(f)
            
        motor = Motor(motor_dict)
        sim_result = motor.runSimulation()
        
        if sim_result.success:
            # Save CSV output
            with open(output_csv, 'w') as f:
                f.write(sim_result.getCSV())
            print(f"CSV file saved to: {output_csv}")
            return motor_path, True
        else:
            print(f"Simulation failed for motor: {motor_path}")
            for alert in sim_result.getAlerts():
                print(f"Alert: {alert.description}")
            return motor_path, False
            
    except Exception as e:
        print(f"Failed to simulate motor: {motor_path}")
        print(f"Error: {str(e)}")
        return motor_path, False

def generate_single_config(args):
    """Worker function for parallel processing"""
    id_num, failure_mode, output_dir = args
    
    try:
        motor = Motor()
        
        # Generate propellant
        propellant = Propellant()
        base_a = np.random.uniform(1e-5, 5e-5)
        base_n = np.random.uniform(0.3, 0.4)
        base_k = np.random.uniform(1.2, 1.3)
        base_t = np.random.uniform(2500, 3300)
        
        if failure_mode and failure_mode in [FailureMode.BURN_RATE_IRREGULARITY, 
                                           FailureMode.INCOMPLETE_COMBUSTION,
                                           FailureMode.HOT_SPOTS,
                                           FailureMode.COMBUSTION_INSTABILITY]:
            if failure_mode == FailureMode.BURN_RATE_IRREGULARITY:
                base_a *= np.random.uniform(1.5, 3.0)
                base_n *= np.random.uniform(1.5, 2.0)
            elif failure_mode == FailureMode.INCOMPLETE_COMBUSTION:
                base_t *= 0.7
                base_k *= 0.8
            elif failure_mode == FailureMode.HOT_SPOTS:
                base_t *= np.random.uniform(1.3, 1.5)
            elif failure_mode == FailureMode.COMBUSTION_INSTABILITY:
                base_a *= np.random.uniform(0.5, 2.0)
                base_n = np.random.uniform(0.1, 0.8)

        propellant.setProperties({
            'name': 'Generated Propellant',
            'density': np.random.uniform(1600, 1900),
            'tabs': [{
                'minPressure': 0.5e6,
                'maxPressure': 10e6,
                'a': base_a,
                'n': base_n,
                'k': base_k,
                't': base_t,
                'm': np.random.uniform(20, 30)
            }]
        })
        motor.propellant = propellant

        # Generate grain
        grain = BatesGrain()
        diameter = np.random.uniform(0.05, 0.12)
        length = np.random.uniform(0.15, 0.25)
        core_diameter = diameter * np.random.uniform(0.3, 0.6)
        
        if failure_mode and failure_mode in [FailureMode.GRAIN_CRACKING,
                                           FailureMode.DELAMINATION,
                                           FailureMode.GRAIN_DEFORMATION]:
            if failure_mode == FailureMode.GRAIN_CRACKING:
                core_diameter *= np.random.uniform(1.1, 1.3)
            elif failure_mode == FailureMode.DELAMINATION:
                length *= np.random.uniform(0.7, 0.9)
            elif failure_mode == FailureMode.GRAIN_DEFORMATION:
                diameter *= np.random.uniform(0.8, 1.2)
        
        grain.setProperties({
            'diameter': diameter,
            'length': length,
            'coreDiameter': core_diameter,
            'inhibitedEnds': 'Neither'
        })
        motor.grains.append(grain)

        # Generate nozzle
        throat_diameter = np.random.uniform(0.01, 0.03)
        efficiency = np.random.uniform(0.85, 0.95)
        erosion_coeff = np.random.uniform(1e-9, 1e-8)
        
        if failure_mode and failure_mode in [FailureMode.NOZZLE_EROSION,
                                           FailureMode.NOZZLE_THROAT_CLOGGING,
                                           FailureMode.NOZZLE_DEFORMATION]:
            if failure_mode == FailureMode.NOZZLE_EROSION:
                erosion_coeff = np.random.uniform(1e-8, 1e-7)
            elif failure_mode == FailureMode.NOZZLE_THROAT_CLOGGING:
                throat_diameter *= np.random.uniform(0.6, 0.8)
            elif failure_mode == FailureMode.NOZZLE_DEFORMATION:
                efficiency *= np.random.uniform(0.5, 0.7)
        
        nozzle_props = {
            'throat': throat_diameter,
            'exit': throat_diameter * np.random.uniform(2.0, 3.5),
            'efficiency': efficiency,
            'divergenceAngle': np.random.uniform(12, 15),
            'throatLength': throat_diameter * np.random.uniform(0.6, 1.2),
            'erosionCoeff': erosion_coeff
        }
        motor.nozzle.setProperties(nozzle_props)

        # Save configuration with updated path structure
        config_data = motor.getDict()
        if failure_mode:
            output_folder = os.path.join("ric", failure_mode.name.lower())
        else:
            output_folder = os.path.join("ric", "normal")
            
        ric_path = os.path.join(output_dir, output_folder, f"motor_{id_num}.ric")
        os.makedirs(os.path.join(output_dir, output_folder), exist_ok=True)
        
        with open(ric_path, 'w') as f:
            json.dump(config_data, f)
        
        return (
            int(id_num),
            bool(failure_mode is not None),
            str(failure_mode.value if failure_mode else "None"),
            ric_path  # Return the ric path for simulation
        )
    
    except Exception as e:
        print(f"Error generating motor {id_num}: {str(e)}")
        return None

class MotorConfigGenerator:
    def __init__(self, output_dir="generated_data"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Create base directories
        os.makedirs(os.path.join(output_dir, "ric", "normal"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "csv", "normal"), exist_ok=True)
        
        # Create directories for each failure mode
        for failure_mode in FailureMode:
            failure_dir = failure_mode.name.lower()
            os.makedirs(os.path.join(output_dir, "ric", failure_dir), exist_ok=True)
            os.makedirs(os.path.join(output_dir, "csv", failure_dir), exist_ok=True)
        
    def generate_configurations(self, num_samples_per_failure):
        """Generate motor configurations only"""
        print(f"Generating {num_samples_per_failure} samples per failure mode...")
        
        # Prepare arguments for parallel processing
        args_list = []
        
        # Normal motors
        for i in range(num_samples_per_failure):
            args_list.append((i, None, self.output_dir))
        
        # Faulty motors
        counter = num_samples_per_failure
        for failure_mode in FailureMode:
            for i in range(num_samples_per_failure):
                args_list.append((counter, failure_mode, self.output_dir))
                counter += 1

        # Use multiprocessing to generate configurations
        num_cores = mp.cpu_count()
        print(f"Using {num_cores} CPU cores for configuration generation")
        
        with mp.Pool(num_cores) as pool:
            results = list(tqdm(
                pool.imap(generate_single_config, args_list),
                total=len(args_list),
                desc="Generating motor configurations"
            ))

        # Filter out None results
        valid_results = [r for r in results if r is not None]
        
        print(f"\nGenerated {len(valid_results)} motor configurations successfully")
        return valid_results

    def run_simulations(self, motor_configs):
        """Run simulations one at a time"""
        if not motor_configs:
            print("No valid configurations to simulate!")
            return None
        
        # Extract paths from configs
        ric_paths = [config[3] for config in motor_configs]
        
        print("\nRunning simulations one at a time...")
        total_motors = len(ric_paths)
        
        all_results = []
        for idx, motor_path in enumerate(ric_paths):
            print(f"\nProcessing motor {idx + 1}/{total_motors}")
            result = run_simulation(motor_path)
            all_results.append(result)
            
            # Report success/failure for current motor
            success = result[1]
            status = "Successfully simulated" if success else "Failed to simulate"
            print(f"{status} motor {idx + 1}")
        
        total_successful = sum(1 for _, success in all_results if success)
        print(f"\nTotal: Successfully simulated {total_successful} out of {total_motors} motors")
        
        return all_results

    # def create_dataset(self, motor_configs, sim_results):
        
    #     pass


    def generate_dataset(self, num_samples_per_failure):
        """Main method to generate dataset with clear separation of steps"""
        # Step 1: Generate configurations
        print("Step 1: Generating motor configurations...")
        motor_configs = self.generate_configurations(num_samples_per_failure)
        
        # Step 2: Run simulations
        print("\nStep 2: Running simulations...")
        sim_results = self.run_simulations(motor_configs)
        
        # Step 3: Create dataset
        # print("\nStep 3: Creating dataset...")
        # self.create_dataset(motor_configs, sim_results)
if __name__ == "__main__":
    generator = MotorConfigGenerator()
    generator.generate_dataset(num_samples_per_failure=1)
