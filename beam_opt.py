import os
import numpy as np
import math
from pathlib import Path
import multiprocessing
import random
import numpy as np
import shutil
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.problem import Problem
from pymoo.optimize import minimize
import multiprocessing
from multiprocessing import Manager
from pymoo.operators.sampling.lhs import LHS
from scipy.spatial.distance import cdist


def remove_elements(array, value):
    mask = array[:, 2] < value
    filtered_array = array[mask]
    return filtered_array.reshape(-1, 3)


def calculate_diameter_mid_height(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
        coordinates = [list(map(float, line.strip().split())) for line in lines]

    # Convert the coordinates to a NumPy array
        coordinates = np.array(coordinates)

    # Determine the minimum and maximum z-values
        min_z = np.min(coordinates[:, 2])
        max_z = np.max(coordinates[:, 2])

    # Calculate the mid-height of the wellbore
        mid_height = (min_z + max_z) / 2.0
        closest_idx = np.argmin(np.abs(coordinates[:, 2] - mid_height))
    # Get the closest coordinate
        closest_coordinate = coordinates[closest_idx]
        print("CLOSEST_COOR",closest_coordinate)
    # Extract the coordinates at the mid-height
        mid_height_coordinates = coordinates[np.isclose(coordinates[:, 2], closest_coordinate[2], atol = 1e-5)]
    # Calculate the pairwise distances between all points at the mid-height
        distances = cdist(mid_height_coordinates[:, :2], mid_height_coordinates[:, :2])
    # Find the maximum distance, which corresponds to the diameter at mid-height
        diameter_mid_height = np.max(distances)

        return diameter_mid_height

def calculate_volume(file_path):
    # Read the coordinates from the file
    with open(file_path, 'r') as file:
        lines = file.readlines()
        coordinates_unf = [list(map(float, line.strip().split())) for line in lines]

    # Convert the coordinates to a NumPy array
    coordinates_unf = np.array(coordinates_unf)
    coordinates = remove_elements(coordinates_unf, 0)
    # Calculate the volume using the Shoelace formula
    n = len(coordinates)
    volume = 0.0
    for i in range(n):
        j = (i + 1) % n
        volume += coordinates[i, 0] * coordinates[j, 1] * coordinates[j, 2] - coordinates[j, 0] * coordinates[i, 1] * coordinates[j, 2] \
                  + coordinates[j, 0] * coordinates[i, 1] * coordinates[i, 2] - coordinates[i, 0] * coordinates[j, 1] * coordinates[i, 2]
    volume /= 6.0
    return abs(volume)

def call_and_process(parameters):
    retval = dict([])
    os.system("mkdir workdir."+ str(parameters[len(parameters)-1])) 
    os.chdir("./workdir."+str(parameters[len(parameters)-1]))
    os.system("../amr3d.gnu.ex ../GraniteVaporisationTest2 he.laser_power = "+str(parameters[0])+" he.stand_off_distance = " +str(parameters[1])+" he.beam_radius = " +str(parameters[2])+" he.wave_guide_speed = "+str(parameters[3])+" he.rotation_radius = " +str(parameters[4])+" mat.liquid_density1 = " + str(parameters[5])+ " mat.latent_heat_vap1 = " + str(parameters[6]))
    path = Path("./codeComparisonTest1/points430")
    if (path.is_file()== False):
        f = 0.1
        g = 0.12
        h = float(parameters[0])
        os.chdir("..")
        shutil.rmtree("./workdir."+str(parameters[len(parameters)-1]))
        return f,g,h
    beam_power = float(parameters[0])
    beam_radius = float(parameters[1])
    f = -calculate_volume(path)
    if (calculate_diameter_mid_height(path)>(beam_radius)):
        g = -abs(calculate_diameter_mid_height(path)- (beam_radius))
    else:
        g = abs(calculate_diameter_mid_height(path)-beam_radius)
    h = beam_power
    os.chdir("..")
    shutil.rmtree("./workdir."+str(parameters[len(parameters)-1]))
    return f,g,h

class MyProblem(Problem):
    def __init__(self, thermal_params,it, **kwargs):
        super().__init__(n_var=5, n_obj=3, n_ieq_constr=0, xl=[10000, 0.001, 0.0009, 0.00005, 0.0005], xu=[30000, 0.01, 0.005,0.01, 0.05], **kwargs)
        self.thermal_params = thermal_params
        self.it = it
    def _evaluate(self, X, out, *args, **kwargs):
        def runSimulation(input):
            f,g,h = call_and_process(input)
            f_s.append([f,g,h])

          #  file3 = open('test_output'+str(it), 'a')
          #  file3.write(str(input[len(input)-1])+" "+str(f)+" "+ str(g)+" "+str(h)+'\n')
          #  file3.close()


        with Manager() as manager:
            f_s = manager.list()
            params = [[X[k]] for k in range(len(X))]
            processes = []
            n_procs=15
            for j in range(6):
                for i in range(n_procs):
 #                print([params[i][0],params[i][0], self.thermal_params[0],self.thermal_params[0],i])
                    p = multiprocessing.Process(target=runSimulation, args = np.column_stack([params[i], self.thermal_params[0],self.thermal_params[1],it,i+j*n_procs]))
                    p.start()
                    processes.append(p)
                for process in processes:
                    process.join()
            out["F"] = np.array(f_s)
latent_heat_evap = [4500000,5500000]
liquid_density = [2300,2500]
vertices=[[]]*(len(liquid_density)*len(latent_heat_evap))
i=0
solutions = [[]]*4
fitness = [[]]*4
for k in range(len(latent_heat_evap)):
    for l in range(len(liquid_density)):
        vertices[i] = [liquid_density[l], latent_heat_evap[k]]
        i+=1
for it in range(1):
    problem = MyProblem(vertices[it],it)
    algorithm = NSGA2(pop_size=90)
    res = minimize(problem, algorithm, termination=("n_gen", 5), seed=1, save_history = True, verbose = True)
    print(res.X)
    print(res.F)
    hist = res.history
    print(hist)
    solutions[it]=res.X
    fitness[it]=res.F
file1 = open('mesh_six.dat', 'w')
for i in range(1):
    file1.write("vertex:"+ str(i)+ str(vertices[i])+"solution:"+str(solutions[i])+" best fitness:"+ str(fitness[i])+'\n')
file1.close()

file2 = open('mesh_six_list.dat', 'w')
file2.write("vertices: "+str(vertices)+"solutions: "+str(solutions)+" fitness: "+ str(fitness)+'\n')
file2.close()
