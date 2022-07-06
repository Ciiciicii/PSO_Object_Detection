# CANONICAL PSO
# Inclusion of constriction factor, controlling convergence properties of particles

import random 
import math
from cv2 import mean
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.animation import PillowWriter

# COST FUNCTION:

def distanceFunc(pos):
    centers = [(6,4), (1,2), (3.5,-2)]
    confidence = [0.99, 0.4, 0.2]
    for i, center in enumerate(centers):
        distance = math.sqrt((pos[0] - center[0])**2 + (pos[1] - center[1])**2)
        if distance <= 1:
            raw_fitness = (1-confidence[i]) + distance
            mapped_distance = 1 - ((1.5 - raw_fitness)/1.5)
            return mapped_distance
    return 1

class Particle:
    def __init__(self, initial):
        self.pos = []
        self.pos.append(initial)
        self.vel = []
        self.best_pos = []
        self.distance_travelled = 0
        self.best_fit = -1
        self.fit = -1
        for i in range(0,num_dimensions):
            self.vel.append(random.uniform(-1,1))

    def fitness(self, fitnessFunc, iter):
        self.fit = fitnessFunc(self.pos[iter])

        if self.fit < self.best_fit or self.best_fit == -1:
            self.best_pos = self.pos[iter]
            self.best_fit = self.fit

    def update_velocity(self, global_best, iter):
        w = 0.5        # inertial weight
        c1 = 2.05      # cognitive factor (local best)     
        c2 = 2.05      # social factor (global best)
        r1 = np.array([random.random(), random.random()])
        r2 = np.array([random.random(), random.random()])
        psi = c1 + c2
        k = 1
        X = 2*k/abs(2 - psi - math.sqrt(psi**2 - 4*psi))
        
        inertial = w * np.array(self.vel)
        cognitive = r1 * c1 * (np.array(self.best_pos) - np.array(self.pos[iter]))
        social = r2 * c2 * (np.array(global_best) - np.array(self.pos[iter]))
        vel = list(X * (inertial + cognitive + social))
        if abs(vel[0]) > 1:
            vel[0] == math.copysign(1,vel[0])
        if abs(vel[1]) > 1:
            vel[1] == math.copysign(1,vel[1])
        self.vel = vel

    def update_position(self, bounds, iter):
        next_pos = list(np.array(self.pos[iter]) + np.array(self.vel))
        for i in range(num_dimensions):
            dimbound = bounds[i] 
            if next_pos[i] < dimbound[0]:
                next_pos[i] = dimbound[0]
            if next_pos[i] > dimbound[1]:
                next_pos[i] = dimbound[1]
        self.distance_travelled += math.dist(self.pos[iter], next_pos)
        self.pos.append(next_pos)


class PSO():
    def __init__(self,bounds,num_particles,maxiter,fitnessFunc,tolerance=0.1):
        self.bounds = bounds
        self.num_particles = num_particles
        self.maxiter = maxiter
        self.fitnessFunc = fitnessFunc
        self.tolerance = tolerance
        self.swarm = []
        self.fit_history = []
        self.iteration_end = 0
        self.global_best_pos = []

    def optimize(self):
        global num_dimensions

        self.swarm = []
        self.fit_history = []
        self.iteration_end = 0

        num_dimensions = len(self.bounds)
        global_best_fit = -1
        self.global_best_pos = []

        for i in range(self.num_particles):
            particle_init_position = []
            for j in range(num_dimensions):
                ith_dim_bounds = self.bounds[j]
                pos_dim = random.uniform(ith_dim_bounds[0], ith_dim_bounds[1])
                particle_init_position.append(pos_dim)
            particle = Particle(particle_init_position)
            self.swarm.append(particle)

        # begin PSO loop
        for i in range(self.maxiter):
            # cycle through particles and check fitness
            for j in range(self.num_particles):
                self.swarm[j].fitness(self.fitnessFunc, i)

                # update global best if possible
                if self.swarm[j].fit < global_best_fit or global_best_fit == -1:
                    self.global_best_pos = self.swarm[j].pos[i]
                    global_best_fit = self.swarm[j].fit

            if global_best_fit == -1:
                self.fit_history.append(14.142)
            else:
                self.fit_history.append(global_best_fit)

            # update velocities and positions
            for j in range(self.num_particles):
                self.swarm[j].update_velocity(self.global_best_pos, i)
                self.swarm[j].update_position(self.bounds, i)

            if self.converge():
                self.iteration_end = i
                break
            elif i == self.maxiter - 1:
                self.iteration_end = self.maxiter

    def converge(self):
        for particle in self.swarm:
            if particle.best_fit > self.tolerance:
                return False
        return True
    
    def plot_fit_history(self):
        plt.plot(self.fit_history)
        plt.xlim(0,self.iteration_end)
        plt.show()

    def animate_pso(self,save=False):
        fig, ax = plt.subplots()
        ln, = plt.plot(1,1)
        ln, = plt.plot([], [],'go',markersize=5)
        

        def lim():
            ax.set_xlim(bounds[0])
            ax.set_ylim(bounds[1])
            return ln,
        
        def update(frame):
            xdata, ydata = [], []
            for particle in self.swarm:
                pos = particle.pos[frame]
                xdata.append(pos[0])
                ydata.append(pos[1])
            ln.set_data(xdata,ydata)
            return ln,
        
        anim = FuncAnimation(fig, update, frames=self.iteration_end, interval=100, blit=True, init_func=lim)
        plt.show()
        anim.save('swarm_ani.gif', writer='pillow')
    
    def analyze_pso(self, runs=200):
        iterations = []
        global_optimum_reached = []
        distance = []
        for i in range(runs):
            self.optimize()
            iterations.append(self.iteration_end)
            if abs(self.global_best_pos[0] - 6) <= 1 and abs(self.global_best_pos[1] - 4) <= 1:
                global_optimum_reached.append(1)
            else:
                global_optimum_reached.append(0)
            run_distance = 0
            for particle in self.swarm:
                run_distance += particle.distance_travelled
            distance.append(run_distance)

            if i%10 == 0: 
                print("Runs completed:", i)

        print("Average No. of Iterations:", np.average(iterations))
        print("Average distance travelled by swarm:", np.average(distance))
        print("Percentage of global optimum reached:", sum(global_optimum_reached)/runs)




if __name__ == "__main__":
    bounds = [(-10,10),(-10,10)] # x range, y range
    num_particles = 3
    pso = PSO(bounds, num_particles, 1000, distanceFunc, 0.1)

    # Run PSO once 
    # pso.optimize()

    # Plot fitness history
    # pso.plot_fit_history()

    # Show animation, save = True if animation will be saved as .gif 
    # pso.animate_pso()

    # Analyze over a number of runs, default = 200
    pso.analyze_pso()

