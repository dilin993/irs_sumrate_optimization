from simulation import MUIRSSimuation, SimulationResult
import util
import numpy as np
import math
import matplotlib.pyplot as plt
from multiprocessing import Pool
simulation_count = 100
r1 = 20
gamma = 1
SNR_db = np.arange(-4, 21, 4)


def create_simulations():
    simulations = []
    for idx in range(simulation_count):
        simulation = MUIRSSimuation(
            1e9,  # fc Hz
            4,  # BS antennas
            (0, 0, 0),  # BS position
            10,  # IRS elements
            (0, 21, 0),  # IRS position
            -40,  # No dB
            SNR_db,  # SNR dB
            -20,  # c0 dB
            2.7,  # alpha bs-irs
            math.inf,  # beta bs-irs
            gamma)  # gamma

        for u in range(4):  # add 4 users
            theta = np.pi / 4 + (u - 1) * np.pi / 2
            p1 = simulation.bs.pos[0] + r1 * np.sin(theta)
            p2 = simulation.bs.pos[1] + r1 * np.cos(theta)
            p3 = simulation.bs.pos[2]
            simulation.add_user((p1, p2, p3),
                                2.5,  # alpha_bs_u
                                2.1,  # alpha_irs_u
                                util.db2lin(3),  # beta_bs_u
                                util.db2lin(3))  # beta_irs_u
        simulations.append(simulation)
    return simulations


def run_simulation(simulation):
    return simulation.simulate()


if __name__ == '__main__':
    results = []
    create_simulations()
    with Pool() as p:
        all_results = p.map(run_simulation, create_simulations())
    for sim_result in all_results:
        if len(results) < 1:
            for r in sim_result:
                results.append(r)
        else:
            for i in range(len(sim_result)):
                for k in range(len(sim_result[i].results)):
                    results[i].results[k].x = results[i].results[k].x + sim_result[i].results[k].x
                    results[i].results[k].y = results[i].results[k].y + sim_result[i].results[k].y

    for i in range(len(results)):
        for k in range(len(results[i].results)):
            results[i].results[k].x = results[i].results[k].x / simulation_count
            results[i].results[k].y = results[i].results[k].y / simulation_count

    for result in results:
        fig, ax = plt.subplots()
        r = result.results
        plt.plot(r[0].x, r[0].y, label=r[0].text, marker='s')
        plt.plot(r[1].x, r[1].y, label=r[1].text, marker='*', linestyle='--')
        plt.xlabel(result.xlabel)
        plt.ylabel(result.ylabel)
        plt.title(result.title)
        plt.legend()
        plt.show()
        fig.savefig(result.title + str(gamma))



