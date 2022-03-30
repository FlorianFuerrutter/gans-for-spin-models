import numpy as np
import data_visualization
import model_evaluation
import data_analysis
import os
import matplotlib
import matplotlib.pyplot as plt
matplotlib.rcParams.update({
    'text.usetex': False,
    'font.family': 'serif',
    'font.serif': 'cmr10',
    'font.size': 16,
    'mathtext.fontset': 'cm',
    'font.family': 'STIXGeneral',
    'axes.unicode_minus': True})

#--------------------------------------------------------------------

plot_path  = os.path.dirname(__file__)
def savePdf(filename): 
    plt.savefig(plot_path + "/" + filename + '.pdf', bbox_inches='tight')
def savePng(filename):
    plt.savefig(plot_path + "/" + filename + '.png', bbox_inches='tight')

#--------------------------------------------------------------------

def load_spin_states(TJ, L):
    path = os.path.join(os.path.dirname(__file__), "..", "data", "train", str(L))
    file_path = os.path.join(path, "simulation_states_TJ_{TJ}.npy".format(TJ=TJ))

    states = np.load(file_path)

    print("[load_spin_states] (TJ, L) = (%.2f, %d), Found states count:" % (TJ, L), states.shape[0])
    return states

def load_spin_observables(TJ, L):
    path = os.path.join(os.path.dirname(__file__), "..", "data", "train", L)
    file_path = os.path.join(path, "simulation_observ_TJ_{TJ}.npy".format(TJ=TJ))

    obser = np.transpose(np.load(file_path))
    energy = obser[0]
    m      = obser[1]
    mAbs   = obser[2]
    m2     = obser[3]
    mAbs3  = obser[4]
    m4     = obser[5]
    
    print("[load_spin_observables] (TJ, L) = (%.2f, %d), Found data count:" % (TJ, L), energy.shape[0])
    return energy, m, mAbs, m2, mAbs3, m4

#--------------------------------------------------------------------

def main():
    Tc = 1.0 * 2.0 / np.log(1.0 + np.sqrt(2.0))

    Ls  = np.array([8, 16, 32, 64, 128], dtype=np.int32)
    TJs = np.array([1.0, 1.5, 1.8, 2.0, 2.1, 2.2, 2.25, 2.26, 2.27, 2.3, 2.4, 2.5, 2.6, 3.0, 3.4,
                    2.28, 2.29, 2.31, 2.32, 2.33, 2.34, 2.35, 2.36, 2.37, 2.38, 2.39])

    TJs = np.sort(TJs)

    #--------------------------

    size=(12, 5)
    fig = plt.figure(figsize = size, constrained_layout = True) 
    plt.xlabel(r"$T/J$")
    plt.ylabel(r"$\xi$")

    plt.axvline(Tc, color="gray", linestyle="--")

    clrs = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple"]

    #--------------------------

    for i in range(Ls.size):
        L   = Ls[i]
        clr = clrs[i]

        #--------------------------
        #legend
        te = r"$L$ = %d" % L
        args = dict(horizontalalignment='left',verticalalignment='top', transform=plt.gca().transAxes, color=clr, size="large")
        plt.text(1.02, 0.98-0.12*i, te, args)

        #--------------------------
        xis     = list()
        xis_err = list()
      
        for TJ in TJs:
            print("------ (TJ, L) = (%.2f, %d) ------" % (TJ, L))
            N = L*L

            #load sim data
            #energy, m, mAbs, m2, mAbs3, m4 = load_spin_observables(TJ, L)

            states = load_spin_states(TJ, L)

            #calc obs
            xi, xi_err = data_analysis.calc_spin_spin_correlation(states, N)

            xis.append(xi)
            xis_err.append(xi_err)
        #--------------------------

        plt.plot(TJs, xis, "--", color=clr, alpha=0.5, linewidth=0.8)
        plt.errorbar(TJs, xis, fmt='.', yerr=xis_err, elinewidth=1, capsize=5, markersize=5, color=clr)
   
    savePdf("simulation_evaluation")
    savePng("simulation_evaluation")
    plt.show()
    return

#--------------------------------------------------------------------

if __name__ == '__main__':
    main()
