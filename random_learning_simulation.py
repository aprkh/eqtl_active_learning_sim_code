#!/usr/bin/env python3
############################################################
# active_learning_simulation.py
############################################################

import sys
import numpy as np
import subprocess

# some other parameters
THRESHOLD = 200
MAXITER = 200
LTHRESH = 0.85
GTHRESH = 0.85

# directory names
ACTIVE_LEARNING_DIR = "active_learning_sims"

GENERAL_PREFIX = "/mnt/c/Users/apare/Desktop/KimResearchGroup/Spring2022/"
TO_CITRUSS = GENERAL_PREFIX + "mlcggm/Mega-sCGGM/citruss.py"
TO_DATA = "input_simulation/simulateCode2/"


def main():
    ysum_file_large = ACTIVE_LEARNING_DIR + "/" + "0Ysum_large.txt"
    ym_file_large = ACTIVE_LEARNING_DIR + "/" + "0Ym_large.txt"
    yp_file_large = ACTIVE_LEARNING_DIR + "/" + "0Yp_large.txt"
    xm_file_large = ACTIVE_LEARNING_DIR + "/" + "0Xm_large.txt"
    xp_file_large = ACTIVE_LEARNING_DIR + "/" + "0Xp_large.txt"

    ysum_file_small = ACTIVE_LEARNING_DIR + "/" + "0Ysum_small.txt"
    ym_file_small = ACTIVE_LEARNING_DIR + "/" + "0Ym_small.txt"
    yp_file_small = ACTIVE_LEARNING_DIR + "/" + "0Yp_small.txt"
    xm_file_small = ACTIVE_LEARNING_DIR + "/" + "0Xm_small.txt"
    xp_file_small = ACTIVE_LEARNING_DIR + "/" + "0Xp_small.txt"

    random_learning_sim(ysum_file_large, ym_file_large, yp_file_large, xm_file_large, xp_file_large,
                        ysum_file_small, ym_file_small, yp_file_small, xm_file_small, xp_file_small,
                        maxiter=MAXITER, general_prefix=GENERAL_PREFIX,
                        active_learning_dir=ACTIVE_LEARNING_DIR,
                        to_citruss=TO_CITRUSS, threshold=THRESHOLD)


# ---------------------------------------------------------------------
# run the random learning simulation in an automated fashion
# ---------------------------------------------------------------------
def random_learning_sim(start_ysum_large, start_ym_large, start_yp_large, start_xm_large, start_xp_large,
                        start_ysum_small, start_ym_small, start_yp_small, start_xm_small, start_xp_small,
                        maxiter=MAXITER, general_prefix=GENERAL_PREFIX,
                        active_learning_dir=ACTIVE_LEARNING_DIR,
                        to_citruss=TO_CITRUSS, to_data=TO_DATA,
                        threshold=THRESHOLD):
    """
    Run the active learning simulation.
    Inputs:
        start_ysum (np.array) - starting total gene expressions
        start_ym (np.array) - starting maternal gene expressions
        start_yp (np.array) - starting paternal gene expressions
        start_xm (np.array) - starting maternal SNPs
        start_xp (np.array) - starting paternal SNPs
        maxiter (int) - number of maximum iterations
        general_prefix (str) - general prefix to project directory
        active_learning_dir (str) - directory to where we store the results
                                    (relative to general_prefix)
        to_citruss (str) - path to citruss.py command
        threshold (int) - minimum number of ASE needed for each gene
        prop (float) - initial proportion of observations sampled
    Outputs:
        None - files saved to active_learning_dir
    """
    initialize_dataset(active_learning_dir, '0', start_ysum_small, start_ym_small, start_yp_small,
                       start_xm_small, start_xp_small, start_ysum_large, start_ym_large, start_yp_large,
                       start_xm_large, start_xp_large)

    for iiter in range(maxiter):
        fysum = general_prefix + to_data + active_learning_dir + \
            "/{}Ysum_small_random.txt".format(iiter)
        fym = general_prefix + to_data + active_learning_dir + \
            "/{}Ym_small_random.txt".format(iiter)
        fyp = general_prefix + to_data + active_learning_dir + \
            "/{}Yp_small_random.txt".format(iiter)
        fxm = general_prefix + to_data + active_learning_dir + \
            "/{}Xm_small_random.txt".format(iiter)
        fxp = general_prefix + to_data + active_learning_dir + \
            "/{}Xp_small_random.txt".format(iiter)

        run_citruss(fysum, fym, fyp, fxm, fxp,
                    general_prefix + to_data +
                    active_learning_dir + "/" + str(iiter) + "random",
                    0.01, 0.01, 0.01, 0.01, to_citruss)

        V = np.loadtxt(general_prefix + to_data +
                       active_learning_dir + "/" + str(iiter) + "V.txt")
        F = np.loadtxt(general_prefix + to_data +
                       active_learning_dir + "/" + str(iiter) + "F.txt")
        Gamma = np.loadtxt(general_prefix + to_data +
                           active_learning_dir + "/" + str(iiter) + "Gamma.txt")
        Psi = np.loadtxt(general_prefix + to_data +
                         active_learning_dir + "/" + str(iiter) + "Psi.txt")

        Omega, Xi, Pi = get_params(V, F, Gamma, Psi)

        # find people heterozygous for these traits in the remaining samples
        fysum_large = general_prefix + to_data + \
            active_learning_dir + "/" + str(iiter) + "Ysum_large_random.txt"
        fym_large = general_prefix + to_data + \
            active_learning_dir + "/" + str(iiter) + "Ym_large_random.txt"
        fyp_large = general_prefix + to_data + \
            active_learning_dir + "/" + str(iiter) + "Yp_large_random.txt"
        fxm_large = general_prefix + to_data + \
            active_learning_dir + "/" + str(iiter) + "Xm_large_random.txt"
        fxp_large = general_prefix + to_data + \
            active_learning_dir + "/" + str(iiter) + "Xp_large_random.txt"

        fysum_small = general_prefix + to_data + \
            active_learning_dir + "/" + str(iiter) + "Ysum_small_random.txt"
        fym_small = general_prefix + to_data + \
            active_learning_dir + "/" + str(iiter) + "Ym_small_random.txt"
        fyp_small = general_prefix + to_data + \
            active_learning_dir + "/" + str(iiter) + "Yp_small_random.txt"
        fxm_small = general_prefix + to_data + \
            active_learning_dir + "/" + str(iiter) + "Xm_small_random.txt"
        fxp_small = general_prefix + to_data + \
            active_learning_dir + "/" + str(iiter) + "Xp_small_random.txt"

        # ym_large = np.loadtxt(fym_large)
        # yp_large = np.loadtxt(fyp_large)

        # get number of samples needed
        fysum_next = general_prefix + to_data + \
            active_learning_dir + "/" + str(iiter+1) + "Ysum_small.txt"
        N = np.loadtxt(fysum_large).shape[0]
        nnext = np.loadtxt(
            fysum_next).shape[0] - np.loadtxt(fysum_small).shape[0]
        print('nnext:', nnext)
        print('N:', N)
        new_people = np.random.choice(np.arange(0, N, 1, dtype=np.int64), nnext,
                                      replace=False)

        print("{} new people".format(len(new_people)), file=sys.stderr)

        update_dataset(active_learning_dir, str(iiter+1), fysum_large, fysum_small, fym_large, fym_small,
                       fyp_large, fyp_small, fxm_large, fxm_small, fxp_large, fxp_small,
                       new_people)

    fysum = general_prefix + to_data + active_learning_dir + \
        "/{}Ysum_small.txt".format(maxiter)
    fym = general_prefix + to_data + active_learning_dir + \
        "/{}Ym_small.txt".format(maxiter)
    fyp = general_prefix + to_data + active_learning_dir + \
        "/{}Yp_small.txt".format(maxiter)
    fxm = general_prefix + to_data + active_learning_dir + \
        "/{}Xm_small.txt".format(maxiter)
    fxp = general_prefix + to_data + active_learning_dir + \
        "/{}Xp_small.txt".format(maxiter)
    run_citruss(fysum, fym, fyp, fxm, fxp,
                general_prefix + to_data +
                active_learning_dir + "/" + str(maxiter) + "random",
                0.01, 0.01, 0.01, 0.01, to_citruss)

# ---------------------------------------------------------------------
# Run citruss.py on a dataset; reconstruct parameters
# ---------------------------------------------------------------------


def run_citruss(fysum, fym, fyp, fxm, fxp, output_prefix,
                vreg, freg, gammareg, psireg, citruss_path):
    """
    Run citruss on a dataset with the given parameters.
    """
    # get N, q, p
    N, q = np.loadtxt(fysum).shape
    _, p = np.loadtxt(fxm).shape

    cmd_list = ['python', citruss_path, str(N), str(q), str(p),
                fysum, fym, fyp, fxm, fxp, output_prefix,
                str(vreg), str(freg), str(gammareg), str(psireg)]

    subprocess.run(cmd_list, check=True)


def get_params(V, F, Gamma, Psi):
    """
    Reconstruct Omega, Xi, and Pi from the input parameters.
    """
    Omega = V - Gamma
    Pi = 2 * Psi
    Xi = F
    Xi[np.nonzero(Pi)] = 0
    return Omega, Xi, Pi


# ---------------------------------------------------------------------
# Initialize active learning dataset, update dataset after round
# ---------------------------------------------------------------------
def update_dataset(outdir, outprefix,
                   fysum_large, fysum_small,
                   fym_large, fym_small,
                   fyp_large, fyp_small,
                   fxm_large, fxm_small,
                   fxp_large, fxp_small,
                   set_cover_people):
    """
    Adds people from set cover to new dataset of RNA-sequenced people.
    Removes people from set cover of non-RNA-sequences people.
    Inputs:
        outdir (str) - the folder in which to save the initialized dataset.
        outprefix (str) - the prefix to give the saved files
        fysum_large (str) - the name of the file containing old Ysum_large
        fysum_small (str) - the name of the file containing old Ysum_small
        fym_large (str) - the name of the file containing old Ym_large
        fym_small (str) - the name of the file containing old Ym_small
        fyp_large (str) - the name of the file containing old Yp_large
        fyp_small (str) - the name of the file containingm old Yp_small
        fxm_large (str) - the name of the file containing old Xm_large
        fxm_small (str) - the name of the file containing old Xm_small
        fxp_large (str) - the name of the file containing old Xp_large
        fxp_small (str) - the name of the file containing old Xp_small
        set_cover_people (np.array) - people to be sequenced
    Outputs - none (saves files to outdir)
    """
    ysum_large = np.loadtxt(fysum_large)
    ysum_small = np.loadtxt(fysum_small)
    ym_large = np.loadtxt(fym_large)
    ym_small = np.loadtxt(fym_small)
    yp_large = np.loadtxt(fyp_large)
    yp_small = np.loadtxt(fyp_small)
    xm_large = np.loadtxt(fxm_large)
    xm_small = np.loadtxt(fxm_small)
    xp_large = np.loadtxt(fxp_large)
    xp_small = np.loadtxt(fxp_small)

    Nr, q = ysum_large.shape
    _, p = xp_large.shape

    mask = np.zeros(Nr, dtype=bool)
    mask[set_cover_people] = 1

    ysum_small = np.vstack((ysum_small, ysum_large[mask, :]))
    ym_small = np.vstack((ym_small, ym_large[mask, :]))
    yp_small = np.vstack((yp_small, yp_large[mask, :]))
    xm_small = np.vstack((xm_small, xm_large[mask, :]))
    xp_small = np.vstack((xp_small, xp_large[mask, :]))

    ysum_large = ysum_large[np.logical_not(mask), :]
    ym_large = ym_large[np.logical_not(mask), :]
    yp_large = yp_large[np.logical_not(mask), :]
    xm_large = xm_large[np.logical_not(mask), :]
    xp_large = xp_large[np.logical_not(mask), :]

    np.savetxt(file_path(outdir, outprefix,
               "Ysum_small_random.txt"), ysum_small)
    np.savetxt(file_path(outdir, outprefix, "Ym_small_random.txt"), ym_small)
    np.savetxt(file_path(outdir, outprefix, "Yp_small_random.txt"), yp_small)
    np.savetxt(file_path(outdir, outprefix, "Xm_small_random.txt"), xm_small)
    np.savetxt(file_path(outdir, outprefix, "Xp_small_random.txt"), xp_small)

    np.savetxt(file_path(outdir, outprefix,
               "Ysum_large_random.txt"), ysum_large)
    np.savetxt(file_path(outdir, outprefix, "Ym_large_random.txt"), ym_large)
    np.savetxt(file_path(outdir, outprefix, "Yp_large_random.txt"), yp_large)
    np.savetxt(file_path(outdir, outprefix, "Xm_large_random.txt"), xm_large)
    np.savetxt(file_path(outdir, outprefix, "Xp_large_random.txt"), xp_large)


def initialize_dataset(outdir, outprefix, fysum, fym, fyp, fxm, fxp,
                       fysum2, fym2, fyp2, fxm2, fxp2):
    """
    Initialize a dataset for an active learning simulation.
    Inputs:
        outdir (str) - the folder in which to save the initialized dataset.
        outprefix (str) - the prefix to give the saved files
        fysum (str) - the name of the file containing Ysum
        fym (str) - the name of the file containing Ym
        fyp (str) - the name of the file containingm Yp
        fxm (str) - the name of the file containing Xm
        fxp (str) - the name of the file containing Xp
    Outputs - none (saves files to outdir)
    """
    ysum = np.loadtxt(fysum)
    ym = np.loadtxt(fym)
    yp = np.loadtxt(fyp)
    xm = np.loadtxt(fxm)
    xp = np.loadtxt(fxp)

    ysum2 = np.loadtxt(fysum2)
    ym2 = np.loadtxt(fym2)
    yp2 = np.loadtxt(fyp2)
    xm2 = np.loadtxt(fxm2)
    xp2 = np.loadtxt(fxp2)

    np.savetxt(file_path(outdir, outprefix,
               "Ysum_small_random.txt"), ysum)
    np.savetxt(file_path(outdir, outprefix, "Ym_small_random.txt"), ym)
    np.savetxt(file_path(outdir, outprefix, "Yp_small_random.txt"), yp)
    np.savetxt(file_path(outdir, outprefix, "Xm_small_random.txt"), xm)
    np.savetxt(file_path(outdir, outprefix, "Xp_small_random.txt"), xp)

    np.savetxt(file_path(outdir, outprefix,
               "Ysum_large_random.txt"), ysum2)
    np.savetxt(file_path(outdir, outprefix, "Ym_large_random.txt"), ym2)
    np.savetxt(file_path(outdir, outprefix, "Yp_large_random.txt"), yp2)
    np.savetxt(file_path(outdir, outprefix, "Xm_large_random.txt"), xm2)
    np.savetxt(file_path(outdir, outprefix, "Xp_large_random.txt"), xp2)


def file_path(outdir, outprefix, fname):
    return ''.join((outdir, '/', outprefix, fname))


# ---------------------------------------------------------------------
# Taking subsets of the people and determining needed genes, set cover
# ---------------------------------------------------------------------
# will need to change!
def random_subset_data(ysum, ym, yp, xm, xp, nsample):
    """
    Return SNP and gene expression matrices for a random subset of the N
    people.
    Inputs:
        ysum (np.array) - total gene expression array
        ym (np.array) - maternal gene expression array
        yp (np.array) - paternal gene expression array
        xm (np.array) - maternal SNP genotypes
        xp (np.array) - paternal SNP genotypes
        prop (float) - proportion of people to sample
    Outputs:
        [(ysum_small, ym_small, yp_small, xm_small, xp_small),
         (ysum_large, ym_large, yp_large, xm_large, xp_large)]
    """
    # get parameters of whole dataset
    N, q = ysum.shape
    _, p = xm.shape

    # get a random sample
    sample_mask = np.repeat(False, N)
    true_idx = np.random.choice(np.arange(0, N, 1, dtype=np.int64), nsample,
                                replace=False)
    sample_mask[true_idx] = True

    ysum_small = ysum[sample_mask, :]
    ym_small = ym[sample_mask, :]
    yp_small = yp[sample_mask, :]
    xm_small = xm[sample_mask, :]
    xp_small = xp[sample_mask, :]

    ysum_large = ysum[np.logical_not(sample_mask), :]
    ym_large = ym[np.logical_not(sample_mask), :]
    yp_large = yp[np.logical_not(sample_mask), :]
    xm_large = xm[np.logical_not(sample_mask), :]
    xp_large = xp[np.logical_not(sample_mask), :]

    return [(ysum_small, ym_small, yp_small, xm_small, xp_small),
            (ysum_large, ym_large, yp_large, xm_large, xp_large)]


if __name__ == '__main__':
    main()
