#!/usr/bin/env python3 
############################################################
# active_learning_simulation.py 
############################################################

import sys 
import numpy as np 
import subprocess

# for set cover algorithm 
sys.path.append("/mnt/c/Users/apare/Desktop/KimResearchGroup/Spring2022/setCoverProblem/")

import set_cover_greedy


# for initializing the dataset
INIT_PROP = 0.10

# some other parameters
THRESHOLD = 200
MAXITER = 2
LTHRESH = 0.80
GTHRESH = 0.80

# directory names
ACTIVE_LEARNING_DIR = "active_learning_sims"

GENERAL_PREFIX = "/mnt/c/Users/apare/Desktop/KimResearchGroup/Spring2022/"
TO_CITRUSS = GENERAL_PREFIX + "mlcggm/Mega-sCGGM/citruss.py"
TO_DATA = GENERAL_PREFIX + "input_simulation/simulateCode2/missing"


def main():
    ysum_file = "missing35/Ysum1.txt"
    ym_file = "missing35/Ym1.txt"
    yp_file = "missing35/Yp1.txt"
    xm_file = "missing35/Xm1.txt"
    xp_file = "missing35/Xp1.txt"

    active_learning_sim(ysum_file, ym_file, yp_file, xm_file, xp_file,
                        maxiter=MAXITER, general_prefix=GENERAL_PREFIX, 
                        active_learning_dir=ACTIVE_LEARNING_DIR, 
                        to_citruss=TO_CITRUSS, threshold=THRESHOLD, 
                        prop=INIT_PROP)

#---------------------------------------------------------------------
# run the active learning simulation in an automated fashion
#---------------------------------------------------------------------
def active_learning_sim(start_ysum, start_ym, start_yp, start_xm, start_xp,
                        maxiter=MAXITER, general_prefix=GENERAL_PREFIX, 
                        active_learning_dir=ACTIVE_LEARNING_DIR, 
                        to_citruss=TO_CITRUSS, threshold=THRESHOLD, 
                        prop=INIT_PROP):
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
    initialize_dataset(active_learning_dir, '0', start_ysum, start_ym, start_yp,
                       start_xm, start_xp, prop)

    for iiter in range(maxiter):
        fysum = general_prefix + "input_simulation/simulateCode2/"  + active_learning_dir + "/{}Ysum_small.txt".format(iiter)
        fym = general_prefix + "input_simulation/simulateCode2/"  + active_learning_dir + "/{}Ym_small.txt".format(iiter)
        fyp = general_prefix + "input_simulation/simulateCode2/"  + active_learning_dir + "/{}Yp_small.txt".format(iiter)
        fxm = general_prefix + "input_simulation/simulateCode2/"  + active_learning_dir + "/{}Xm_small.txt".format(iiter)
        fxp = general_prefix + "input_simulation/simulateCode2/"  + active_learning_dir + "/{}Xp_small.txt".format(iiter)

        run_citruss(fysum, fym, fyp, fxm, fxp,
                    general_prefix + "input_simulation/simulateCode2/" + active_learning_dir + "/" + str(iiter),
                    0.01, 0.01, 0.01, 0.01, to_citruss)

    
        V = np.loadtxt(general_prefix + "input_simulation/simulateCode2/"  + active_learning_dir + "/" + str(iiter) + "V.txt")
        F = np.loadtxt(general_prefix + "input_simulation/simulateCode2/"  + active_learning_dir + "/" + str(iiter) + "F.txt")
        Gamma = np.loadtxt(general_prefix + "input_simulation/simulateCode2/"  + active_learning_dir + "/" + str(iiter) + "Gamma.txt")
        Psi = np.loadtxt(general_prefix + "input_simulation/simulateCode2/"  + active_learning_dir + "/" + str(iiter) + "Psi.txt")

        Omega, Xi, Pi = get_params(V, F, Gamma, Psi)

        # determine needed genes
        #ym = np.loadtxt(fym) 
        #yp = np.loadtxt(fyp)
        needed_eQTLs = determine_needed_eqtls(Xi, Pi, np.loadtxt(fym), np.loadtxt(fyp), 
                                              np.loadtxt(fxm), np.loadtxt(fxp), LTHRESH, GTHRESH)

        # determine if we even need to do another sampling 
        if len(needed_eQTLs) < 1:
            print("All genes have been sampled")
            break

        # find people heterozygous for these traits in the remaining samples 
        fysum_large = general_prefix + "input_simulation/simulateCode2/"  + active_learning_dir + "/" + str(iiter) + "Ysum_large.txt"
        fym_large = general_prefix + "input_simulation/simulateCode2/"  + active_learning_dir + "/" + str(iiter) + "Ym_large.txt"
        fyp_large = general_prefix + "input_simulation/simulateCode2/"  + active_learning_dir + "/" + str(iiter) + "Yp_large.txt"
        fxm_large = general_prefix + "input_simulation/simulateCode2/"  + active_learning_dir + "/" + str(iiter) + "Xm_large.txt"
        fxp_large = general_prefix + "input_simulation/simulateCode2/"  + active_learning_dir + "/" + str(iiter) + "Xp_large.txt"

        fysum_small = general_prefix + "input_simulation/simulateCode2/"  + active_learning_dir + "/" + str(iiter) + "Ysum_small.txt"
        fym_small = general_prefix + "input_simulation/simulateCode2/"  + active_learning_dir + "/" + str(iiter) + "Ym_small.txt"
        fyp_small = general_prefix + "input_simulation/simulateCode2/"  + active_learning_dir + "/" + str(iiter) + "Yp_small.txt"
        fxm_small = general_prefix + "input_simulation/simulateCode2/"  + active_learning_dir + "/" + str(iiter) + "Xm_small.txt"
        fxp_small = general_prefix + "input_simulation/simulateCode2/"  + active_learning_dir + "/" + str(iiter) + "Xp_small.txt"

        #ym_large = np.loadtxt(fym_large)
        #yp_large = np.loadtxt(fyp_large)

        people_array, people_sets = to_set_cover(np.loadtxt(fxm_large), np.loadtxt(fxp_large),
                                                 np.loadtxt(fym_large), np.loadtxt(fyp_large), 
                                                 needed_eQTLs)

        _, new_people = set_cover_greedy.set_cover_greedy(people_sets, needed_eQTLs)
        new_people = [people_array[j] for j in new_people]

        print("{} new people".format(len(new_people)))

        update_dataset(active_learning_dir, str(iiter+1), fysum_large, fysum_small, fym_large, fym_small, 
                       fyp_large, fyp_small, fxm_large, fxm_small, fxp_large, fxp_small,
                       new_people)

    run_citruss(fysum, fym, fyp, fxm, fxp,
                general_prefix + "input_simulation/simulateCode2/" + active_learning_dir + "/" + "Final",
                0.01, 0.01, 0.01, 0.01, to_citruss)

#---------------------------------------------------------------------
# Run citruss.py on a dataset; reconstruct parameters 
#---------------------------------------------------------------------
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


#---------------------------------------------------------------------
# Initialize active learning dataset, update dataset after round
#---------------------------------------------------------------------
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

    np.savetxt(file_path(outdir, outprefix, "Ysum_small.txt"), ysum_small)
    np.savetxt(file_path(outdir, outprefix, "Ym_small.txt"), ym_small)
    np.savetxt(file_path(outdir, outprefix, "Yp_small.txt"), yp_small)
    np.savetxt(file_path(outdir, outprefix, "Xm_small.txt"), xm_small)
    np.savetxt(file_path(outdir, outprefix, "Xp_small.txt"), xp_small)

    np.savetxt(file_path(outdir, outprefix, "Ysum_large.txt"), ysum_large)
    np.savetxt(file_path(outdir, outprefix, "Ym_large.txt"), ym_large)
    np.savetxt(file_path(outdir, outprefix, "Yp_large.txt"), yp_large)
    np.savetxt(file_path(outdir, outprefix, "Xm_large.txt"), xm_large)
    np.savetxt(file_path(outdir, outprefix, "Xp_large.txt"), xp_large)


def initialize_dataset(outdir, outprefix, fysum, fym, fyp, fxm, fxp, prop):
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
        prop (float) - proportion of people to sample
    Outputs - none (saves files to outdir)
    """
    ysum = np.loadtxt(fysum)
    ym = np.loadtxt(fym) 
    yp = np.loadtxt(fyp)
    xm = np.loadtxt(fxm) 
    xp = np.loadtxt(fxp) 
    
    subset, remaining = random_subset_data(ysum, ym, yp, xm, xp, prop) 

    ysum_small, ym_small, yp_small, xm_small, xp_small = subset 
    ysum_large, ym_large, yp_large, xm_large, xp_large = remaining 

    np.savetxt(file_path(outdir, outprefix, "Ysum_small.txt"), ysum_small)
    np.savetxt(file_path(outdir, outprefix, "Ym_small.txt"), ym_small)
    np.savetxt(file_path(outdir, outprefix, "Yp_small.txt"), yp_small)
    np.savetxt(file_path(outdir, outprefix, "Xm_small.txt"), xm_small)
    np.savetxt(file_path(outdir, outprefix, "Xp_small.txt"), xp_small)

    np.savetxt(file_path(outdir, outprefix, "Ysum_large.txt"), ysum_large)
    np.savetxt(file_path(outdir, outprefix, "Ym_large.txt"), ym_large)
    np.savetxt(file_path(outdir, outprefix, "Yp_large.txt"), yp_large)
    np.savetxt(file_path(outdir, outprefix, "Xm_large.txt"), xm_large)
    np.savetxt(file_path(outdir, outprefix, "Xp_large.txt"), xp_large)


def file_path(outdir, outprefix, fname):
    return ''.join((outdir, '/', outprefix, fname))


#---------------------------------------------------------------------
# Taking subsets of the people and determining needed genes, set cover
#---------------------------------------------------------------------
# will need to change!
def random_subset_data(ysum, ym, yp, xm, xp, prop):
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
    nsample = np.int64(N * prop) 
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


# will need to change!
def determine_needed_eqtls(xi, pi, ym, yp, xm, xp, Lthresh, Gthresh):
    """
    Determines the needed eQTLs, which happens when we do not have enough 
    people who are heterozygous at both the SNP and the expressed gene.
    """
    needed_eqtls = []
    N, q = ym.shape
    _, p = xm.shape

    # check cis eqtls
    x, y = np.nonzero(pi)
    for i, j in zip(x, y):
        print(determine_percentage_ase(ym, yp, j))
        if determine_percentage_ase(ym, yp, j) < Gthresh:
            needed_eqtls.append((i, j)) 
        elif determine_percentage_heterozygotes_at_locus(xm, xp, i) < Lthresh:
            needed_eqtls.append((i, j))

    # check trans eqtls
    x, y = np.nonzero(xi)
    for i, j in zip(x, y):
        print(determine_percentage_ase(ym, yp, j))
        if determine_percentage_ase(ym, yp, j) < Gthresh:
            print("appending")
            needed_eqtls.append((i, j)) 
        elif determine_percentage_heterozygotes_at_locus(xm, xp, i) < Lthresh:
            print("appending")
            needed_eqtls.append((i, j))

    return list(set(needed_eqtls))


def determine_percentage_ase(ym, yp, loc):
    """
    Determines the percentage of the sample for which ASE is available at a given 
    gene locus. 
    """
    assert all(np.isfinite(ym[:, loc]) == np.isfinite(yp[:, loc])),\
            "Error: maternal and paternal matrices must have the same ASE availability."
    return np.mean(np.isfinite(ym[:, loc]))



def determine_percentage_heterozygotes_at_locus(Xm, Xp, loc):
    """
    Determines the number of heterozygotes at locus loc given 
    genotype expression array Xm and Xp. 
    """
    return np.mean(Xm[:, loc] != Xp[:, loc])


# will need to change!
def to_set_cover(xm, xp, ym, yp, eqtls_needed):
    """
    For each person, gets list of eQTLs for which person has ASE and is heterozygous
    at locus.
    Inputs:
        ym (np.array) - maternal expression matrix 
        yp (np.array) - paternal expression matrix 
        genes_needed (np.array) - the list of needed genes (represented as 0...q-1)
    Outputs:
        people_array (np.array) - i-th element j corresponds to person j
        people_sets (np.array) - i-th element is set for which person j has ASE.
    """
    if len(ym) == 0:
        return [], []
    people_array = [] 
    people_sets = []
    Nr, q = ym.shape
    _, p = xm.shape
    for k in range(Nr):
        person_set = set()
        for i, j in eqtls_needed:
            if xm[k, i] == xp[k, i]:  # person is homozygous
                continue
            if not np.isfinite(ym[k, j]):  # no ASE available for this person
                continue
            else:
                assert np.isfinite(yp[k, j]),\
                        print("Error: maternal and paternal expression " +
                                "matrices must have the same ASE availability")
            person_set.add((i, j))
        if len(person_set) > 0:
            people_sets.append(person_set)
            people_array.append(k)
    return people_array, people_sets
    


if __name__ == '__main__':
    main()
