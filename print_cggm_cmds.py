######################################################################
# print_cggm_cmds.py
# prints commands for cggm output from simulations.
######################################################################

# for timing 
import time

import sys
import subprocess

GENERAL_PREFIX = "/mnt/c/Users/apare/Desktop/KimResearchGroup/Spring2022/"
TO_CITRUSS = GENERAL_PREFIX + "mlcggm/Mega-sCGGM_cpp/citruss"
TO_DATA = GENERAL_PREFIX + "input_simulation/simulateCode2/missing"

N = 2000
Q = 150
P = 400

MISSING_RATIOS = [35] # [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
NSIMS = 1 # NSIMS = 10


def main():
    fname_out = "cmds.sh"

    reg_v = 0.01
    reg_f = 0.01
    reg_gamma = 0.01
    reg_psi = 0.01

    for ratio in MISSING_RATIOS:
        for i in range(1, NSIMS+1):
            ysum_fname = TO_DATA + "{}/Ysum{}.txt".format(ratio, i)
            ym_fname = TO_DATA + "{}/Ym{}.txt".format(ratio, i)
            yp_fname = TO_DATA + "{}/Yp{}.txt".format(ratio, i)
            xm_fname = TO_DATA + "{}/Xm{}.txt".format(ratio, i)
            xp_fname = TO_DATA + "{}/Xp{}.txt".format(ratio, i)
            output_fname = TO_DATA + "{}/{}".format(ratio, i)
            commands = [TO_CITRUSS,
                        str(N), str(Q), str(P),
                        ysum_fname,
                        ym_fname,
                        yp_fname,
                        xm_fname,
                        xp_fname,
                        str(reg_v), str(reg_f), str(reg_gamma), str(reg_psi),
                        '-o', output_fname]
            subprocess.run(commands, check=True)


if __name__ == '__main__':
    start_main = time.time()
    main()
    print("Program ran in {:.3f} seconds".format(time.time() - start_main), 
          file=sys.stderr)
