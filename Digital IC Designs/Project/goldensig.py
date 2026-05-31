# -*- coding: utf-8 -*-
"""
Script for determining the Perceptron golden signature

@author: Cory Merkel (Alterations by Michael Zuzak)
"""

import numpy as np

prsgstyle = 1  # 0 for Fiboniacci, 1 (or not 0) for Galois
sisrstyle = 1  # 0 for Fiboniacci, 1 (or not 0) for Galois
prsgn = 16  # Number of bits in the PRSG
prsgtaps = np.array([10, 12, 13, 15])  # PRSG tap locations
prsgseed = [1, 0, 1, 1, 0, 0, 1, 1, 0]  # PRSG seed
sisrn = 16  # Number of bits in the SISR
sisrtaps = np.array([10, 12, 13, 15])  # SISR tap locations
nsteps = 1000  # Number of iterations to run the test for


###############################################################################
prsghist = []
prsg = np.ones((prsgn))
for step in range(2**prsgn - 1):
    # Update PRSG
    if prsgstyle == 0:
        prsg = np.concatenate(([(np.sum(prsg[prsgtaps]) + prsg[-1]) % 2], prsg[:-1]))
    else:
        prsg = np.concatenate(
            (
                [prsg[-1]],
                (
                    prsg[:-1]
                    + prsg[-1]
                    * np.array([int(np.isin(x, prsgtaps)) for x in range(prsgn - 1)])
                )
                % 2,
            )
        )
    prsghist.append(prsg)

if np.shape(np.unique(np.array(prsghist), axis=0))[0] != 2**prsgn - 1:
    print("You do not have a maximal length PRSG tap configuration!")

sisrhist = []
sisr = np.ones((sisrn))
for step in range(2**sisrn - 1):
    # Update SISR
    if sisrstyle == 0:
        sisr = np.concatenate(([(np.sum(sisr[sisrtaps]) + sisr[-1]) % 2], sisr[:-1]))
    else:
        sisr = np.concatenate(
            (
                [sisr[-1]],
                (
                    sisr[:-1]
                    + sisr[-1]
                    * np.array([int(np.isin(x, sisrtaps)) for x in range(sisrn - 1)])
                )
                % 2,
            )
        )
    sisrhist.append(sisr)

if np.shape(np.unique(np.array(sisrhist), axis=0))[0] != 2**sisrn - 1:
    print("You do not have a maximal length SISR tap configuration!")


###############################################################################
# Golden Signature Calculation
#     If you change your architecture/function, you will have to change this!
###############################################################################
prsg = prsgseed
sisr = np.zeros((sisrn))
history = []
for step in range(nsteps):
    ###########################################################################
    # Apply current PRSG value to the circuit
    # DEFINE YOUR CIRCUIT HERE -- EXAMPLE CODE FOR ADDER
    ###########################################################################
    # Convert prsg into 2 UNSIGNED inputs
    # Input 1
    in1 = 0
    for ele in prsg[: len(prsg) // 2]:
        in1 = (in1 << 1) | ele

    # Input 2
    in2 = 0
    for ele in prsg[len(prsg) // 2 :]:
        in2 = (in2 << 1) | ele

    # Calculate addition
    multibit_output = in1 + in2

    # This is an SISR so the circuit output must be 1 or 0, no multibit answers.
    # Here I have the case where determine whether there are an even or odd amount of 1's in the output
    single_bit_output = bin(multibit_output).count("1") % 2
    ############################################################################

    # Input to SISR
    if sisrstyle == 0:
        sisr = np.concatenate(
            ([(single_bit_output + np.sum(sisr[sisrtaps]) + sisr[-1]) % 2], sisr[:-1])
        )
    else:
        sisr = np.concatenate(
            (
                [(single_bit_output + sisr[-1]) % 2],
                (
                    sisr[:-1]
                    + sisr[-1]
                    * np.array([int(np.isin(x, sisrtaps)) for x in range(sisrn - 1)])
                )
                % 2,
            )
        )

    # Update PRSG
    if prsgstyle == 0:
        prsg = np.concatenate(([(np.sum(prsg[prsgtaps]) + prsg[-1]) % 2], prsg[:-1]))
    else:
        prsg = np.concatenate(
            (
                [prsg[-1]],
                (
                    prsg[:-1]
                    + prsg[-1]
                    * np.array([int(np.isin(x, prsgtaps)) for x in range(prsgn - 1)])
                )
                % 2,
            )
        )

    history.append(prsg)

print("Golden Signature: " + str(sisr))
