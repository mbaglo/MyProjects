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
prsgseed = [0, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 0, 0, 1, 1]  # PRSG seed: 0x3BB3
sisrn = 16  # Number of bits in the SISR
sisrtaps = np.array([10, 12, 13, 15])  # SISR tap locations
nsteps = 32  # Number of iterations to run the test for

# Sel sequence: 16 cycles at 0 (addition), 16 cycles at 1 (multiplication)
sel_sequence = [0] * 16 + [1] * 16
c_in = 0  # Carry-in for addition

###############################################################################
prsghist = []
prsg = np.ones((prsgn))
for step in range(2**prsgn - 1):
    # Update PRSG
    if prsgstyle == 0:
        prsg = np.concatenate(([(np.sum(prsg[prsgtaps]) + prsg[-1]) % 2], prsg[:-1]))
    else:
        # Right-shifting LFSR: MSB (bit 15) is at prsg[15]
        # Compute feedback as XOR of tapped bits (positions 15, 13, 12, 10)
        tapped_bits = [prsg[t] for t in prsgtaps]
        feedback = 0
        for bit in tapped_bits:
            feedback = (feedback + bit) % 2
        # Apply feedback to positions excluding the MSB (bit 15)
        tap_mask = np.array([int(np.isin(x, [t for t in prsgtaps if t != 15])) for x in range(prsgn - 1)])
        prsg = np.concatenate(
            (
                (prsg[1:] + feedback * tap_mask) % 2,
                [feedback]
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
        # Right-shifting SISR: MSB (bit 15) is at sisr[15]
        # Compute feedback as XOR of tapped bits (positions 15, 13, 12, 10)
        tapped_bits = [sisr[t] for t in sisrtaps]
        feedback = 0
        for bit in tapped_bits:
            feedback = (feedback + bit) % 2
        # Apply feedback to positions excluding the MSB (bit 15)
        tap_mask = np.array([int(np.isin(x, [t for t in sisrtaps if t != 15])) for x in range(sisrn - 1)])
        sisr = np.concatenate(
            (
                (sisr[1:] + feedback * tap_mask) % 2,
                [feedback]
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
    # DEFINE YOUR CIRCUIT HERE -- MODIFIED FOR cgm_ALU_BIST
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

    # ALU operation based on Sel
    sel = sel_sequence[step]
    if sel == 0:  # Addition
        multibit_output = in1 + in2 + c_in
    else:  # Multiplication
        multibit_output = in1 * in2

    # Ensure 16-bit output
    multibit_output = multibit_output & 0xFFFF  # Mask to 16 bits

    # Compute SISR input: XOR of all 16 bits of ALU output (S0-S15)
    bits = [(multibit_output >> i) & 1 for i in range(16)]
    single_bit_output = 0
    for bit in bits:
        single_bit_output = (single_bit_output + bit) % 2  # XOR all bits
    ############################################################################

    # Input to SISR
    if sisrstyle == 0:
        sisr = np.concatenate(
            ([(single_bit_output + np.sum(sisr[sisrtaps]) + sisr[-1]) % 2], sisr[:-1])
        )
    else:
        # Right-shifting SISR: MSB (bit 15) is at sisr[15]
        # Compute feedback as XOR of tapped bits (positions 15, 13, 12, 10)
        tapped_bits = [sisr[t] for t in sisrtaps]
        feedback = 0
        for bit in tapped_bits:
            feedback = (feedback + bit) % 2
        # XOR feedback with the input bit to get the new LSB (bit 0, sisr[0])
        new_lsb = (feedback + single_bit_output) % 2
        # Apply feedback to positions excluding the MSB (bit 15)
        tap_mask = np.array([int(np.isin(x, [t for t in sisrtaps if t != 15])) for x in range(sisrn - 1)])
        sisr = np.concatenate(
            (
                (sisr[1:] + feedback * tap_mask) % 2,
                [feedback]
            )
        )
        # Update the new LSB (bit 0) after the shift
        sisr[0] = new_lsb

    # Update PRSG
    if prsgstyle == 0:
        prsg = np.concatenate(([(np.sum(prsg[prsgtaps]) + prsg[-1]) % 2], prsg[:-1]))
    else:
        # Right-shifting LFSR: MSB (bit 15) is at prsg[15]
        # Compute feedback as XOR of tapped bits (positions 15, 13, 12, 10)
        tapped_bits = [prsg[t] for t in prsgtaps]
        feedback = 0
        for bit in tapped_bits:
            feedback = (feedback + bit) % 2
        # Apply feedback to positions excluding the MSB (bit 15)
        tap_mask = np.array([int(np.isin(x, [t for t in prsgtaps if t != 15])) for x in range(prsgn - 1)])
        prsg = np.concatenate(
            (
                (prsg[1:] + feedback * tap_mask) % 2,
                [feedback]
            )
        )
        # Update the new LSB (bit 0) with the feedback
        prsg[0] = feedback

    history.append(prsg)

# Convert SISR to a binary string and hex
sisr_binary = ''.join(map(str, sisr))
sisr_decimal = int(sisr_binary, 2)
sisr_hex = hex(sisr_decimal)[2:].zfill(4)  # Remove '0x' prefix, pad to 4 characters

print("Golden Signature (binary): " + sisr_binary)
print("Golden Signature (hex): " + sisr_hex)