# -*- coding: utf-8 -*-
"""
Script for determining the golden signature for cgm_ALU_BIST

@author: Cory Merkel (Alterations by Michael Zuzak, further modified by Grok)
"""

import numpy as np

# Configuration for LFSR and SISR
prsgstyle = 1  # 0 for Fibonacci, 1 (or not 0) for Galois
sisrstyle = 1  # 0 for Fibonacci, 1 (or not 0) for Galois
prsgn = 16  # Number of bits in the PRSG (LFSR)
prsgtaps = np.array(
    [10, 12, 13, 15]
)  # PRSG tap locations: x^16 + x^14 + x^13 + x^11 + 1 (positions 15, 13, 12, 10)
prsgseed = [0, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 0, 0, 1, 1]  # PRSG seed: 0x3BB3
sisrn = 16  # Number of bits in the SISR
sisrtaps = np.array(
    [10, 12, 13, 15]
)  # SISR tap locations: x^16 + x^14 + x^13 + x^11 + 1 (positions 15, 13, 12, 10)
nsteps = 32  # Number of iterations to run the test for

# Sel sequence: 16 cycles at 0 (addition), 16 cycles at 1 (multiplication)
sel_sequence = [0] * 16 + [1] * 16
c_in = 0  # Carry-in for addition

###############################################################################
prsghist = []
prsg = np.ones((prsgn))
for step in range(2**prsgn - 1):
    # Update PRSG (LFSR)
    if prsgstyle == 0:
        prsg = np.concatenate(([(np.sum(prsg[prsgtaps]) + prsg[-1]) % 2], prsg[:-1]))
    else:
        # Feedback bit is the MSB (position 15)
        feedback = prsg[-1] if 15 in prsgtaps else 0
        # Apply feedback to positions excluding the feedback bit
        tap_mask = np.array(
            [
                int(np.isin(x, [t for t in prsgtaps if t != 15]))
                for x in range(prsgn - 1)
            ]
        )
        prsg = np.concatenate(([feedback], (prsg[:-1] + feedback * tap_mask) % 2))
    prsghist.append(prsg)

if np.shape(np.unique(np.array(prsghist), axis=0))[0] != 2**prsgn - 1:
    print("You do not have a maximal length PRSG tap configuration!")
else:
    print("PRSG tap configuration is maximal length.")

sisrhist = []
sisr = np.ones((sisrn))
for step in range(2**sisrn - 1):
    # Update SISR
    if sisrstyle == 0:
        sisr = np.concatenate(([(np.sum(sisr[sisrtaps]) + sisr[-1]) % 2], sisr[:-1]))
    else:
        # Feedback bit is the MSB (position 15), left-shifting like LFSR
        feedback = sisr[-1] if 15 in sisrtaps else 0
        # Apply feedback to positions excluding the feedback bit
        tap_mask = np.array(
            [
                int(np.isin(x, [t for t in sisrtaps if t != 15]))
                for x in range(sisrn - 1)
            ]
        )
        sisr = np.concatenate(([feedback], (sisr[:-1] + feedback * tap_mask) % 2))
    sisrhist.append(sisr)

if np.shape(np.unique(np.array(sisrhist), axis=0))[0] != 2**sisrn - 1:
    print("You do not have a maximal length SISR tap configuration!")
else:
    print("SISR tap configuration is maximal length.")

###############################################################################
# Golden Signature Calculation
#     If you change your architecture/function, you will have to change this!
###############################################################################
prsg = np.array(prsgseed, dtype=int)  # Initialize PRSG with seed
sisr = np.zeros((sisrn), dtype=int)  # Initialize SISR with all zeros
history = []
for step in range(nsteps):
    ###########################################################################
    # Apply current PRSG value to the circuit
    # DEFINE YOUR CIRCUIT HERE -- MODIFIED FOR cgm_ALU_BIST
    ###########################################################################
    # Convert PRSG into two 8-bit unsigned inputs (A0-A7, B0-B7)
    in1 = 0  # A0-A7 (bits 15:8)
    for ele in prsg[: len(prsg) // 2]:
        in1 = (in1 << 1) | int(ele)

    in2 = 0  # B0-B7 (bits 7:0)
    for ele in prsg[len(prsg) // 2 :]:
        in2 = (in2 << 1) | int(ele)

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
        single_bit_output ^= bit  # XOR all bits
    ############################################################################

    # Input to SISR
    if sisrstyle == 0:
        sisr = np.concatenate(
            ([(single_bit_output + np.sum(sisr[sisrtaps]) + sisr[-1]) % 2], sisr[:-1])
        )
    else:
        # Feedback bit is the MSB (position 15), XOR with input bit
        feedback = (single_bit_output + (sisr[-1] if 15 in sisrtaps else 0)) % 2
        # Apply feedback to positions excluding the feedback bit
        tap_mask = np.array(
            [
                int(np.isin(x, [t for t in sisrtaps if t != 15]))
                for x in range(sisrn - 1)
            ]
        )
        sisr = np.concatenate(([feedback], (sisr[:-1] + feedback * tap_mask) % 2))

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
                    * np.array(
                        [
                            int(np.isin(x, [t for t in prsgtaps if t != 15]))
                            for x in range(prsgn - 1)
                        ]
                    )
                )
                % 2,
            )
        )

    history.append(prsg)

# Convert SISR to a binary string and hex
sisr_binary = "".join(map(str, sisr))
sisr_decimal = int(sisr_binary, 2)
sisr_hex = hex(sisr_decimal)[2:].zfill(4)  # Remove '0x' prefix, pad to 4 characters

print("Golden Signature (binary): " + sisr_binary)
print("Golden Signature (hex): " + sisr_hex)
