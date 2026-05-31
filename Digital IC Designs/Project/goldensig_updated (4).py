# -*- coding: utf-8 -*-
"""
Script for determining the golden signature for CMPE 630 ALU BIST project

@author: Cory Merkel (Alterations by Michael Zuzak, further modified by Grok)
"""

import numpy as np

# Configuration for LFSR and SISR (matching project setup)
prsgstyle = 1  # Galois LFSR
sisrstyle = 0  # Fibonacci SISR
prsgn = 16     # Number of bits in the PRSG (LFSR)
prsgtaps = np.array([10, 12, 13, 15])  # PRSG tap locations (bits 15, 13, 12, 10)
prsgseed = [0, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 0, 0, 1, 1]  # PRSG seed (0x3BB3)
sisrn = 16     # Number of bits in the SISR
sisrtaps = np.array([10, 12, 13, 15])  # SISR tap locations (same as PRSG)
nsteps = 32    # Number of BIST cycles (matching testbench)

# Verify PRSG and SISR configurations for maximal length
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

# Golden Signature Calculation
prsg = prsgseed
sisr = np.zeros((sisrn))  # SISR initialized to all zeros
history = []
for step in range(nsteps):
    # Apply current PRSG value to the circuit (ALU with BIST)
    # Convert PRSG into two 8-bit unsigned inputs (A0-A7, B0-B7)
    in1 = 0  # A0-A7 (PRSG bits 0-7)
    for ele in prsg[: len(prsg) // 2]:
        in1 = (in1 << 1) | int(ele)

    in2 = 0  # B0-B7 (PRSG bits 8-15)
    for ele in prsg[len(prsg) // 2 :]:
        in2 = (in2 << 1) | int(ele)

    # Calculate addition (ALU with Sel = 0)
    multibit_output = in1 + in2  # 16-bit sum (S0-S15)

    # Convert the 16-bit sum to binary and extract bits
    sum_bits = [int(bit) for bit in format(multibit_output & 0xFFFF, '016b')]  # Ensure 16 bits

    # Reduce S0-S15 to a single bit using a NAND tree (29 NAND gates)
    bits = sum_bits
    # Layer 1: 15 NAND gates + 1 inverter
    layer1 = []
    for i in range(15, 1, -1):
        nand_result = 1 - (bits[i] & bits[i - 1])  # NAND: 1 - (A & B)
        layer1.append(nand_result)
    inv_s0 = 1 - bits[0]  # NOT S0
    n15 = 1 - (inv_s0 & bits[1])  # NAND: 1 - (NOT S0 & S1)
    layer1.append(n15)

    # Layer 2: 7 NAND gates (pair N1-N2, N3-N4, ..., N13-N14)
    layer2 = []
    for i in range(0, 14, 2):
        nand_result = 1 - (layer1[i] & layer1[i + 1])
        layer2.append(nand_result)

    # Layer 3: 4 NAND gates (L2_1-L2_2, L2_3-L2_4, L2_5-L2_6, L2_7-N15)
    layer3 = []
    for i in range(0, 6, 2):
        nand_result = 1 - (layer2[i] & layer2[i + 1])
        layer3.append(nand_result)
    l3_4 = 1 - (layer2[6] & layer1[14])
    layer3.append(l3_4)

    # Layer 4: 2 NAND gates
    layer4 = []
    for i in range(0, 4, 2):
        nand_result = 1 - (layer3[i] & layer3[i + 1])
        layer4.append(nand_result)

    # Layer 5: 1 NAND gate
    single_bit_output = 1 - (layer4[0] & layer4[1])

    # Input to SISR (Fibonacci configuration, shift right D0 to D15)
    if sisrstyle == 0:
        # Compute feedback: XOR of D15 with tapped bits
        feedback = np.sum(sisr[sisrtaps]) % 2  # XOR of tapped bits
        feedback = (feedback + sisr[0]) % 2    # XOR with D15 (sisr[0])
        # New bit: XOR the input with the feedback
        new_bit = (single_bit_output + feedback) % 2
        # Shift right (D0 to D15) and insert new bit at D0
        sisr = np.concatenate(([new_bit], sisr[:-1]))
    else:
        feedback = sisr[-1]
        new_sisr = np.concatenate(([single_bit_output], sisr[:-1]))
        tap_mask = np.zeros(sisrn, dtype=int)
        for tap in sisrtaps:
            tap_mask[tap] = 1
        for i in range(sisrn):
            if tap_mask[i] == 1:
                new_sisr[i] = (new_sisr[i] + feedback) % 2
        sisr = new_sisr

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

# Convert SISR state to hex (matching testbench output)
signature_int = 0
for bit in sisr:
    signature_int = (signature_int << 1) | int(bit)
signature_hex = format(signature_int, '04x')  # 4 hex digits for 16 bits

print("Golden Signature: " + signature_hex)