# -*- coding: utf-8 -*-
"""
Script for determining the golden signature for cgm_ALU_BIST

@author: Cory Merkel (Alterations by Michael Zuzak, further modified by Grok)
"""

import numpy as np

# Configuration for LFSR and SISR
prsgstyle = 1  # 1 for Galois (as used in your LFSR)
sisrstyle = 1  # 1 for Galois (as used in your SISR)
prsgn = 16  # Number of bits in the LFSR (16-bit)
prsgtaps = np.array(
    [10, 12, 13, 15]
)  # LFSR taps: x^16 + x^14 + x^13 + x^11 + 1 (0-based indexing)
prsgseed = [
    0,
    0,
    1,
    1,
    1,
    0,
    1,
    1,
    1,
    0,
    1,
    1,
    0,
    0,
    1,
    1,
]  # Seed: 0x3BB3 = 00111011_10110011
sisrn = 16  # Number of bits in the SISR (16-bit)
sisrtaps = np.array(
    [10, 12, 13, 15]
)  # SISR taps: x^16 + x^14 + x^13 + x^11 + 1 (0-based indexing)
nsteps = 32  # Number of test cycles (32 cycles)

# Sel sequence: 16 cycles at 0 (addition), 16 cycles at 1 (multiplication)
sel_sequence = [0] * 16 + [1] * 16
c_in = 0  # Carry-in is 0 for all cycles

###############################################################################
# Verify LFSR (PRSG) configuration
prsghist = []
prsg = np.ones((prsgn), dtype=int)
for step in range(2**prsgn - 1):
    # Update PRSG (LFSR)
    if prsgstyle == 0:
        prsg = np.concatenate(([(np.sum(prsg[prsgtaps]) + prsg[-1]) % 2], prsg[:-1]))
    else:
        # Left-shifting Galois LFSR: MSB (bit 15) is at prsg[15]
        # Compute feedback as XOR of tapped bits (positions 15, 13, 12, 10)
        feedback = 0
        for tap in prsgtaps:
            feedback = (feedback + prsg[tap]) % 2
        # Shift left: bits 0 to 14 move to positions 1 to 15
        new_prsg = np.zeros(prsgn, dtype=int)
        new_prsg[1:] = prsg[:-1]
        # Apply feedback bit (prsg[15]) to tapped positions (excluding 15)
        for tap in prsgtaps:
            if tap != 15:
                new_prsg[tap] = (new_prsg[tap] + prsg[-1]) % 2
        # Set new LSB (bit 0) to feedback
        new_prsg[0] = feedback
        prsg = new_prsg
    prsghist.append(prsg.copy())

if np.shape(np.unique(np.array(prsghist), axis=0))[0] != 2**prsgn - 1:
    print("You do not have a maximal length PRSG tap configuration!")
else:
    print("PRSG tap configuration is maximal length.")

# Verify SISR configuration
sisrhist = []
sisr = np.ones((sisrn), dtype=int)
for step in range(2**sisrn - 1):
    # Update SISR
    if sisrstyle == 0:
        sisr = np.concatenate(([(np.sum(sisr[sisrtaps]) + sisr[-1]) % 2], sisr[:-1]))
    else:
        # Left-shifting Galois SISR: MSB (bit 15) is at sisr[15]
        # Compute feedback as XOR of tapped bits (positions 15, 13, 12, 10)
        feedback = 0
        for tap in sisrtaps:
            feedback = (feedback + sisr[tap]) % 2
        # Shift left: bits 0 to 14 move to positions 1 to 15
        new_sisr = np.zeros(sisrn, dtype=int)
        new_sisr[1:] = sisr[:-1]
        # Apply feedback bit (sisr[15]) to tapped positions (excluding 15)
        for tap in sisrtaps:
            if tap != 15:
                new_sisr[tap] = (new_sisr[tap] + sisr[-1]) % 2
        # Set new LSB (bit 0) to feedback (will be updated in main loop with input bit)
        new_sisr[0] = feedback
        sisr = new_sisr
    sisrhist.append(sisr.copy())

if np.shape(np.unique(np.array(sisrhist), axis=0))[0] != 2**sisrn - 1:
    print("You do not have a maximal length SISR tap configuration!")
else:
    print("SISR tap configuration is maximal length.")

###############################################################################
# Golden Signature Calculation for cgm_ALU_BIST
###############################################################################
prsg = np.array(prsgseed, dtype=int)  # Initialize LFSR with seed
sisr = np.zeros((sisrn), dtype=int)  # Initialize SISR with all zeros
history = []

for step in range(nsteps):
    ###########################################################################
    # Apply current PRSG value to the ALU
    ###########################################################################
    # Convert PRSG into two 8-bit unsigned inputs (A0-A7, B0-B7)
    in1 = 0  # A0-A7 (bits 15:8, prsg[8:16])
    for ele in prsg[8:16]:
        in1 = (in1 << 1) | int(ele)

    in2 = 0  # B0-B7 (bits 7:0, prsg[0:8])
    for ele in prsg[0:8]:
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

    ###########################################################################
    # Update SISR with the single-bit output
    if sisrstyle == 0:
        sisr = np.concatenate(
            ([(single_bit_output + np.sum(sisr[sisrtaps]) + sisr[-1]) % 2], sisr[:-1])
        )
    else:
        # Left-shifting Galois SISR: MSB (bit 15) is at sisr[15]
        # Compute feedback as XOR of tapped bits (positions 15, 13, 12, 10)
        feedback = 0
        for tap in sisrtaps:
            feedback = (feedback + sisr[tap]) % 2
        # XOR feedback with the input bit to get the new LSB (bit 0)
        new_lsb = (feedback + single_bit_output) % 2
        # Shift left: bits 0 to 14 move to positions 1 to 15
        new_sisr = np.zeros(sisrn, dtype=int)
        new_sisr[1:] = sisr[:-1]
        # Apply feedback bit (sisr[15]) to tapped positions (excluding 15)
        for tap in sisrtaps:
            if tap != 15:
                new_sisr[tap] = (new_sisr[tap] + sisr[-1]) % 2
        # Set new LSB (bit 0)
        new_sisr[0] = new_lsb
        sisr = new_sisr

    # Update PRSG (LFSR)
    if prsgstyle == 0:
        prsg = np.concatenate(([(np.sum(prsg[prsgtaps]) + prsg[-1]) % 2], prsg[:-1]))
    else:
        # Left-shifting Galois LFSR: MSB (bit 15) is at prsg[15]
        # Compute feedback as XOR of tapped bits (positions 15, 13, 12, 10)
        feedback = 0
        for tap in prsgtaps:
            feedback = (feedback + prsg[tap]) % 2
        # Shift left: bits 0 to 14 move to positions 1 to 15
        new_prsg = np.zeros(prsgn, dtype=int)
        new_prsg[1:] = prsg[:-1]
        # Apply feedback bit (prsg[15]) to tapped positions (excluding 15)
        for tap in prsgtaps:
            if tap != 15:
                new_prsg[tap] = (new_prsg[tap] + prsg[-1]) % 2
        # Set new LSB (bit 0) to feedback
        new_prsg[0] = feedback
        prsg = new_prsg

    history.append(prsg.copy())

# Convert SISR to a binary string and hex
sisr_binary = "".join(map(str, sisr))
sisr_decimal = int(sisr_binary, 2)
sisr_hex = hex(sisr_decimal)[2:].zfill(4)  # Remove '0x' prefix, pad to 4 characters

print("Golden Signature (binary): " + sisr_binary)
print("Golden Signature (hex): " + sisr_hex)