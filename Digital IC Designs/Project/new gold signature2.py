import numpy as np

# Settings
prsg_taps = [10, 12, 13, 15]
sisr_taps = [10, 12, 13, 15]
nsteps = 32

# Initial seed from Verilog (A = 10110011, B = 00111011)
prsg = np.array(
    [
        1,
        0,
        1,
        1,
        0,
        0,
        1,
        1,  # A = 10110011
        0,
        0,
        1,
        1,
        1,
        0,
        1,
        1,  # B = 00111011
    ],
    dtype=int,
)

sisr = np.zeros(16, dtype=int)


# Function to update PRSG (Galois style)
def update_prsg(prsg):
    feedback = prsg[15]
    for t in prsg_taps:
        if t != 15:
            prsg[t] ^= feedback
    return np.roll(prsg, 1)


# Function to update SISR (Galois style)
def update_sisr(sisr, input_bit):
    feedback = sisr[15] ^ input_bit
    for t in sisr_taps:
        if t != 15:
            sisr[t] ^= feedback
    return np.roll(sisr, 1)


# Golden signature with ADD only
prsg_add = prsg.copy()
sisr_add = sisr.copy()

for step in range(nsteps):
    A = int("".join(str(x) for x in prsg_add[:8]), 2)
    B = int("".join(str(x) for x in prsg_add[8:]), 2)
    alu_out = A + B
    bit_out = bin(alu_out).count("1") % 2
    sisr_add = update_sisr(sisr_add, bit_out)
    prsg_add = update_prsg(prsg_add)

sig_add = "".join(str(x) for x in sisr_add)
hex_add = f"{int(sig_add, 2):04X}"

# Golden signature with ADD (16) + MUL (16)
prsg_mix = prsg.copy()
sisr_mix = sisr.copy()

for step in range(nsteps):
    A = int("".join(str(x) for x in prsg_mix[:8]), 2)
    B = int("".join(str(x) for x in prsg_mix[8:]), 2)
    if step < 16:
        alu_out = A + B
    else:
        alu_out = A * B
    bit_out = bin(alu_out).count("1") % 2
    sisr_mix = update_sisr(sisr_mix, bit_out)
    prsg_mix = update_prsg(prsg_mix)

sig_mix = "".join(str(x) for x in sisr_mix)
hex_mix = f"{int(sig_mix, 2):04X}"

(hex_add, hex_mix)
