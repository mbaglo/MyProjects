import numpy as np

# Configuration
prsgstyle = 1
sisrstyle = 1
prsgn = 16
sisrn = 16
nsteps = 32  # Required by your project

prsgtaps = np.array([10, 12, 13, 15])
sisrtaps = np.array([10, 12, 13, 15])
prsgseed = [1, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]  # 10110011 + 00111011

# Initialize PRSG and SISR
prsg = np.array(prsgseed, dtype=int)
sisr = np.zeros((sisrn,), dtype=int)

# Select line (0 for ADD, 1 for MUL)
Sel = 0  # change to 1 if testing multiplication

# BIST loop
for step in range(nsteps):
    # Split PRSG bits into A and B
    A = 0
    B = 0
    for bit in prsg[:8]:
        A = (A << 1) | bit
    for bit in prsg[8:]:
        B = (B << 1) | bit

    # ALU operation
    alu_out = A * B if Sel else A + B  # match Verilog behavior

    # Reduce to 1-bit output for SISR
    sisr_input = bin(alu_out).count("1") % 2

    # SISR update
    sisr_feedback = sisr[15] ^ sisr[13] ^ sisr[12] ^ sisr[10]
    new_bit = sisr_feedback ^ sisr_input
    sisr = np.concatenate(([new_bit], sisr[:-1]))

    # PRSG update
    prsg_feedback = prsg[15] ^ prsg[13] ^ prsg[12] ^ prsg[10]
    prsg = np.concatenate(([prsg_feedback], prsg[:-1]))

# Final output
golden_signature = "".join(map(str, sisr))
print("Golden Signature:", golden_signature)
