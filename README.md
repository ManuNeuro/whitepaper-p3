# White-paper: Hash Function with Chaotic Artificial Neural networks (Part 3)

This code accompany a blog [article](https://manuneuro.github.io/EmmanuelCalvet//quantum,/crypto/2024/04/01/whiepaper-p3.html) published on my website, belonging to the serie: "White-paper: IA and crypto". For a more thorough introduction and explanation of the context, [check it out](https://manuneuro.github.io/EmmanuelCalvet//quantum,/crypto/2022/09/01/whitepaper-p1.html)!

# Installation

To run this code, only standard library like `numpy`, `matplotlib` and `time` are needed, at the exception of the BiEntropy library.

# Content

This code contains three experiments, testing an ANN with various weights matrices initialization:

- Confusion tests, showing the phase transition using the BiEntropy.
- Diffusion testing showing the hamming distance between input and output.
- Diffusion testing showing the hamming distance between outputs when bits are flipped in the input.
