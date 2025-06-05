# 🧠 NeuroVoc: From Spikes to Speech 🔊

[![arXiv](https://img.shields.io/badge/arXiv-Preprint-orange)](https://arxiv.org/abs/XXXX.XXXXX)  
*A biologically plausible vocoder for auditory perception modeling and cochlear implant simulation.*

---

## 🔍 Overview

**NeuroVoc** is a flexible, biologically inspired vocoder that reconstructs audio signals from simulated auditory nerve activity. It is designed to support both **normal hearing (NH)** and **electrical hearing (EH)** models, allowing for a seamless comparison of auditory perception under different hearing conditions.

![NeuroVoc Diagram](diagram.png)

### 🧭 Diagram Explanation

The diagram above illustrates the NeuroVoc processing pipeline:

1. **Sound** — An input waveform (e.g., speech) is passed to an auditory model.
2. **Hearing Model** — This model (e.g., normal hearing or cochlear implant simulation) transforms the sound into a neural representation.
3. **Neurogram** — The output is a time–frequency matrix of spike counts, simulating auditory nerve activity.
4. **Decoder** — The neurogram is then converted back into an acoustic waveform using an inverse short-time Fourier transform (STFT)- based decoder.

This modular flow enables the flexible substitution of different models or model parameters while maintaining a consistent reconstruction backend.

---

This repository contains:

- The **NeuroVoc Python package** (modular vocoder)
- All code for the **experiments and figures** from the paper
- Data pipeline tools for working with **neurogram representations**
---

## 📁 Repository Structure

```
neurovoc/
├── neurovoc/               # Core vocoder framework (Python package)
├── experiments/            # Scripts for model runs and evaluations
├── data/                   # Example data neurogram data and metadata
├── figures/                # Diagrams and visualizations
├── requirements.txt        # Python dependencies
└── README.md               # You're here
```

---

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/jacobdenobel/neurovoc.git
cd neurovoc
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run a reconstruction example

```bash
python experiments/reconstruct_from_neurogram.py --input data/example_neurogram.npz
```

This will generate a `.wav` file from a simulated neurogram. You can experiment with different models by swapping in neurogram outputs from alternative auditory simulations. 

---


## 🧠 Citation

If you use NeuroVoc in your work, please cite the following:

```bibtex
@misc{denobel2025spikesspeechneurovoc,
    title={From Spikes to Speech: NeuroVoc -- A Biologically Plausible Vocoder Framework for Auditory Perception and Cochlear Implant Simulation}, 
    author={Jacob de Nobel and Jeroen J. Briaire and Thomas H. W. Baeck and Anna V. Kononova and Johan H. M. Frijns},
    year={2025},
    eprint={2506.03959},
    archivePrefix={arXiv},
    primaryClass={cs.SD},
    url={https://arxiv.org/abs/2506.03959}, 
}
```

---

## 📫 Contact

For questions or feedback, contact [nobeljpde1@liacs.leidenuniv.nl](mailto:nobeljpde1@liacs.leidenuniv.nl)  
Or open an issue in this repository.

---

## 🛠 License

This project is licensed under the **MIT License**. See the [LICENSE](LICENSE) file for details.
