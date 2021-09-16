[![Apache License][license-shield]][license-url]
[![Google Scholar][gs-shield]][gs-url]
[![PDF][arxiv-shield]][arxiv-url]

# WaveBlocks

* [About the Project](#about)
* [Workflow](#workflow)
* [Current implemented WaveBlocks](#current-implemented-waveblocks)
  * [OpticConfig](#opticconfig)
  * [OpticalBlock](#opticalblock)
  * [WavePropagation](#wavepropagation)
  * [Lens](#lens)
  * [DiffractiveElement](#diffractiveelement)
  * [MicroLensArray](#microlensarray)
  * [Camera](#camera)
* [Examples](#microscope-example)
* [Asumptions](#asumptions)
* [Citing this work](#citing-this-work)
* [Contact](#contact)

## About
A polimorfic optimization framework for Fluorescence Microscopy based on Pytorch, capable of:
* Simulation.
* Calibration.
* PSF engineering.
* Joint optimization of optics and reconstruction, segmentation and arbitrary optimization tasks.
* Plug and play to Pytorch neural networks.

Wave-optics involves complex diffraction integrals that when stacking many optical elements become very hard to derivate and optimize. Here we take advantage of the linearity of the posible operations and modularize them into blocks, enabling building arbitrarly large systems.

By building a WaveBlocks microscope similar to the one in your lab you can first calibrate the unknown parameters, like an accurate PSF, distance between elements, and even errors caused by aberrations and SLM diffraction. Once with the capability of simulating images similar to your microscope you can plug in any Pytorch based network to the imaging system and find and optimize the optimal optical parameters together with your network.

<img src="images/WBMicro_img.jpg">

## Workflow
* Each optical element in a microscope (lenses, propagation through a medium, cameras, PSFs, Spatial Light modulator, appertures, Micro-lens arrays, etc) is represented by a block (Lego like) that can be asembled in any order, each block has as an input/output a complex Wave-Front and processes the input given the wave-optics behavior of the block.

* Most of the variables of the blocks are optimizable, allowing straight forward tasks like calibrating distances between optical elements, correction factors, compute real PSFs, all based on real data. 

* All the blocks are based on a OpticBlock class. Which takes care of gathering the parameters to optimize selected by the user and return them to be fed to the optimizer (Adam, SGD, etc).

* Currently the main objective + tube-lens PSF is imported from an external file. Use Matlab, Fiji or your chosen software to generate the PSF. Its whape should be *[1,nDepths,dim1,dim2,2]* (the last dimension gathers the real and imaginary parts.

## Current implemented Waveblocks
### OpticConfig
Class containing global information about the optical setup, such as wavelength, wave-number and refractive index. It is also used for populating parameters accross the microscope.
    
### OpticalBlock
Base Class of the framework, containing an *OpticConfig* and *members_to_learn*, identifing which Parameters from the current block should be optimized. These are provided by the user.
### WavePropagation
Uses the Rayleight-Sommerfield integral to propagate an incoming complex wave-front a given distance.
### Lens
This class can propagate a wave from the focal plane to the back focal plane or from any point i in front of the lens, to any point o in the back of the lens.
### DiffractiveElement
Class simulating any optical element that modifies either the amplitude or phase of an incoming wave-front, such as: appertures, masks and phase-masks.
### MicroLensArray
This class contains the functionality for space invariant PSF's like in micro lens arrays, used in Light-field microscopes.
### Camera
Computes the intensity of the field hitting the camera and convolves it with an observed object. Works with space variant and invariant PSFs.

## Microscope example
Microscope Class derived from Optical block, simulates the behavior of a microscope
```python
import WaveBlocksPytorch as ob
import torch
import torch.nn as nn
    
class Microscope(ob.OpticBlock):
    def __init__(self, psfIn, vars_to_learn, optical_config):
        super(Microscope, self).__init__(optical_config, vars_to_learn)

        # create parameter for input PSF
        self.psf = nn.Parameter(psfIn, requires_grad=True)
        
        # Fetch pixel size and sampling rate in general
        self.sampling_rate = optical_config.sensor_pitch
        
        # Lateral size of PSF
        field_length = psfIn.shape[2]

        # Create Wave-Propagation block, to defocus the PSF to a given depth
        self.wave_prop = ob.WavePropagation(self.optic_config, [], self.sampling_rate, optical_config.minDefocus, field_length)
        
        # Create Camera Block
        self.camera = ob.Camera(self.optic_config, [], self.sampling_rate)

    def forward(self, realObject):
        # Fetch PSF
        psf = self.psf

        # Defocus PSF given wave_prop.propagation_distance
        defocused_psf = self.wave_prop(psf)
        
        # Compute PSF irradiance and convolve with object
        finalImg,psf = self.camera(realObject, defocused_psf)
        return finalImg
```

## Asumptions
The following design assumptions and functionalites are taken into account:
* All the blocks work with a wave-optics approach of light simulation, appropiate for difracted limited systems like microscopes.
* Light sources are assumed to be incoherent (as in fluorescence microscopy) (coherent is not yet iplemented but will be in the future for lasers and holography).
* The scalar diffraction theory is assumed (paraxial approximation, small angles and low NA)

## Citing this work
```bibtex
@article{pageWaveBlocks2020,
    author = {Page, Josue and Favaro, Paolo},
    booktitle = {IEEE International Conference on Image Processing (ICIP)},
    title = {{L}earning to {M}odel and {C}alibrate {O}ptics via a {D}ifferentiable {W}ave {O}ptics {S}imulator},
    year = {2020}
}
```
## Contact
Josue Page - josue.page@tum.de

[forks-shield]: https://img.shields.io/github/forks/othneildrew/Best-README-Template.svg?style=flat-square
[forks-url]: https://github.com/pvjosue/WaveBlocks/network/members
[stars-shield]: https://img.shields.io/github/stars/othneildrew/Best-README-Template.svg?style=flat-square
[stars-url]: https://github.com/pvjosue/WaveBlocks/stargazers
[issues-shield]: https://img.shields.io/github/issues/othneildrew/Best-README-Template.svg?style=flat-square
[issues-url]: https://github.com/pvjosue/WaveBlocks/issues
[license-shield]: https://img.shields.io/github/license/othneildrew/Best-README-Template.svg?style=flat-square
[license-url]: https://github.com/pvjosue/WaveBlocks/blob/master/LICENSE
[gs-shield]: https://img.shields.io/badge/-GoogleScholar-black.svg?style=flat-square&logo=google-scholar&colorB=555
[gs-url]: https://scholar.google.com/citations?user=5WfCRjQAAAAJ&hl=en
[product-screenshot]: images/screenshot.png
[arxiv-shield]: https://img.shields.io/badge/-PDF-black.svg?style=flat-square&logo=arXiv&colorB=555
[arxiv-url]: https://arxiv.org/abs/2005.08562
