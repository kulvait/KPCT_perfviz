# KPCT perfviz

Creating perfusion maps from perfusion CT (PCT) acquisitions or flat detector perfusion CT (FDPCT) acquisitions. FDCT is often also called C-arm CT or Dyna-CT.

These maps are usually the main visual output of perfusion imaging for the physician.

Exporting volume series of time attenuation curves or time resolved CT volumes based on the underlying model is also supported.

## Created perfusion maps

Specifically, maps of cerebral blood flow CBF, cerebral blood volume CBV, mean transit time MTT and time to peak TTP are created. If not brain perfusion, the abbreviations BF, BV, MTT and TTP are used for these parameters.

Deconvolution based model for computing these parameters is described in the paper [Time separation technique with the basis of trigonometric functions as an efficient method for flat detector CT brain perfusion imaging](https://arxiv.org/abs/2110.09438). It is derived from the model in [Fieselmann et.al. 2011](http://dx.doi.org/10.1155/2011/467563) but the CBF computation is modified by so called CBF time parameter. Using this parameter and setting it to 6s in accordance to [DEFUSE3 study protocol](https://clinicaltrials.gov/ProvidedDocs/15/NCT02586415/Prot_001.pdf) shall make MTT a better indicator of colateral flow.

## Dimension reduction models

For CT data, the original data of the reconstructed volumes can be processed. However, it is also possible to use data reduction models. For FDCT this is specifically the TST model as described in the paper [Time separation technique with the basis of trigonometric functions as an efficient method for flat detector CT brain perfusion imaging](https://arxiv.org/abs/2110.09438). 

Different bases are implemented in this model. Specifically, these are two orthogonal bases of real polynomials, the Chebyshev and Legendre bases. Then the basis of the first few functions forming the Hilbert space L2 is implemented, namely the trigonometric functions in the first terms of the trigonometric polynomial used in the Fourier decomposition. Furthermore, any basis with a given discretization can be used. For the method to work properly, it is necessary to ensure that the basis is orthogonal as in the case of SVD decomposition vectors. 

Details about this software will be presented in the prepared publication.


## Submodules

Submodules lives in the submodules directory. To clone project including submodules one have to use the following commands

```
git submodule init
git submodule update
```
or use the following command when cloning repository

```
git clone --recurse-submodules
```

### [CTIOL](https://github.com/kulvait/KCT_ctiol)

Input output routines for asynchronous thread safe reading/writing CT data. The DEN format read/write is implemented.

### [CTMAL](https://github.com/kulvait/KCT_ctmal)

Mathematic/Algebraic algorithms for supporting CT data manipulation.

### [Plog](https://github.com/SergiusTheBest/plog)

Logger Plog is used for logging. It is licensed under the Mozilla Public License Version 2.0.

### [CLI11](https://github.com/CLIUtils/CLI11)

Comand line parser CLI11. It is licensed under 3 Clause BSD License.

### [Catch2](https://github.com/catchorg/Catch2)

Testing framework. Licensed under Boost Software License 1.0.

### [CTPL](https://github.com/vit-vit/ctpl)

Threadpool library.


## Licensing

When there is no other licensing and/or copyright information in the source files of this project, the following apply for the source files in the directories include and src and for CMakeLists.txt file:

Copyright (C) 2018-2021 Vojtěch Kulvait

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, version 3 of the License.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.


This licensing applies to the direct source files in the directories include and src of this project and not for submodules.
