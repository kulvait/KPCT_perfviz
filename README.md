# KPCT Perfusion visualization

Visualization of perfusion parameters. Computation of TTP, MTT, CBV, CBF. Exporting volume series of time attenuation curves or time resolved CT volumes based on the underlying model.

TST model of dimension reduction works with apriori-knowledge or engineer bases, two bases of real polynomials Legendre and Chebyshev and trigonometric functions.

## Perfusion maps processing

Deconvolution based model for computing these parameters is described in the paper [Time separation technique with the basis of trigonometric functions as an efficient method for flat detector CT brain perfusion imaging](https://arxiv.org/abs/2110.09438).

It is derived from the model in [Fieselmann et.al. 2011](http://dx.doi.org/10.1155/2011/467563) but the CBF computation can be modified by so called CBF time parameter. Using this parameter and setting it to 6s in accordance to [DEFUSE3 study protocol](https://clinicaltrials.gov/ProvidedDocs/15/NCT02586415/Prot_001.pdf) shall make MTT a better indicator of colateral flow.



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

### [Plog](https://github.com/SergiusTheBest/plog) logger

Logger Plog is used for logging. It is licensed under the Mozilla Public License Version 2.0.

### [CLI11](https://github.com/CLIUtils/CLI11)

Comand line parser CLI11. It is licensed under 3 Clause BSD License.

### [Catch2](https://github.com/catchorg/Catch2)

Testing framework. Licensed under Boost Software License 1.0.

### [CTPL](https://github.com/vit-vit/ctpl)

Threadpool library.

### [CTIOL](https://github.com/kulvait/KCT_ctiol)

Input output routines for asynchronous thread safe reading/writing CT data. The DEN format read/write is implemented.

### [CTMAL](https://github.com/kulvait/KCT_ctmal)

Mathematic/Algebraic algorithms for supporting CT data manipulation.

## Documentation

Documentation is generated using doxygen and lives in doc directory.
First the config file for doxygen was prepared runing doxygen -g.
Doc files and this file can be written using [Markdown syntax](https://daringfireball.net/projects/markdown/syntax), JAVADOC_AUTOBRIEF is set to yes to treat first line of the doc comment as a brief description, comments are of the format 
```
/**Brief description.
*
*Long description
*thay might span multiple lines.
*/
```
.
