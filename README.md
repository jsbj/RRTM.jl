# RRTM.jl

[![Build Status](https://travis-ci.org/jsbj/RRTM.jl.svg?branch=master)](https://travis-ci.org/jsbj/RRTM.jl)

[![Coverage Status](https://coveralls.io/repos/jsbj/RRTM.jl/badge.svg?branch=master&service=github)](https://coveralls.io/github/jsbj/RRTM.jl?branch=master)

[![codecov.io](http://codecov.io/github/jsbj/RRTM.jl/coverage.svg?branch=master)](http://codecov.io/github/jsbj/RRTM.jl?branch=master)

This is an implementation of the widely used Fortran radiative transfer code RRTM in Julia. Specifically, this is the version of RRTM used in MPI's general circulation model ECHAM version 6.1. Radiative transfer codes calculate how "shortave" light (sunlight) and "longwave" light (infrared light given off by the Earth and its atmosphere) are absorbed, emitted, reflected, and transmitted by the atmosphere.

## Getting started

First, let's go over installation instructions.

### Prerequisites

You will need a working copy of [Julia](http://julialang.org/) (this was developed for Julia 0.6), and a few other libraries. Most of the dependencies will be handled by Julia's package manager, but you will first need to download and install the Python module for handling NetCDFs, [xarray](http://xarray.pydata.org/en/stable/).

### Installing

Assuming you have Julia and xarray installed, simply use Julia's package manager:
```
Pkg.add("RRTM")
```
That should do it.


## Using RRTM.jl

RRTM.jl was developed to run offline radiative code on output from ECHAM 6.1. It takes as its input a NetCDF file with the following variables defined: `mlev`, `ilev`, `time`, `ktype`, `pp\_fl`, `pp\_hl`, `pp\_sfc`, `tk\_fl`, `tk\_hl`, `tk\_sfc`, `q\_vap`, `q\_liq`, `q\_ice`, `cdnc`, `cld\_frc`, `m\_o3`, `m\_ch4`, `m\_n2o`, `psctm`, `cos\_mu0`, `cos\_mu0m`, `alb`, `hyai`, `hybi`

Then you can run:
```
radiation(input\_fn,time\_i)
```
where `input\_fn` is your netcdf file, and `time\_i` is the time index you want calculated (it is strongly advised that if you calculate multiple time steps in parallel).

This assumes a T63 grid, and uses ECHAM's standard Earth mask. If you have a different resolution/mask file, you will have to provide that mask file directly as follows: 

```
radiation(input\_fn,time\_i,mark\_fn)
```

## Authors

* **Jonah Bloch-Johnson** - *Initial work* - [jsbj](https://github.com/jsbj)

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

## Acknowledgments

Thanks to:
* Stephan Hoyer and all the people who make xarray awesome
* Hossein Pourreza and everyone at the UChicago Research Computing Center
* Thorsten Mauritsen at MPI for answering annoying questions about ECHAM arcana
