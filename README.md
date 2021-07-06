# ipd-autodiff

Code for the paper "Application of Automatic Differentiation in Electromagnetic Dosimetry - Assessment of the Absorbed Power Density in the mmWave Frequency Spectrum"

## Application of Automatic Differentiation in Electromagnetic Dosimetry - Assessment of the Absorbed Power Density in the mmWave Frequency Spectrum

This repository contains all code necessary for reproducing the results presented in the paper.

Authors : [Ante Lojic Kapetanovic](http://adria.fesb.hr/~alojic00/), Dragan Poljak

### Abstract

This paper introduces the concept of automatic differentiation in the evaluation of the absorbed power density in the mmWave frequency spectrum for the new generation of mobile telecommunication technology.
Automatic differentiation is proved to be far superior over numerical differentiation by means of speed and accuracy.
To demonstrate the full capacity of the proposed method, a comprehensive analysis of computing the absorbed power density on the surface of irradiated human skin in various configurations is presented.

### Reproduce the results

Install prerequisities from `requirements.txt`.

To enable GPU support install `jax` with CUDA support by first installing CUDA and CUDNN and run:
```shel
pip install --upgrade pip
pip install --upgrade "jax[cuda111]" -f https://storage.googleapis.com/jax-releases/jax_releases.html

```
Details available in the official `jax` documentation.

All figures are available by running `main.ipynb` in Jupyter Notebook.

### Citatio

The paper has not yet been published online.

A. L. Kapetanovic and D. Poljak, "Application of Automatic Differentiation in Electromagnetic Dosimetry - Assessment of the Absorbed Power Density in the mmWave Frequency Spectrum," 2021 6th International Conference on Smart and Sustainable Technologies (SpliTech), 2021, pp. 1-6, doi: tba.

```bibtex
@inproceedings{kapetanovic2021application,
  title={Application of Automatic Differentiation in Electromagnetic Dosimetry - {Assessment} of the Absorbed Power Density in the {mmWave} Frequency Spectrum},
  author={Lojic Kapetanovic, Ante and Poljak, Dragan},
  year={2021},
  booktitle={2020 5th International Conference on Smart and Sustainable Technologies (SpliTech)},
  pages={1--6},
  doi={},
  url={}
}
```

### License

[MIT](https://github.com/antelk/ipd-autodiff/blob/main/LICENSE)