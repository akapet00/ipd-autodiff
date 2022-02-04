# ipd-autodiff

Code for the paper:

## Application of Automatic Differentiation in Electromagnetic Dosimetry - Assessment of the Absorbed Power Density in the mmWave Frequency Spectrum
Authors: [Ante Lojic Kapetanovic](http://adria.fesb.hr/~alojic00/), Dragan Poljak

### Abstract
This paper introduces the concept of automatic differentiation in the evaluation of the absorbed power density in the mmWave frequency spectrum for the new generation of mobile telecommunication technology. Automatic differentiation has been shown to be far superior over numerical differentiation by means of speed and accuracy. To demonstrate the full capacity of the proposed method, a comprehensive analysis of computing the absorbed power density on the surface of irradiated human skin in various configurations is presented.

### Reproduce the results
Install prerequisities from `requirements.txt`.

To enable GPU support install `jax` with CUDA support by first installing CUDA and CUDNN and run:
```shel
pip install --upgrade pip
pip install --upgrade "jax[cuda111]" -f https://storage.googleapis.com/jax-releases/jax_releases.html
```

Details available in the official `jax` documentation.

All figures and results in the paper can be reproduced by running `main.ipynb` in Jupyter Notebook.

### Citation
To cite this work, please use the following:
  
  A. L. Kapetanović and D. Poljak, "Application of Automatic Differentiation in Electromagnetic Dosimetry - Assessment of the Absorbed Power Density in the mmWave Frequency Spectrum," 2021 6th International Conference on Smart and Sustainable Technologies (SpliTech), 2021, pp. 1-6, doi: 10.23919/SpliTech52315.2021.9566429.

or bibtex entry:

```bibtex
@INPROCEEDINGS{Kapetanovic2021Application,
  author={Lojić Kapetanović, Ante and Poljak, Dragan},
  booktitle={2021 6th International Conference on Smart and Sustainable Technologies (SpliTech)},
  title={Application of Automatic Differentiation in Electromagnetic Dosimetry - {Assessment} of the Absorbed Power Density in the mmWave Frequency Spectrum},
  year={2021},
  pages={1-6},
  doi={10.23919/SpliTech52315.2021.9566429},
  url={https://doi.org/10.23919/SpliTech52315.2021.9566429}}
```

### License

[MIT](https://github.com/antelk/ipd-autodiff/blob/main/LICENSE)
