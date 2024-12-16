# Property Enhancer (PropEn)

This is the official open source repository for [PropEn](https://openreview.net/pdf?id=dhFHO90INk) developed by [tagas](https://tagas.github.io/aboutme/) from [Prescient Design, a Genentech accelerator.](https://gene.com/prescient)

## Setup
Assuming you have [miniconda](https://docs.conda.io/en/latest/miniconda.html) installed, clone the repository, navigate inside, and run:
```bash
pip install -f requirenments.in
```

## Run demo code
The entrypoint `run_propen_demo` is a wrapper around the matching, training and sampling for a toy dataset. To change the datasets you can use `utils` and to modify the model `propen`.

## Contributing

We welcome contributions. If you would like to submit pull requests, please make sure you base your pull requests off the latest version of the `main` branch.

## License
Licensed under a modified Apache License, Version 2.0 (the "License" which is Genentech Non-Commercial Software License); you may not use this file except in compliance with the License. You may obtain a [copy of the License](Genentech Non-Commercial Apache 2.0.tx).

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissionst and limitations under the License.


## Citations
If you use the code and/or model, please cite:
```
@article{tagasovska2024implicitly,
  title={Implicitly Guided Design with PropEn: Match your Data to Follow the Gradient},
  author={Tagasovska, Nata{\v{s}}a and Gligorijevi{\'c}, Vladimir and Cho, Kyunghyun and Loukas, Andreas},
  journal={arXiv preprint arXiv:2405.18075},
  year={2024}
}
```
