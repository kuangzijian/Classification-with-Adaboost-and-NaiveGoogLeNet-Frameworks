
<!-- PROJECT LOGO
<br />
<p align="center">
  <a href="https://github.com/othneildrew/Best-README-Template">
    <img src="images/logo.png" alt="Logo" width="80" height="80">
  </a>

  <h3 align="center">Best-README-Template</h3>

  <p align="center">
    An awesome README template to jumpstart your projects!
    <br />
    <a href="https://github.com/othneildrew/Best-README-Template"><strong>Explore the docs »</strong></a>
    <br />
    <br />
    <a href="https://github.com/othneildrew/Best-README-Template">View Demo</a>
    ·
    <a href="https://github.com/othneildrew/Best-README-Template/issues">Report Bug</a>
    ·
    <a href="https://github.com/othneildrew/Best-README-Template/issues">Request Feature</a>
  </p>
</p>
-->


<!-- TABLE OF CONTENTS 
## Table of Contents

* [About the Project](#about-the-project)
  * [Built With](#built-with)
* [Getting Started](#getting-started)
  * [Prerequisites](#prerequisites)
  * [Installation](#installation)
* [Usage](#usage)
* [Roadmap](#roadmap)
* [Contributing](#contributing)
* [License](#license)
* [Contact](#contact)
* [Acknowledgements](#acknowledgements)
-->


<!-- ABOUT THE PROJECT -->
## UAlberta-Multimedia-Master-Program-MM811-2020-Assignment-2

This is a school project assigned by University of Alberta Mast of Multimedia program. This project focuses on topics:
* AdaBoost
* Pytorch autograd
* Mnist data classification with original GoogLeNet and NaiveGoogLeNet frameworks.

### Built With
* [Pytorch](https://github.com/pytorch)

### Prerequisites
```sh
1. Clone the repo
2. pip install -r requirements.txt
```

### Q2 Implement the Adaboost algorithm class according to the Adaboost Algorithm 

```
cd Q2_AdaBoost
python Q2_test.py
```

### Q3 Implement the following network (with i1=0.05, i2=0.1, b1=0.35, o1=0.01, o2=0.99)

```
cd Q3_NN_with_Autograd
python NN_with_Autograd.py
```

### Q4 Implement Naive Inception Module for GoogLeNet and use it to train and test by using the MNIST dataset

```
cd Q4_mnist_googlenet
python train_googlenet.py
python train_naive_googlenet.py
```

<!-- CONTRIBUTING -->
## Contributing

Contributions are what make the open source community such an amazing place to be learn, inspire, and create. Any contributions you make are **greatly appreciated**.

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request



<!-- LICENSE -->
## License

Distributed under the MIT License. See `LICENSE` for more information.

## References
MNIST Data example. https://github.com/pytorch/examples/tree/master/mnist

PyTorch GoogLeNet implementation. https://github.com/pytorch/vision/blob/master/torchvision/models/googlenet.py


