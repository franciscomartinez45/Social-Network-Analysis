# Graph Attention Network 🚀

Supervised Machine Learning sandbox used to explore different message passing and influence algorithms from various synthetic datasets. Used as a means to research Addressing Health Disparities through Improved Health Literacy in Minority Populations using AI/ML Models and Social Network Analysis. Created to analyze and potentially improve how minority populations access and comprehend health infromation by leveraging predictive and pattern detection using Machine Learning.

## Table of Contents

- [Features](#features)
- [Screenshots](#screenshots)
- [Installation](#installation)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgments](#acknowledgments)

## Features

- **Interactive Visualization**: Leverages NetworkX and Matplotlib to visualize graph structures, node influences, and model predictions.
- **SHAP Explainer**: Utilized to measure of feature importance by computing Shapley values
- **Health Equity Focus**: Tailored to study and address health disparities in underserved or misrepresented communities.
- **Misinformation Modeling**: Analyzes and models the spread of health-related information (or misinformation) across social networks, providing actionable insights for intervention strategies.

## Screenshots

Sample result of 128 nodes x 125 edges using GTgraph Synthetic Data Generator Suite
![App Screenshot](https://github.com/franciscomartinez45/Graph-Convolutional-Network/blob/main/img/visualization.png)

## Installation

To get a local copy up and running, follow these steps:

1. Clone the repository:
   ```bash
   git clone https://github.com/franciscomartinez45/Graph-Convolutional-Network.git
   ```

## Acknowledgments

- **GTgraph**: This project utilizes [GTgraph](https://github.com/Bader-Research/GTgraph), a suite of synthetic random graph generators developed for the 9th DIMACS Shortest Paths Challenge. GTgraph supports various classes of graphs, including:

  - Input graph instances used in the DARPA HPCS SSCA#2 graph theory benchmark (version 1.0).
  - Erdős-Rényi random graphs.
  - Small-world graphs based on the Recursive Matrix (R-MAT) model.

- **NetworkX**: Utilize synthetic graph generators from NetworkX such as Barabasi-Albert and Zachary's Karate Club
