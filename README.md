# Survey Data Processing

This repository contains scripts and data for processing survey samples, calculating weights based on stratified sampling (Gender, Age, Rural/Urban), and generating a dashboard of results.

## Files

- `process_sample.py`: Main script to process data, calculate weights, and generate `dashboard.html`.
- `stratified_sampling.py`: Alternative script for design error calculation.
- `analysis_notebook.ipynb`: Jupyter Notebook designed to run the analysis in Google Colab.
- `sample_data.csv`: Input survey data (semicolon separated).
- `diseño_muestral.xlsx`: Sampling design targets.
- `RURAL_URBANO.xlsx`: Rural/Urban classification of municipalities.

## Running in Google Colab

To run the analysis in Google Colab using the provided notebook:

1.  **Open the Notebook**: Upload `analysis_notebook.ipynb` to Google Drive and open it with Google Colab.
2.  **Configure Repository**: In the "Clone Repository" cell, replace `YOUR_REPO_URL_HERE` with the URL of this repository (e.g., `https://github.com/username/repo.git`).
3.  **Run All Cells**: Execute the cells in order.
    -   The notebook will mount your Google Drive.
    -   It will clone the repository into your Drive (folder `MyDrive/<repo_name>`).
    -   It will install necessary dependencies (`pandas`, `numpy`, `openpyxl`, `plotly`).
    -   It will run the processing script and display the dashboard.

## Local Execution

To run locally:

1.  Install dependencies:
    ```bash
    pip install pandas numpy openpyxl plotly
    ```
2.  Run the script:
    ```bash
    python process_sample.py
    ```
3.  Open `dashboard.html` in your browser.
