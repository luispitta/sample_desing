# Survey Data Processing

This repository contains scripts and data for processing survey samples, calculating weights based on stratified sampling (Gender, Age, Rural/Urban), and generating a dashboard of results. It also includes tools to generate detailed field work plans based on the sample design.

## Files

- `process_sample.py`: Main script to process collected data, calculate weights, and generate `dashboard.html`.
- `create_field_design.py`: Script to generate `plan_de_campo.xlsx`. It scales the design to 3000 surveys, selects municipalities (Rural/Urban), and calculates detailed quotas.
- `stratified_sampling.py`: Alternative script for design error calculation.
- `analysis_notebook.ipynb`: Jupyter Notebook designed to run the analysis and generate field plans in Google Colab.
- `sample_data.csv`: Input survey data (semicolon separated).
- `diseño_muestral.xlsx`: Sampling design targets (State level).
- `RURAL_URBANO.xlsx`: Rural/Urban classification of municipalities.

## Running in Google Colab

To run the analysis or generate field plans in Google Colab using the provided notebook:

1.  **Open the Notebook**: Upload `analysis_notebook.ipynb` to Google Drive and open it with Google Colab.
2.  **Configure Repository**: In the "Clone Repository" cell, replace `YOUR_REPO_URL_HERE` with the URL of this repository.
3.  **Run All Cells**: Execute the cells in order.
    -   The notebook will mount your Google Drive and clone the repository.
    -   It will install necessary dependencies.
    -   **Generate Field Design**: A cell is provided to run `create_field_design.py`. This produces `plan_de_campo.xlsx` with two sheets:
        -   `Plan de Campo`: Rows with State, Municipality, Type, Gender, Age, and Quota.
        -   `Métricas`: Sampling metrics (Confidence Level, Margin of Error, etc.).
    -   **Run Analysis**: It will run `process_sample.py` to process `sample_data.csv` and display the dashboard.

## Local Execution

To run locally:

1.  Install dependencies:
    ```bash
    pip install pandas numpy openpyxl plotly
    ```

2.  **Generate Field Design Plan**:
    ```bash
    python create_field_design.py
    ```
    This creates `plan_de_campo.xlsx`.

3.  **Process Collected Data**:
    ```bash
    python process_sample.py
    ```
    This processes `sample_data.csv` and creates `dashboard.html`.
