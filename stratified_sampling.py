import pandas as pd
import numpy as np
import unicodedata

# 1. Text Normalization Function
def normalize_text(text):
    if pd.isna(text):
        return ""
    text = str(text)
    # Normalize unicode characters to decompose combined characters (like accents)
    text = unicodedata.normalize('NFD', text)
    # Filter out non-spacing mark characters (accents)
    text = ''.join(c for c in text if unicodedata.category(c) != 'Mn')
    # Convert to lowercase and strip whitespace
    return text.lower().strip()

# Load Data
try:
    # Assuming standard CSV format (comma separated).
    # If your CSV uses a different delimiter (e.g., semicolon), add sep=';'
    df_diseno = pd.read_csv('diseño_muestral.csv')
    df_rural = pd.read_csv('RURAL_URBANO.csv')
except FileNotFoundError:
    print("Error: Input files not found. Please upload 'diseño_muestral.csv' and 'RURAL_URBANO.csv'.")
    raise

# Manual mapping for known discrepancies (can be expanded)
# "Vargas" is now "La Guaira" in many datasets
state_mapping = {
    'vargas': 'la guaira',
    # Add other mappings if necessary
}

# Normalize State Names
df_diseno['estado_norm'] = df_diseno['estado'].apply(normalize_text)
# Apply manual mapping to design dataframe's normalized states
df_diseno['estado_norm'] = df_diseno['estado_norm'].replace(state_mapping)

df_rural['ESTADO_NORM'] = df_rural['ESTADO'].apply(normalize_text)

# 2. Calculate Rural/Urban Weights
# Count municipalities by State and Type
muni_counts = df_rural.groupby(['ESTADO_NORM', 'TIPO']).size().reset_index(name='count')

# Ensure both 'Rural' and 'Urbano' exist for all states
# Get all unique states from the rural/urban dataset
states = df_rural['ESTADO_NORM'].unique()
# Create a DataFrame with all combinations of State and Type
all_combinations = pd.MultiIndex.from_product([states, ['Rural', 'Urbano']], names=['ESTADO_NORM', 'TIPO']).to_frame(index=False)

# Merge counts onto all combinations
muni_counts = pd.merge(all_combinations, muni_counts, on=['ESTADO_NORM', 'TIPO'], how='left').fillna(0)

# Calculate total municipalities per state
state_totals = muni_counts.groupby('ESTADO_NORM')['count'].transform('sum')

# Calculate weights
# Division is safe here because every state in 'all_combinations' exists in df_rural and thus has at least one municipality.
muni_counts['weight'] = muni_counts['count'] / state_totals

# 3. Merge Weights onto Design Data
# We merge on the normalized state column
df_merged = pd.merge(df_diseno, muni_counts[['ESTADO_NORM', 'TIPO', 'weight']],
                     left_on='estado_norm', right_on='ESTADO_NORM', how='left')

# Check for any states that didn't match and warn the user
unmatched = df_merged[df_merged['weight'].isna()]['estado'].unique()
if len(unmatched) > 0:
    print(f"Warning: The following states in diseño_muestral.csv did not match any state in RURAL_URBANO.csv and will have NaN estimates: {unmatched}")

# 4. Calculate Estimated Population and Sample Size
df_merged['poblacion_est'] = df_merged['poblacion'] * df_merged['weight']
# Round sample size up to the nearest whole number
df_merged['muestra_est'] = np.ceil(df_merged['muestra'] * df_merged['weight'])

# 5. Error Calculations
Z = 1.645 # 90% confidence
p = 0.5
q = 0.5

# Standard Error (SE): sqrt((p * q) / n)
# Handle division by zero or NaN n
df_merged['SE'] = np.sqrt((p * q) / df_merged['muestra_est'])

# Finite Population Correction (FPC): sqrt((N - n) / (N - 1))
N = df_merged['poblacion_est']
n = df_merged['muestra_est']

# Calculate FPC term
# Use numpy where to handle conditions safely
# Condition 1: N > 1 (to avoid division by zero or negative denominator)
# Condition 2: N >= n (to avoid negative value in sqrt)
fpc_term = (N - n) / (N - 1)
# If N <= 1, FPC is not applicable (or variance is 0).
# We'll set term to 0 if invalid.
fpc_term = fpc_term.where((N > 1) & (fpc_term >= 0), 0)

df_merged['FPC'] = np.sqrt(fpc_term)

# Design Error (Adjusted SE)
df_merged['Design_Error'] = df_merged['SE'] * df_merged['FPC']

# Margin of Error (MoE)
df_merged['MoE'] = Z * df_merged['Design_Error']

# Handle edge cases explicitly
# If n <= 0, MoE is NaN
df_merged.loc[df_merged['muestra_est'] <= 0, 'MoE'] = np.nan
# If n >= N, MoE is 0
df_merged.loc[(df_merged['muestra_est'] >= df_merged['poblacion_est']) & (df_merged['muestra_est'] > 0), 'MoE'] = 0.0

# 6. Output Formatting
# Format MoE as percentage string for display
df_merged['MoE_Percentage'] = df_merged['MoE'].apply(lambda x: f"{x:.2%}" if not pd.isna(x) else "NaN")

# Select relevant columns for final output
# Using original column names where possible
final_columns = [
    'estado', 'genero', 'grupo_edad', 'TIPO',
    'poblacion', 'muestra', 'weight',
    'poblacion_est', 'muestra_est',
    'SE', 'FPC', 'Design_Error', 'MoE', 'MoE_Percentage'
]

# Create final DataFrame
df_final = df_merged[final_columns].copy()

print("Processing complete.")
print(f"Original rows: {len(df_diseno)}")
print(f"Expanded rows: {len(df_final)}")

# Display first 15 rows
print("\nFirst 15 rows of the final dataframe:")
print(df_final[['estado', 'genero', 'grupo_edad', 'TIPO', 'poblacion_est', 'muestra_est', 'MoE_Percentage']].head(15).to_string())

# Export to CSV
output_filename = 'stratified_design_final.csv'
df_final.to_csv(output_filename, index=False)
print(f"\nFinal dataset exported to {output_filename}")
