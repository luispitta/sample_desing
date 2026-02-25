import pandas as pd
import numpy as np
import unicodedata

def normalize_text(text):
    if pd.isna(text):
        return ""
    text = str(text)
    text = unicodedata.normalize('NFD', text)
    text = ''.join(c for c in text if unicodedata.category(c) != 'Mn')
    return text.lower().strip()

def load_data():
    print("Loading data...")
    df_design = pd.read_excel('diseño_muestral.xlsx')
    df_rural = pd.read_excel('RURAL_URBANO.xlsx')
    return df_design, df_rural

def generate_design(df_design, df_rural):
    print("Processing design...")

    # Normalize State Names
    df_design['estado_norm'] = df_design['estado'].apply(normalize_text)
    df_rural['ESTADO_NORM'] = df_rural['ESTADO'].apply(normalize_text)

    # Handle Vargas -> La Guaira if necessary
    unique_rural_states = df_rural['ESTADO_NORM'].unique()
    if 'vargas' not in unique_rural_states and 'la guaira' in unique_rural_states:
        df_design.loc[df_design['estado_norm'] == 'vargas', 'estado_norm'] = 'la guaira'
        print("Mapped 'vargas' to 'la guaira'")

    # Calculate Municipality Proportions per State and Type
    muni_counts = df_rural.groupby(['ESTADO_NORM', 'TIPO']).size().reset_index(name='n_munis')
    state_muni_totals = df_rural.groupby('ESTADO_NORM').size().reset_index(name='total_munis')

    muni_props = pd.merge(muni_counts, state_muni_totals, on='ESTADO_NORM')
    muni_props['type_prop'] = muni_props['n_munis'] / muni_props['total_munis']

    # Merge proportions into Design
    df_merged = pd.merge(df_design, muni_props[['ESTADO_NORM', 'TIPO', 'type_prop']],
                         left_on='estado_norm', right_on='ESTADO_NORM', how='left')

    # Check for unmapped states
    unmapped = df_merged[df_merged['TIPO'].isna()]['estado'].unique()
    if len(unmapped) > 0:
        print(f"Warning: The following states in design were not found in Rural/Urban file: {unmapped}")
        # Fill NaN type_prop with 0 to avoid errors, effectively skipping these for the detailed design
        df_merged['type_prop'] = df_merged['type_prop'].fillna(0)

    # Calculate Target Sample for this specific stratum (State-Type-Gender-Age)
    df_merged['target_sample'] = df_merged['muestra'] * df_merged['type_prop']
    df_merged['target_sample'] = df_merged['target_sample'].fillna(0)

    # Rounding and converting to int
    df_merged['target_sample_int'] = df_merged['target_sample'].round().astype(int)

    # Add list of municipalities for reference
    muni_lists = df_rural.groupby(['ESTADO_NORM', 'TIPO'])['MUNICIPIO'].apply(lambda x: ', '.join(x)).reset_index(name='municipios_validos')

    df_final = pd.merge(df_merged, muni_lists, on=['ESTADO_NORM', 'TIPO'], how='left')

    # Select and rename columns for output
    output_cols = [
        'estado', 'TIPO', 'genero', 'grupo_edad',
        'muestra', 'type_prop', 'target_sample', 'target_sample_int',
        'municipios_validos'
    ]

    df_output = df_final[output_cols].copy()
    df_output.rename(columns={
        'estado': 'Estado',
        'TIPO': 'Tipo (Rural/Urbano)',
        'genero': 'Género',
        'grupo_edad': 'Grupo de Edad',
        'muestra': 'Meta Estado-Genero-Edad',
        'type_prop': 'Proporción Tipo',
        'target_sample': 'Meta Calculada (Decimal)',
        'target_sample_int': 'Meta Calculada (Entero)',
        'municipios_validos': 'Municipios Disponibles'
    }, inplace=True)

    # Remove rows where Meta Calculada (Entero) is 0 to clean up the sheet?
    # Or keep them? Maybe keep them but sort by State.
    df_output = df_output.sort_values(['Estado', 'Tipo (Rural/Urbano)', 'Género', 'Grupo de Edad'])

    return df_output

def main():
    try:
        df_design, df_rural = load_data()
        df_field_design = generate_design(df_design, df_rural)

        output_file = 'plan_de_campo.xlsx'
        df_field_design.to_excel(output_file, index=False)
        print(f"Field design saved to {output_file}")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
