import pandas as pd
import numpy as np
import unicodedata

# Constants for default configuration
TARGET_N = 3000
N_STATES = 10
MUNIS_PER_STATE = 2

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

def simulate_scenarios(df_design):
    print("Simulating Scenarios (Sample Size vs Error)...")

    # Simple simulation of Error vs Sample Size
    # MoE = Z * sqrt(p*q/n) * sqrt(Deff) * sqrt((N-n)/(N-1))

    scenarios = []

    # Total Population
    N = df_design['poblacion'].sum()
    p = 0.5
    q = 0.5
    Z = 1.96 # 95% Confidence Level
    Deff = 1.5 # Assumed Design Effect for cluster sampling

    sample_sizes = [1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000]

    for n in sample_sizes:
        if n >= N:
            moe = 0
        else:
            se = np.sqrt((p*q)/n)
            # Finite Population Correction
            fpc = np.sqrt((N-n)/(N-1))
            moe = Z * se * np.sqrt(Deff) * fpc

        scenarios.append({
            'Tamaño de Muestra': n,
            'Margen de Error (95%)': moe,
            'Nivel de Confianza': '95%',
            'Efecto de Diseño (Asumido)': Deff
        })

    return pd.DataFrame(scenarios)

def select_sample(df_design, df_rural, target_n=TARGET_N, n_states=N_STATES, munis_per_state=MUNIS_PER_STATE):
    print(f"Selecting Sample (Target N={target_n}, States={n_states}, Munis/State={munis_per_state})...")

    # Normalize
    df_design['estado_norm'] = df_design['estado'].apply(normalize_text)
    df_rural['ESTADO_NORM'] = df_rural['ESTADO'].apply(normalize_text)

    # Handle Vargas -> La Guaira mapping in design data
    # Check if 'vargas' exists in design but only 'la guaira' in rural
    unique_rural_states = df_rural['ESTADO_NORM'].unique()
    if 'vargas' not in unique_rural_states and 'la guaira' in unique_rural_states:
        df_design.loc[df_design['estado_norm'] == 'vargas', 'estado_norm'] = 'la guaira'

    # Get list of unique states in design
    all_states = df_design['estado_norm'].unique()

    # Select States Randomly
    if n_states >= len(all_states):
        selected_states = all_states
    else:
        # Sort for reproducibility if needed, or just random
        # Using random choice
        selected_states = np.random.choice(all_states, size=n_states, replace=False)

    print(f"Selected States: {selected_states}")

    # Calculate Total Population of Selected States
    df_design_selected = df_design[df_design['estado_norm'].isin(selected_states)].copy()
    total_pop_selected = df_design_selected['poblacion'].sum()

    # Calculate Target Sample per State (Proportional Allocation)
    state_pops = df_design_selected.groupby('estado_norm')['poblacion'].sum().reset_index()
    state_pops['state_prop'] = state_pops['poblacion'] / total_pop_selected
    state_pops['target_n_state'] = (state_pops['state_prop'] * target_n).round().astype(int)

    # Prepare Output List
    sample_list = []

    for state in selected_states:
        # Get target n for this state
        n_state_series = state_pops.loc[state_pops['estado_norm'] == state, 'target_n_state']
        if n_state_series.empty:
            continue
        n_state = n_state_series.values[0]

        if n_state == 0:
            continue

        # Get available municipalities for this state
        state_munis = df_rural[df_rural['ESTADO_NORM'] == state]

        # Determine Rural/Urban split based on municipality counts
        n_rural_munis = state_munis[state_munis['TIPO'] == 'Rural'].shape[0]
        n_urban_munis = state_munis[state_munis['TIPO'] == 'Urbano'].shape[0]
        total_munis = n_rural_munis + n_urban_munis

        if total_munis == 0:
            continue

        prop_rural = n_rural_munis / total_munis
        prop_urban = n_urban_munis / total_munis

        target_n_rural = int(round(n_state * prop_rural))
        target_n_urban = int(round(n_state * prop_urban))

        # Adjust remainder
        remainder = n_state - (target_n_rural + target_n_urban)
        if remainder != 0:
            if target_n_rural >= target_n_urban:
                target_n_rural += remainder
            else:
                target_n_urban += remainder

        # Select Municipalities
        selected_munis = []

        # Select Rural Munis
        rural_options = state_munis[state_munis['TIPO'] == 'Rural']['MUNICIPIO'].unique()
        if target_n_rural > 0 and len(rural_options) > 0:
            # We try to pick proportional number of munis based on type split,
            # but ensure we don't pick 0 if we need sample
            n_pick_rural = max(1, int(round(munis_per_state * prop_rural)))
            if n_pick_rural > len(rural_options):
                n_pick_rural = len(rural_options)

            picked_rural = np.random.choice(rural_options, size=n_pick_rural, replace=False)

            base_n = target_n_rural // n_pick_rural
            rem = target_n_rural % n_pick_rural

            for i, muni in enumerate(picked_rural):
                quota = base_n + (1 if i < rem else 0)
                selected_munis.append({'MUNICIPIO': muni, 'TIPO': 'Rural', 'QUOTA': quota})

        elif target_n_rural > 0:
            pass # No rural munis available but needed quota? Unlikely if props are from same file.

        # Select Urban Munis
        urban_options = state_munis[state_munis['TIPO'] == 'Urbano']['MUNICIPIO'].unique()
        if target_n_urban > 0 and len(urban_options) > 0:
            n_pick_urban = max(1, int(round(munis_per_state * prop_urban)))
            if n_pick_urban > len(urban_options):
                n_pick_urban = len(urban_options)

            # Ensure we pick at least 1 urban if quota exists and options exist
            if n_pick_urban == 0 and target_n_urban > 0:
                n_pick_urban = 1

            picked_urban = np.random.choice(urban_options, size=n_pick_urban, replace=False)

            base_n = target_n_urban // n_pick_urban
            rem = target_n_urban % n_pick_urban

            for i, muni in enumerate(picked_urban):
                quota = base_n + (1 if i < rem else 0)
                selected_munis.append({'MUNICIPIO': muni, 'TIPO': 'Urbano', 'QUOTA': quota})

        # Distribute Quota into Gender/Age buckets
        # using State's demographic distribution from design file

        # Get State Demographics
        state_demo = df_design[df_design['estado_norm'] == state].copy()
        state_total_pop = state_demo['poblacion'].sum()

        if state_total_pop == 0:
            continue

        for muni_info in selected_munis:
            muni_quota = muni_info['QUOTA']
            if muni_quota == 0:
                continue

            # Create a distribution list for this muni
            muni_sample_distribution = []

            # First pass: calculate float quotas
            state_demo['temp_quota'] = (state_demo['poblacion'] / state_total_pop) * muni_quota
            state_demo['int_quota'] = state_demo['temp_quota'].astype(int)
            state_demo['decimal_part'] = state_demo['temp_quota'] - state_demo['int_quota']

            # Calculate remainder
            assigned_total = state_demo['int_quota'].sum()
            remainder_quota = muni_quota - assigned_total

            # Distribute remainder to largest decimals
            if remainder_quota > 0:
                # Sort by decimal part descending
                state_demo = state_demo.sort_values('decimal_part', ascending=False)
                # Add 1 to the top 'remainder_quota' rows
                state_demo.iloc[:remainder_quota, state_demo.columns.get_loc('int_quota')] += 1

            # Collect results
            for _, row in state_demo.iterrows():
                if row['int_quota'] > 0:
                    sample_list.append({
                        'Estado': row['estado'], # Original name
                        'Municipio': muni_info['MUNICIPIO'],
                        'Tipo': muni_info['TIPO'],
                        'Género': row['genero'],
                        'Grupo de Edad': row['grupo_edad'],
                        'Cuota': row['int_quota']
                    })

    df_sample_design = pd.DataFrame(sample_list)
    # Sort for cleaner output
    if not df_sample_design.empty:
        df_sample_design = df_sample_design.sort_values(['Estado', 'Municipio', 'Tipo', 'Género', 'Grupo de Edad'])

    return df_sample_design

def main():
    try:
        df_design, df_rural = load_data()

        # 1. Simulation Sheet
        df_simulation = simulate_scenarios(df_design)

        # 2. Sample Design Sheet
        df_sample_design = select_sample(df_design, df_rural, target_n=TARGET_N, n_states=N_STATES, munis_per_state=MUNIS_PER_STATE)

        # 3. Metrics Sheet
        if not df_sample_design.empty:
            actual_n = df_sample_design['Cuota'].sum()
            N = df_design['poblacion'].sum()
            p = 0.5
            q = 0.5
            Z = 1.96
            Deff = 1.5

            se = np.sqrt((p*q)/actual_n) if actual_n > 0 else 0
            fpc = np.sqrt((N-actual_n)/(N-1)) if N > 1 else 1
            moe = Z * se * np.sqrt(Deff) * fpc

            metrics = [{
                'Tamaño de Muestra Real': actual_n,
                'Margen de Error (95%)': moe,
                'Nivel de Confianza': '95%',
                'Efecto de Diseño': Deff,
                'Población Total (Referencia)': N,
                'Estados Seleccionados': df_sample_design['Estado'].nunique(),
                'Municipios Seleccionados': df_sample_design['Municipio'].nunique()
            }]
            df_metrics = pd.DataFrame(metrics)
        else:
            df_metrics = pd.DataFrame([{'Info': 'No sample generated'}])

        # Write to Excel with multiple sheets
        output_file = 'plan_de_campo.xlsx'
        with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
            df_sample_design.to_excel(writer, sheet_name='Plan de Campo', index=False)
            df_metrics.to_excel(writer, sheet_name='Métricas', index=False)
            df_simulation.to_excel(writer, sheet_name='Simulación Muestra', index=False)

        print(f"Field design saved to {output_file}")

    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
