import pandas as pd
import numpy as np
import unicodedata
import math
import sys

def normalize_text(text):
    if pd.isna(text):
        return ""
    text = str(text)
    text = unicodedata.normalize('NFD', text)
    text = ''.join(c for c in text if unicodedata.category(c) != 'Mn')
    return text.lower().strip()

def load_data():
    print("Loading data...")
    try:
        df_design = pd.read_excel('diseño_muestral.xlsx')
        df_rural = pd.read_excel('RURAL_URBANO.xlsx')
    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)
    return df_design, df_rural

def scale_sample(df_design, target_total=3000):
    current_total = df_design['muestra'].sum()
    print(f"Current total sample: {current_total}")
    print(f"Target total sample: {target_total}")

    # Calculate raw target
    df_design['raw_target'] = (df_design['muestra'] / current_total) * target_total

    # Integer part
    df_design['int_target'] = np.floor(df_design['raw_target']).astype(int)

    # Remainder
    df_design['remainder'] = df_design['raw_target'] - df_design['int_target']

    # Distribute remaining count
    current_scaled_total = df_design['int_target'].sum()
    needed = int(target_total - current_scaled_total)

    print(f"Distributing {needed} remaining samples based on fractional parts...")

    # Sort by remainder descending
    df_design = df_design.sort_values('remainder', ascending=False)

    # Initialize final_quota
    df_design['final_quota'] = df_design['int_target']

    if needed > 0:
        # Use numpy array to avoid potential SettingWithCopy warnings or index alignment issues
        # We want to add 1 to the first 'needed' rows in the current sorted order
        # Since we are working on the dataframe directly, iloc relies on position
        col_idx = df_design.columns.get_loc('final_quota')
        # We need to iterate or use vector addition on the slice
        # direct += 1 on iloc slice works in modern pandas
        df_design.iloc[:needed, col_idx] += 1

    print(f"New total sample: {df_design['final_quota'].sum()}")

    return df_design.sort_index()

def select_municipalities(df_rural, seed=42):
    np.random.seed(seed)

    # Group by State and Type
    grouped = df_rural.groupby(['ESTADO_NORM', 'TIPO'])['MUNICIPIO'].apply(list).reset_index()

    selection_map = {} # (State) -> {'Rural': [munis], 'Urbano': [munis]}

    for _, row in grouped.iterrows():
        state = row['ESTADO_NORM']
        tipo = row['TIPO']
        munis = sorted(list(set(row['MUNICIPIO']))) # Unique & Sorted
        n_munis = len(munis)

        # Selection logic: 30% or min 2 (or all if < 2)
        k = max(2, math.ceil(n_munis * 0.3))
        k = min(k, n_munis)

        selected = np.random.choice(munis, size=k, replace=False).tolist()
        selected.sort()

        if state not in selection_map:
            selection_map[state] = {}
        selection_map[state][tipo] = selected

    return selection_map

def generate_field_plan(df_design, df_rural):
    # Normalize
    df_design['estado_norm'] = df_design['estado'].apply(normalize_text)
    df_rural['ESTADO_NORM'] = df_rural['ESTADO'].apply(normalize_text)

    # Fix Vargas -> La Guaira
    if 'vargas' in df_design['estado_norm'].unique() and 'la guaira' in df_rural['ESTADO_NORM'].unique():
         df_design.loc[df_design['estado_norm'] == 'vargas', 'estado_norm'] = 'la guaira'

    # Scale Sample
    df_design = scale_sample(df_design, target_total=3000)

    # Select Munis
    selection_map = select_municipalities(df_rural)

    # Calc Proportions
    muni_counts = df_rural.groupby(['ESTADO_NORM', 'TIPO']).size().reset_index(name='n_munis')
    state_totals = df_rural.groupby('ESTADO_NORM').size().reset_index(name='total_munis')
    muni_props = pd.merge(muni_counts, state_totals, on='ESTADO_NORM')
    muni_props['type_prop'] = muni_props['n_munis'] / muni_props['total_munis']

    prop_map = {}
    for _, row in muni_props.iterrows():
        if row['ESTADO_NORM'] not in prop_map:
            prop_map[row['ESTADO_NORM']] = {}
        prop_map[row['ESTADO_NORM']][row['TIPO']] = row['type_prop']

    output_rows = []

    for _, row in df_design.iterrows():
        state = row['estado_norm']
        orig_state_name = row['estado']
        gender = row['genero']
        age = row['grupo_edad']
        quota = int(row['final_quota'])

        if quota == 0:
            continue

        rural_p = prop_map.get(state, {}).get('Rural', 0.0)
        urban_p = prop_map.get(state, {}).get('Urbano', 0.0)

        # If State not in prop_map (e.g. Dependencias Federales), handle gracefully
        if state not in prop_map:
            urban_p = 1.0 # Default to Urban

        rural_target = quota * rural_p
        urban_target = quota * urban_p

        # Rounding split
        r_q = int(rural_target)
        u_q = int(urban_target)
        rem = quota - (r_q + u_q)

        # Assign remainder to larger fraction
        if (rural_target % 1) >= (urban_target % 1):
             r_q += rem
        else:
             u_q += rem

        # Distribute
        for tipo, q_sub in [('Rural', r_q), ('Urbano', u_q)]:
            if q_sub > 0:
                munis = selection_map.get(state, {}).get(tipo, [])

                # Fallback if no munis found but quota exists
                if not munis:
                    other_tipo = 'Urbano' if tipo == 'Rural' else 'Rural'
                    munis = selection_map.get(state, {}).get(other_tipo, [])
                    if not munis:
                        munis = [f"Sin Municipio ({orig_state_name})"]

                k = len(munis)
                base = q_sub // k
                rem_sub = q_sub % k

                # Random distribution of remainder
                indices = np.random.permutation(k)

                for i in range(k):
                    m_idx = indices[i]
                    val = base + 1 if i < rem_sub else base

                    if val > 0:
                        output_rows.append({
                            'Estado': orig_state_name,
                            'Municipio': munis[m_idx],
                            'Tipo': tipo,
                            'Género': gender,
                            'Grupo de Edad': age,
                            'Cuota': int(val)
                        })

    df_output = pd.DataFrame(output_rows)
    if not df_output.empty:
        df_output = df_output.sort_values(['Estado', 'Tipo', 'Municipio', 'Género', 'Grupo de Edad'])

    return df_output

def generate_metrics_sheet(total_sample, total_population):
    Z = 1.96
    p = 0.5
    N = total_population

    fpc = np.sqrt((N - total_sample) / (N - 1)) if N > total_sample else 1
    moe = Z * np.sqrt((p * (1-p)) / total_sample) * fpc

    metrics = [
        {'Métrica': 'Tamaño de Muestra Total (n)', 'Valor': total_sample},
        {'Métrica': 'Población Total (N)', 'Valor': N},
        {'Métrica': 'Nivel de Confianza', 'Valor': '95% (Z=1.96)'},
        {'Métrica': 'Margen de Error (MoE)', 'Valor': f"{moe:.2%} (+/- {moe*100:.2f}%)"},
        {'Métrica': 'Proporción Asumida (p)', 'Valor': 0.5},
        {'Métrica': 'Efecto de Diseño (Deff)', 'Valor': '1.5 (Estimado)'},
        {'Métrica': 'Nota 1', 'Valor': 'Selección de municipios aleatoria (Uniforme) por falta de datos poblacionales a nivel municipal.'},
        {'Métrica': 'Nota 2', 'Valor': 'Cuotas ajustadas para sumar exactamente 3000.'}
    ]

    return pd.DataFrame(metrics)

def main():
    try:
        df_design, df_rural = load_data()

        total_pop = df_design['poblacion'].sum()

        df_field_plan = generate_field_plan(df_design, df_rural)

        actual_total = df_field_plan['Cuota'].sum() if not df_field_plan.empty else 0
        df_metrics = generate_metrics_sheet(actual_total, total_pop)

        output_file = 'plan_de_campo.xlsx'
        with pd.ExcelWriter(output_file) as writer:
            if not df_field_plan.empty:
                df_field_plan.to_excel(writer, sheet_name='Plan de Campo', index=False)
            df_metrics.to_excel(writer, sheet_name='Métricas', index=False)

        print(f"Field design saved to {output_file}")
        print(f"Total Quota Generated: {actual_total}")

    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
