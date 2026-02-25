import pandas as pd
import numpy as np
import unicodedata
import plotly.express as px
import plotly.graph_objects as go

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

def load_data():
    print("Loading data...")
    try:
        df_sample = pd.read_csv('sample_data.csv', sep=';')
        df_design = pd.read_excel('diseño_muestral.xlsx')
        df_rural = pd.read_excel('RURAL_URBANO.xlsx')
    except Exception as e:
        print(f"Error loading files: {e}")
        raise
    return df_sample, df_design, df_rural

def process_data(df_sample, df_design, df_rural):
    print("Processing and Normalizing data...")

    # --- Normalization ---

    # Normalize State Names in all DFs
    df_sample['estado_norm'] = df_sample['estados'].apply(normalize_text)
    df_design['estado_norm'] = df_design['estado'].apply(normalize_text)
    df_rural['ESTADO_NORM'] = df_rural['ESTADO'].apply(normalize_text)

    # Normalize Municipality Names
    df_sample['municipio_norm'] = df_sample['municipio'].apply(normalize_text)
    df_rural['MUNICIPIO_NORM'] = df_rural['MUNICIPIO'].apply(normalize_text)

    # Manual Mapping for known discrepancies
    # Municipality Mappings (specifically for Sample -> Rural match)

    def map_muni(row):
        muni = row['municipio_norm']
        state = row['estado_norm']
        # 'antonio jose de sucre' in Miranda (Sample) -> 'sucre' in Miranda (Rural)
        if state == 'miranda' and ('antonio jose de sucre' in muni):
            return 'sucre'
        return muni

    df_sample['municipio_mapped'] = df_sample.apply(map_muni, axis=1)

    # Age Group Mapping
    # Sample: '18-25 años', '26-35 años', '36-45 años', '46-55 años', '56-75 años', '76 años o más'
    # Design: 'De 18 a 29 años', 'De 30 a 49 años', 'De 50 a 69 años', 'De 70 años en adelante'

    age_map = {
        '18-25 años': 'De 18 a 29 años',
        '26-35 años': 'De 30 a 49 años', # Approximation
        '36-45 años': 'De 30 a 49 años',
        '46-55 años': 'De 50 a 69 años', # Approximation
        '56-75 años': 'De 50 a 69 años', # Approximation
        '76 años o más': 'De 70 años en adelante'
    }

    # Strip whitespace from sample age before mapping
    df_sample['edad_clean'] = df_sample['edad'].str.strip()
    df_sample['grupo_edad_norm'] = df_sample['edad_clean'].map(age_map)

    # Verify Age Mapping
    missing_age = df_sample[df_sample['grupo_edad_norm'].isna()]['edad'].unique()
    if len(missing_age) > 0:
        print(f"Warning: Unmapped ages found: {missing_age}")

    # Normalize Gender
    df_sample['genero_norm'] = df_sample['genero'].str.title().str.strip()
    df_design['genero_norm'] = df_design['genero'].str.title().str.strip()

    # --- Merge Rural/Urban Classification to Sample ---
    # Lookup using mapped municipality
    rural_lookup = df_rural[['ESTADO_NORM', 'MUNICIPIO_NORM', 'TIPO']].drop_duplicates()

    df_sample = pd.merge(df_sample, rural_lookup,
                         left_on=['estado_norm', 'municipio_mapped'],
                         right_on=['ESTADO_NORM', 'MUNICIPIO_NORM'],
                         how='left')

    # Check for missing TIPO
    missing_tipo = df_sample[df_sample['TIPO'].isna()]
    if not missing_tipo.empty:
        print(f"Warning: {len(missing_tipo)} sample records could not be classified as Rural/Urbano.")
        # Fill with 'Urbano' as fallback for this specific context (Petare/Caracas)
        df_sample['TIPO'] = df_sample['TIPO'].fillna('Urbano')

    # --- Stratification & Weighting ---

    # 1. Calculate Target Population for each Stratum (State, Gender, Age, Type)

    # Note: Rural/Urban population split is estimated based on the proportion of municipalities of each type.
    # This assumes uniform population distribution across municipalities, which is a simplification.
    # Ideally, precise population counts for Rural/Urban strata should be used if available.

    # Count munis per state and type
    muni_counts = df_rural.groupby(['ESTADO_NORM', 'TIPO']).size().reset_index(name='n_munis')
    state_muni_totals = df_rural.groupby('ESTADO_NORM').size().reset_index(name='total_munis')

    muni_props = pd.merge(muni_counts, state_muni_totals, on='ESTADO_NORM')
    muni_props['type_prop'] = muni_props['n_munis'] / muni_props['total_munis']

    # Merge proportions into Design
    df_design_expanded = pd.merge(df_design, muni_props[['ESTADO_NORM', 'TIPO', 'type_prop']],
                                  left_on='estado_norm', right_on='ESTADO_NORM', how='left')

    # Calculate Stratum Population
    df_design_expanded['stratum_pop'] = df_design_expanded['poblacion'] * df_design_expanded['type_prop']
    df_design_expanded['stratum_pop'] = df_design_expanded['stratum_pop'].fillna(0)

    # 2. Calculate Sample Counts per Stratum
    # Keys: estado_norm, genero_norm, grupo_edad_norm, TIPO
    sample_counts = df_sample.groupby(['estado_norm', 'genero_norm', 'grupo_edad_norm', 'TIPO']).size().reset_index(name='sample_count')

    # 3. Calculate Weights
    # Merge Target Population with Sample Counts
    df_weights = pd.merge(df_design_expanded, sample_counts,
                          left_on=['estado_norm', 'genero_norm', 'grupo_edad', 'TIPO'],
                          right_on=['estado_norm', 'genero_norm', 'grupo_edad_norm', 'TIPO'],
                          how='inner')

    df_weights['weight'] = df_weights['stratum_pop'] / df_weights['sample_count']

    # Merge weights back to sample
    df_final = pd.merge(df_sample, df_weights[['estado_norm', 'genero_norm', 'grupo_edad', 'TIPO', 'weight', 'stratum_pop']],
                        left_on=['estado_norm', 'genero_norm', 'grupo_edad_norm', 'TIPO'],
                        right_on=['estado_norm', 'genero_norm', 'grupo_edad', 'TIPO'],
                        how='left')

    print(f"Weights calculated. Mean weight: {df_final['weight'].mean()}")

    return df_final

def calculate_errors_and_estimates(df_weighted):
    print("Calculating estimates and errors...")

    # Questions to analyze
    questions = ['deseo_venezolano', 'liderazgo_pais', 'presos_politicos', 'intencion',
                 'problemas_afecta', 'resolver_problema', 'palabra_definiria']

    results = {}

    for q in questions:
        if q not in df_weighted.columns:
            continue

        df_q = df_weighted[~df_weighted[q].isna()].copy()

        total_weight = df_q['weight'].sum()

        stats = df_q.groupby(q).apply(lambda x: pd.Series({
            'weighted_count': x['weight'].sum(),
            'raw_count': len(x),
            'sum_sq_weights': (x['weight']**2).sum()
        })).reset_index()

        stats['proportion'] = stats['weighted_count'] / total_weight

        def calc_se(row):
            p_hat = row['proportion']
            sum_sq_in = row['sum_sq_weights']
            sum_sq_total = df_q['weight'].pow(2).sum()
            sum_sq_out = sum_sq_total - sum_sq_in

            var_num = ((1 - p_hat)**2 * sum_sq_in) + (p_hat**2 * sum_sq_out)
            var = var_num / (total_weight**2)
            return np.sqrt(var)

        stats['SE'] = stats.apply(calc_se, axis=1)
        stats['MoE_90'] = stats['SE'] * 1.645 # Z for 90%

        results[q] = stats.sort_values('proportion', ascending=False)

    return results

def generate_dashboard(results):
    print("Generating dashboard...")

    html_content = """
    <html>
    <head>
        <title>Survey Results Dashboard</title>
        <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
        <style>
            body { font-family: Arial, sans-serif; margin: 20px; }
            .chart-container { margin-bottom: 50px; }
            h1 { color: #333; }
            h2 { color: #555; border-bottom: 1px solid #ccc; padding-bottom: 5px; }
        </style>
    </head>
    <body>
        <h1>Survey Results Dashboard</h1>
        <p>Weighted estimates with 90% Confidence Intervals (Margin of Error).</p>
    """

    for q, stats in results.items():
        # Create Plotly figure
        fig = px.bar(stats, x=q, y='proportion',
                     error_y='MoE_90',
                     text_auto='.1%',
                     title=f"Results for: {q}",
                     labels={'proportion': 'Weighted Proportion', q: 'Response'})

        fig.update_layout(yaxis_tickformat='.0%')

        # Convert to HTML div
        div = fig.to_html(full_html=False, include_plotlyjs=False)

        html_content += f"""
        <div class="chart-container">
            <h2>{q.replace('_', ' ').title()}</h2>
            {div}
            <details>
                <summary>View Data Table</summary>
                {stats.to_html(classes='table', float_format=lambda x: f'{x:.4f}')}
            </details>
        </div>
        """

    html_content += """
    </body>
    </html>
    """

    with open('dashboard.html', 'w') as f:
        f.write(html_content)
    print("Dashboard saved to dashboard.html")

def main():
    df_sample, df_design, df_rural = load_data()

    df_weighted = process_data(df_sample, df_design, df_rural)

    # Save weighted data
    df_weighted.to_csv('final_predictions.csv', index=False)
    print("Weighted data saved to final_predictions.csv")

    results = calculate_errors_and_estimates(df_weighted)

    generate_dashboard(results)

if __name__ == "__main__":
    main()
