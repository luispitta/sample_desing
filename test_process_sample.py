import unittest
import pandas as pd
import numpy as np
from process_sample import normalize_text, process_data, calculate_errors_and_estimates

class TestProcessSample(unittest.TestCase):

    def test_normalize_text(self):
        self.assertEqual(normalize_text("Miranda"), "miranda")
        self.assertEqual(normalize_text("  MÍranda  "), "miranda")
        self.assertEqual(normalize_text(None), "")

    def test_process_data_simple(self):
        # Create dummy data
        df_sample = pd.DataFrame({
            'estados': ['StateA', 'StateA'],
            'municipio': ['Muni1', 'Muni2'],
            'genero': ['Masculino', 'Femenino'],
            'edad': ['18-25 años', '26-35 años'], # Should map to 18-29, 30-49
            'deseo_venezolano': ['Yes', 'No']
        })

        df_design = pd.DataFrame({
            'estado': ['StateA', 'StateA', 'StateA', 'StateA'],
            'genero': ['Masculino', 'Femenino', 'Masculino', 'Femenino'],
            'grupo_edad': ['De 18 a 29 años', 'De 30 a 49 años', 'De 30 a 49 años', 'De 18 a 29 años'],
            'poblacion': [100, 100, 100, 100]
        })

        df_rural = pd.DataFrame({
            'ESTADO': ['StateA', 'StateA'],
            'MUNICIPIO': ['Muni1', 'Muni2'],
            'TIPO': ['Urbano', 'Rural']
        })

        # Run process
        # Note: Need to mock or ensure mapping works
        # normalize_text handles simple cases

        df_final = process_data(df_sample, df_design, df_rural)

        # Check columns
        self.assertIn('weight', df_final.columns)
        self.assertIn('stratum_pop', df_final.columns)

        # Check specific weight logic
        # StateA has 2 munis: 1 Urban, 1 Rural.
        # So each type prop is 0.5.
        # Design Populations:
        # (Masculino, 18-29): 100. Split 50 Urban, 50 Rural.
        # Sample: (Masculino, 18-25) -> (Masculino, 18-29). Muni1 -> Urban.
        # Target Pop for (StateA, Masculino, 18-29, Urbano) = 100 * 0.5 = 50.
        # Sample Count = 1.
        # Weight = 50 / 1 = 50.

        row1 = df_final.iloc[0]
        self.assertAlmostEqual(row1['weight'], 50.0)

        # Row 2: (Femenino, 26-35) -> (Femenino, 30-49). Muni2 -> Rural.
        # Target Pop for (StateA, Femenino, 30-49, Rural) = 100 * 0.5 = 50.
        # Sample Count = 1.
        # Weight = 50.

        row2 = df_final.iloc[1]
        self.assertAlmostEqual(row2['weight'], 50.0)

    def test_calculate_errors(self):
        # Create weighted df
        df = pd.DataFrame({
            'deseo_venezolano': ['Yes', 'Yes', 'No'],
            'weight': [10, 10, 20]
        })

        results = calculate_errors_and_estimates(df)
        res = results['deseo_venezolano']

        # Yes: weight 20. Total 40. Prop 0.5.
        # No: weight 20. Total 40. Prop 0.5.

        yes_row = res[res['deseo_venezolano'] == 'Yes'].iloc[0]
        self.assertAlmostEqual(yes_row['proportion'], 0.5)

        # SE calculation check
        # p=0.5.
        # Sum sq weights for Yes: 10^2 + 10^2 = 200.
        # Sum sq weights total: 10^2 + 10^2 + 20^2 = 100 + 100 + 400 = 600.
        # Sum sq out = 400.
        # Var num = (1-0.5)^2 * 200 + (0.5)^2 * 400
        # = 0.25 * 200 + 0.25 * 400 = 50 + 100 = 150.
        # Var = 150 / 40^2 = 150 / 1600 = 0.09375.
        # SE = sqrt(0.09375) approx 0.306.

        self.assertAlmostEqual(yes_row['SE'], np.sqrt(0.09375))

if __name__ == '__main__':
    unittest.main()
