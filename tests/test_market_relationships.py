import unittest

try:
    import pandas as pd
except ImportError:
    pd = None

if pd is not None:
    from market_relationships import normalize_item_name, related_percentage_changes


@unittest.skipIf(pd is None, "pandas is not installed")
class MarketRelationshipTests(unittest.TestCase):
    def test_normalizes_display_names(self):
        self.assertEqual(normalize_item_name("Cooked Fish"), "cookedfish")

    def test_iron_receives_only_historical_steel_movement(self):
        history = pd.DataFrame(
            [
                {"Date": "2026-09-01", "Item": "Steel", "Market Midpoint": 2.0},
                {"Date": "2026-09-02", "Item": "Steel", "Market Midpoint": 2.2},
                {"Date": "2026-09-01", "Item": "Wood", "Market Midpoint": 1.0},
                {"Date": "2026-09-02", "Item": "Wood", "Market Midpoint": 5.0},
            ]
        )

        changes = related_percentage_changes(history, "Iron")

        self.assertAlmostEqual(changes.loc[pd.Timestamp("2026-09-02")], 0.1)


if __name__ == "__main__":
    unittest.main()
