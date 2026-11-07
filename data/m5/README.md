# M5 data (not included)

The raw M5 files are not redistributed in this repository. They are Kaggle
competition data, licensed "subject to competition rules" (non-commercial use, with
access limited to people who accepted the rules), and two of the files are larger
than GitHub's 100 MB limit.

They are needed only for the calibration checks (`code/calibrate_effect_sizes.py`,
`code/m5_stats.py`, `code/forecasting_utils.py`). None of the experiments reads them.

## Download

1. Sign in to Kaggle, open the
   [M5 Forecasting – Accuracy data page](https://www.kaggle.com/competitions/m5-forecasting-accuracy/data)
   and accept the competition rules.
2. Download `calendar.csv`, `sell_prices.csv` and `sales_train_evaluation.csv` into
   this folder, from the page or with the Kaggle CLI:

   ```bash
   kaggle competitions download -c m5-forecasting-accuracy -p data/m5
   ```

   then unzip those three files here.

3. Check that you have the same versions as the project:

   ```bash
   python -c "import hashlib; [print(hashlib.sha256(open(f'data/m5/{n}', 'rb').read()).hexdigest(), n) for n in ['calendar.csv', 'sales_train_evaluation.csv', 'sell_prices.csv']]"
   ```

| File | Size (bytes) | Lines incl. header | SHA-256 |
|---|---:|---:|---|
| `calendar.csv` | 103,469 | 1,970 | `d12b5914ef03e66649adf5dd9e996e6602251c22b7a6af8f1f7e3aa12f8860f5` |
| `sales_train_evaluation.csv` | 121,736,518 | 30,491 | `4b4a47c44c38380d2a9168216fea8c9ff2f31b1ddb772f8a0995952a038b8aa0` |
| `sell_prices.csv` | 203,395,785 | 6,841,122 | `9da3ad1f8b8ccacdbdc70612191dd375ec24a4ac6625c24b75b3bc60b0bed2ef` |

The calibration uses store CA_3 and category FOODS: 1,437 products, 277 weeks and
322,166 product-week observations after daily sales are summed to weeks.
