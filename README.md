# WarEraTradingBuddy

WarEraTradingBuddy downloads public WarEra market history, trains one GRU price model per item, and can send ntfy alerts when a forecast indicates a possible turning point.

It is decision-support software. It does not place, cancel, or modify in-game orders. Forecasts can be wrong, and the displayed signal-strength score is a heuristic rather than a calibrated probability.

## What changed

- The current day's market sheet is refreshed on every run instead of being frozen at the first observation.
- The missing `predict.py` runtime is included, so a fresh clone can train and run.
- Direct production-chain markets are supplied as lagged context during training. For example, Iron and Steel can inform one another without future-price leakage.
- Generated history, models, graphs, virtual environments, and secrets are excluded from Git.
- Runtime settings come from environment variables.

Models trained before the related-market feature was added still load. Run `train.py` again to train five-feature models that use the new context.

## Requirements

- Python 3.10 or newer
- Internet access to the public Google Sheets history and ntfy

Create an environment and install dependencies:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

On macOS or Linux, activate with `source .venv/bin/activate`.

## Setup

1. Run `python download_history.py` to download up to 365 daily market sheets into `price_history/`.
2. Review the training constants near the top of `train.py`. The defaults are suitable for a modest CPU, although training every market can take time.
3. Run `python train.py`. Checkpoints are written to `price_models/`.
4. Install the [ntfy app](https://ntfy.sh/) and subscribe to a private, hard-to-guess topic name.
5. Set that topic before starting the notifier.

```powershell
$env:NTFY_TOPIC = "WarEraTrading-your-private-topic"
python main.py
```

`main.py` runs immediately and then checks again on each hour.

## Configuration

| Variable | Default | Purpose |
| --- | ---: | --- |
| `NTFY_TOPIC` | unset | Required ntfy topic name |
| `NTFY_PRIORITY` | `3` | ntfy priority from 1 to 5 |
| `PREDICTION_DAYS` | `3` | Forecast horizon |
| `MIN_HISTORY_DAYS` | `30` | Consecutive days loaded for inference |
| `TREND_DAYS` | `2` | Recent observations used to identify direction |
| `MIN_SIGNAL_STRENGTH` | `0.95` | Minimum heuristic score needed for an alert |

## Data and model notes

- History comes from the public `Economy_History_YYYY-MM-DD` Google Sheets tabs configured in `download_history.py`.
- A complete current-day CSV is written to a temporary file before replacing the prior snapshot.
- Related-market context uses only movements visible on each historical date. Unknown future related-market movement is neutral during recursive prediction.
- The model predicts market midpoints, not executable fills. Always inspect the order book, spread, available quantity, and fees before acting.
- Red/Elite Case coverage depends on the upstream dataset and may be absent.

## Tests

```powershell
python -m unittest discover -s tests
```

## Next improvements

- Add walk-forward backtesting and calibrate signal scores against observed outcomes.
- Add order-book spread and depth from an approved public data source.
- Add reliable Red/Elite Case history if the source dataset begins publishing it.
