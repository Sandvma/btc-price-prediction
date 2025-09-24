"""
run_factory.py
===============

Entry point for running the quant strategy factory pipeline.  This
script reads a YAML configuration file, constructs the pipeline and
executes it end to end.  The results are printed to the console and
written to a JSON file alongside the configuration file for record
keeping.

Example usage:

```
python btc_price_prediction/run_factory.py --config btc_price_prediction/config/example.yaml
```
"""

import argparse
import json
from pathlib import Path

import yaml  # type: ignore

from .factory_pipeline import run_pipeline, StrategyMetrics


def main():
    parser = argparse.ArgumentParser(description="Run quant strategy factory pipeline")
    parser.add_argument('--config', type=str, required=True, help='Path to YAML configuration file')
    args = parser.parse_args()
    config_path = Path(args.config)
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    # load YAML
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    # run pipeline
    metrics: StrategyMetrics = run_pipeline(config)
    print("Pipeline completed.")
    print(f"MSE: {metrics.mse:.6f}")
    print(f"Accuracy: {metrics.accuracy:.4f}")
    print(f"Sharpe: {metrics.sharpe:.4f}")
    print(f"Deflated Sharpe: {metrics.deflated_sharpe:.4f}")
    print(f"PBO: {metrics.pbo:.4f}")
    print(f"Net return: {metrics.net_return:.4f}")
    print(f"Max drawdown: {metrics.max_drawdown:.4f}")
    # write results to JSON next to config
    result = {
        'mse': metrics.mse,
        'accuracy': metrics.accuracy,
        'sharpe': metrics.sharpe,
        'deflated_sharpe': metrics.deflated_sharpe,
        'pbo': metrics.pbo,
        'net_return': metrics.net_return,
        'max_drawdown': metrics.max_drawdown,
    }
    output_path = config_path.with_suffix('.results.json')
    with open(output_path, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"Results written to {output_path}")


if __name__ == '__main__':
    main()