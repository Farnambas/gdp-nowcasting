#!/usr/bin/env python3
"""
GDP Nowcast - Command Line Tool
================================

Automatically fetch latest economic data and predict GDP growth.

Usage:
    # First time: Train and save the model
    python nowcast.py --train

    # After training: Just run nowcast (fast!)
    python nowcast.py

    # Retrain with fresh data
    python nowcast.py --retrain

    # Show feature importance
    python nowcast.py --features

Environment:
    Set FRED_API_KEY environment variable with your FRED API key.
    Get one free at: https://fred.stlouisfed.org/docs/api/api_key.html
"""

import os
import sys
import argparse
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.nowcaster import GDPNowcaster

# Default model path
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'models', 'gdp_nowcaster.pkl')


def print_banner():
    """Print the nowcast banner."""
    print("""
╔══════════════════════════════════════════════════════════════════╗
║                     GDP NOWCASTING SYSTEM                        ║
║                                                                  ║
║  Real-time GDP growth prediction using 30+ economic indicators  ║
╚══════════════════════════════════════════════════════════════════╝
    """)


def train_model(api_key: str, save_path: str = MODEL_PATH):
    """Train a new model and save it."""
    print_banner()
    print("TRAINING NEW MODEL")
    print("="*60)

    nowcaster = GDPNowcaster(api_key=api_key)
    metrics = nowcaster.train(start_date='2000-01-01', test_size=8)

    # Show top features
    print("\n" + "="*60)
    print("TOP 20 PREDICTIVE FEATURES")
    print("="*60)
    importance = nowcaster.get_feature_importance(20)
    for i, (_, row) in enumerate(importance.iterrows(), 1):
        print(f"  {i:2}. {row['feature']:<35} {row['importance']:.4f}")

    # Save model
    nowcaster.save(save_path)

    print("\n" + "="*60)
    print("MODEL READY FOR NOWCASTING")
    print("="*60)
    print(f"\nRun 'python nowcast.py' to generate predictions")

    return nowcaster


def run_nowcast(api_key: str, model_path: str = MODEL_PATH):
    """Load model and run nowcast."""
    print_banner()

    if not os.path.exists(model_path):
        print("ERROR: No trained model found!")
        print(f"Expected at: {model_path}")
        print("\nRun 'python nowcast.py --train' first to train a model.")
        sys.exit(1)

    # Load model
    nowcaster = GDPNowcaster.load(model_path)

    # Override API key if provided
    if api_key:
        nowcaster.api_key = api_key
        nowcaster.fred = __import__('fredapi').Fred(api_key=api_key)

    # Run nowcast
    result = nowcaster.nowcast()

    # Print detailed results
    print("\n" + "="*60)
    print("NOWCAST SUMMARY")
    print("="*60)
    print(f"""
    Date Generated:     {result['nowcast_date']}

    ┌─────────────────────────────────────────────┐
    │  TARGET: {result['target_quarter']:<20}              │
    │  PREDICTED GDP GROWTH: {result['predicted_gdp_growth']:>+6.2f}%            │
    │  CONFIDENCE: {result['confidence_band']:<20}        │
    └─────────────────────────────────────────────┘

    Last Known Data:
      • {result['last_actual_quarter']}: {result['last_actual_gdp_growth']:.2f}% (actual)

    Model Info:
      • Trained RMSE: {result['model_rmse']:.4f}%
      • Indicators: {len(nowcaster.indicators_loaded)}
      • Features: {len(nowcaster.feature_cols)}
    """)

    return result


def show_features(model_path: str = MODEL_PATH):
    """Show feature importance from saved model."""
    print_banner()

    if not os.path.exists(model_path):
        print("ERROR: No trained model found!")
        sys.exit(1)

    nowcaster = GDPNowcaster.load(model_path)

    print("\n" + "="*60)
    print("FEATURE IMPORTANCE RANKING")
    print("="*60)

    importance = nowcaster.get_feature_importance(30)

    print(f"\n{'Rank':<6} {'Feature':<40} {'Importance':<12} {'Bar'}")
    print("-"*70)

    for i, (_, row) in enumerate(importance.iterrows(), 1):
        bar = '█' * int(row['importance'] * 150)
        print(f"{i:<6} {row['feature']:<40} {row['importance']:<12.4f} {bar}")

    # Group by category
    print("\n" + "="*60)
    print("IMPORTANCE BY CATEGORY")
    print("="*60)

    categories = {
        'Trade/Ports': ['IMP', 'EXP', 'BOP', 'trade'],
        'Labor': ['UNRATE', 'PAYEMS', 'ICSA', 'CCSA', 'JTS', 'AWH', 'labor'],
        'Consumer': ['PCE', 'RSAFS', 'UMCSENT', 'DSPI', 'CSC'],
        'Industrial': ['INDPRO', 'TCU', 'DGORDER', 'NEWORDER', 'AMT', 'IPMAN', 'BUSINV'],
        'Housing': ['HOUST', 'PERMIT', 'HSN', 'EXH', 'MSP', 'housing'],
        'Financial': ['FED', 'T10Y', 'BAA', 'SP500', 'VIX', 'DTW', 'yield'],
        'GDP Lags': ['gdp_growth_lag'],
    }

    for cat, keywords in categories.items():
        cat_importance = importance[
            importance['feature'].apply(
                lambda x: any(k in x.upper() for k in [k.upper() for k in keywords])
            )
        ]['importance'].sum()
        bar = '█' * int(cat_importance * 50)
        print(f"  {cat:<15} {cat_importance:>6.1%} {bar}")


def main():
    parser = argparse.ArgumentParser(
        description='GDP Nowcasting System',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python nowcast.py --train      # Train new model
    python nowcast.py              # Run nowcast with saved model
    python nowcast.py --features   # Show feature importance

Environment Variables:
    FRED_API_KEY    Your FRED API key (required)
        """
    )

    parser.add_argument(
        '--train', action='store_true',
        help='Train a new model from scratch'
    )
    parser.add_argument(
        '--retrain', action='store_true',
        help='Retrain the model with latest data'
    )
    parser.add_argument(
        '--features', action='store_true',
        help='Show feature importance from trained model'
    )
    parser.add_argument(
        '--model-path', type=str, default=MODEL_PATH,
        help=f'Path to model file (default: {MODEL_PATH})'
    )
    parser.add_argument(
        '--api-key', type=str, default=None,
        help='FRED API key (or set FRED_API_KEY env var)'
    )

    args = parser.parse_args()

    # Get API key
    api_key = args.api_key or os.getenv('FRED_API_KEY')
    if not api_key:
        print("ERROR: FRED API key required!")
        print("\nSet it via:")
        print("  export FRED_API_KEY=your_key_here")
        print("  or")
        print("  python nowcast.py --api-key your_key_here")
        print("\nGet a free key at: https://fred.stlouisfed.org/docs/api/api_key.html")
        sys.exit(1)

    # Run appropriate command
    if args.train or args.retrain:
        train_model(api_key, args.model_path)
    elif args.features:
        show_features(args.model_path)
    else:
        run_nowcast(api_key, args.model_path)


if __name__ == '__main__':
    main()
