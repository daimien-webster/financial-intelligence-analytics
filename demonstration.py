"""
Financial Intelligence Analytics - Demonstration Script
Author: Daimien Webster

This script demonstrates the complete workflow for data quality assessment
and suspicious pattern detection using the framework.
"""

import pandas as pd
import matplotlib.pyplot as plt
from data_quality_assessment import TransactionDataQuality, calculate_quality_trends
from pattern_detection import SuspiciousPatternDetector

def main():
    print("="*70)
    print("Financial Intelligence Data Analytics Framework - Demonstration")
    print("="*70)
    print()
    
    # Load sample dataset
    print("1. Loading sample transaction dataset...")
    df = pd.read_csv('transactions_medium_quality.csv')
    print(f"   Loaded {len(df):,} transaction records")
    print(f"   Date range: {df['transaction_date'].min()} to {df['transaction_date'].max()}")
    print()
    
    # Data Quality Assessment
    print("2. Performing comprehensive data quality assessment...")
    print("-" * 70)
    assessor = TransactionDataQuality(df)
    
    # Run assessments
    completeness = assessor.assess_completeness()
    accuracy = assessor.assess_accuracy()
    consistency = assessor.assess_consistency()
    timeliness = assessor.assess_timeliness()
    
    # Generate report
    quality_report = assessor.generate_quality_report()
    print("\nData Quality Report:")
    print(quality_report.to_string(index=False))
    print()
    
    # Calculate overall quality score
    overall_score = quality_report['Score (%)'].mean()
    print(f"Overall Data Quality Score: {overall_score:.1f}%")
    print()
    
    # Identify issues
    if assessor.issues:
        print(f"Issues Identified: {len(assessor.issues)}")
        print("\nTop 5 Issues:")
        for i, issue in enumerate(assessor.issues[:5], 1):
            print(f"   {i}. {issue}")
    print()
    
    # Visualize quality scores
    print("3. Generating quality visualization...")
    assessor.visualize_quality_scores(save_path='quality_assessment.png')
    print("   Saved: quality_assessment.png")
    print()
    
    # Pattern Detection
    print("4. Detecting suspicious transaction patterns...")
    print("-" * 70)
    detector = SuspiciousPatternDetector(df, threshold_amount=10000)
    
    # Detect structuring
    structuring = detector.detect_structuring(window_days=30, tolerance=0.1)
    print(f"\nStructuring Alerts: {len(structuring)}")
    if len(structuring) > 0:
        print("\nTop 3 Structuring Risks:")
        for idx, row in structuring.head(3).iterrows():
            print(f"   Account: {row['account']}")
            print(f"   Transactions: {row['num_transactions']}")
            print(f"   Total Amount: ${row['total_amount']:,.2f}")
            print(f"   Risk Score: {row['risk_score']:.1f}")
            print()
    
    # Detect rapid movement
    rapid = detector.detect_rapid_movement(max_hours=24)
    print(f"Rapid Movement Chains: {len(rapid)}")
    if len(rapid) > 0:
        print(f"   Average chain length: {rapid['chain_length'].mean():.1f}")
        print(f"   Average time: {rapid['time_hours'].mean():.1f} hours")
    print()
    
    # Unusual velocity
    velocity = detector.detect_unusual_velocity(window_hours=24, threshold_count=10)
    print(f"Unusual Velocity Alerts: {len(velocity)}")
    print()
    
    # Round amount analysis
    round_stats = detector.analyze_round_amounts()
    print("Round Number Analysis:")
    print(f"   Exact thousands: {round_stats['pct_exact_thousands']:.1f}%")
    print(f"   Exact hundreds: {round_stats['pct_exact_hundreds']:.1f}%")
    print()
    
    # Generate pattern visualizations
    print("5. Generating pattern visualization...")
    detector.visualize_patterns(save_path='pattern_analysis.png')
    print("   Saved: pattern_analysis.png")
    print()
    
    # Quality Trends
    print("6. Calculating quality trends over time...")
    trends = calculate_quality_trends(df, date_column='report_date')
    print(f"\nQuality Trends ({len(trends)} periods):")
    print(trends[['period', 'record_count', 'overall_quality']].to_string(index=False))
    print()
    
    # Export issues
    print("7. Exporting detailed issues log...")
    assessor.export_issues_log('data_quality_issues.csv')
    print()
    
    # Summary
    print("="*70)
    print("ANALYSIS COMPLETE")
    print("="*70)
    print("\nGenerated Files:")
    print("   - quality_assessment.png (Data quality dashboard)")
    print("   - pattern_analysis.png (Suspicious pattern analysis)")
    print("   - data_quality_issues.csv (Detailed issues log)")
    print()
    print(f"Overall Assessment:")
    print(f"   Quality Score: {overall_score:.1f}%")
    print(f"   Data Issues: {len(assessor.issues)}")
    print(f"   Structuring Alerts: {len(structuring)}")
    print(f"   Rapid Movement: {len(rapid)} chains")
    print(f"   Velocity Alerts: {len(velocity)}")
    print()
    print("This framework demonstrates automated assessment and pattern detection")
    print("capabilities supporting financial intelligence and regulatory operations.")
    print()

if __name__ == "__main__":
    main()
