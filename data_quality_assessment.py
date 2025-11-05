"""
Financial Transaction Data Quality Assessment Framework
Author: Daimien Webster
Purpose: Automated data quality assessment for financial transaction reporting

This module provides functions to assess data quality across key dimensions
relevant to financial intelligence and regulatory reporting.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple

class TransactionDataQuality:
    """
    Assess data quality of financial transaction datasets across multiple dimensions.
    
    Quality dimensions assessed:
    - Completeness: Missing values and required field coverage
    - Accuracy: Data type conformance and range validation
    - Consistency: Cross-field logical validation
    - Timeliness: Reporting lag analysis
    """
    
    def __init__(self, df: pd.DataFrame):
        """
        Initialize with transaction dataset.
        
        Args:
            df: DataFrame containing transaction data
        """
        self.df = df
        self.quality_scores = {}
        self.issues = []
        
    def assess_completeness(self) -> Dict[str, float]:
        """
        Assess completeness of required fields.
        
        Returns:
            Dictionary mapping field names to completeness percentages
        """
        required_fields = [
            'transaction_id', 'transaction_date', 'amount', 
            'sender_account', 'receiver_account', 'transaction_type'
        ]
        
        completeness = {}
        for field in required_fields:
            if field in self.df.columns:
                non_null = self.df[field].notna().sum()
                total = len(self.df)
                completeness[field] = (non_null / total) * 100
                
                if completeness[field] < 95:
                    self.issues.append(
                        f"Field '{field}' has low completeness: {completeness[field]:.1f}%"
                    )
            else:
                completeness[field] = 0.0
                self.issues.append(f"Required field '{field}' is missing from dataset")
        
        self.quality_scores['completeness'] = np.mean(list(completeness.values()))
        return completeness
    
    def assess_accuracy(self) -> Dict[str, List[str]]:
        """
        Assess accuracy through data type validation and range checks.
        
        Returns:
            Dictionary of accuracy issues by category
        """
        accuracy_issues = {
            'amount_validation': [],
            'date_validation': [],
            'account_format': []
        }
        
        # Amount validation
        if 'amount' in self.df.columns:
            invalid_amounts = self.df[
                (self.df['amount'] <= 0) | 
                (self.df['amount'] > 10000000)
            ]
            if len(invalid_amounts) > 0:
                accuracy_issues['amount_validation'].append(
                    f"Found {len(invalid_amounts)} transactions with invalid amounts"
                )
        
        # Date validation
        if 'transaction_date' in self.df.columns:
            self.df['transaction_date'] = pd.to_datetime(
                self.df['transaction_date'], 
                errors='coerce'
            )
            invalid_dates = self.df['transaction_date'].isna().sum()
            if invalid_dates > 0:
                accuracy_issues['date_validation'].append(
                    f"Found {invalid_dates} invalid date formats"
                )
            
            # Check for future dates
            future_dates = self.df[
                self.df['transaction_date'] > datetime.now()
            ]
            if len(future_dates) > 0:
                accuracy_issues['date_validation'].append(
                    f"Found {len(future_dates)} transactions with future dates"
                )
        
        # Account format validation (basic pattern check)
        if 'sender_account' in self.df.columns:
            # Assuming account numbers should be alphanumeric, length 8-16
            invalid_accounts = self.df[
                ~self.df['sender_account'].astype(str).str.match(r'^[A-Z0-9]{8,16}$')
            ]
            if len(invalid_accounts) > 0:
                accuracy_issues['account_format'].append(
                    f"Found {len(invalid_accounts)} accounts with invalid format"
                )
        
        total_issues = sum(len(v) for v in accuracy_issues.values())
        accuracy_score = max(0, 100 - (total_issues / len(self.df) * 100))
        self.quality_scores['accuracy'] = accuracy_score
        
        return accuracy_issues
    
    def assess_consistency(self) -> List[str]:
        """
        Assess logical consistency between related fields.
        
        Returns:
            List of consistency issues found
        """
        consistency_issues = []
        
        # Check sender != receiver
        if all(col in self.df.columns for col in ['sender_account', 'receiver_account']):
            same_accounts = self.df[
                self.df['sender_account'] == self.df['receiver_account']
            ]
            if len(same_accounts) > 0:
                consistency_issues.append(
                    f"Found {len(same_accounts)} transactions where sender equals receiver"
                )
        
        # Check amount matches transaction type expectations
        if all(col in self.df.columns for col in ['amount', 'transaction_type']):
            # Large deposits might indicate structuring
            large_cash_deposits = self.df[
                (self.df['transaction_type'] == 'CASH_DEPOSIT') & 
                (self.df['amount'] > 9000) & 
                (self.df['amount'] < 10000)
            ]
            if len(large_cash_deposits) > len(self.df) * 0.05:
                consistency_issues.append(
                    f"High volume of cash deposits just below $10k threshold: "
                    f"{len(large_cash_deposits)} transactions"
                )
        
        consistency_score = max(0, 100 - (len(consistency_issues) / len(self.df) * 1000))
        self.quality_scores['consistency'] = consistency_score
        
        return consistency_issues
    
    def assess_timeliness(self) -> Dict[str, float]:
        """
        Assess reporting timeliness.
        
        Returns:
            Dictionary with timeliness metrics
        """
        timeliness_metrics = {}
        
        if all(col in self.df.columns for col in ['transaction_date', 'report_date']):
            self.df['transaction_date'] = pd.to_datetime(self.df['transaction_date'])
            self.df['report_date'] = pd.to_datetime(self.df['report_date'])
            
            self.df['reporting_lag'] = (
                self.df['report_date'] - self.df['transaction_date']
            ).dt.days
            
            timeliness_metrics['mean_lag_days'] = self.df['reporting_lag'].mean()
            timeliness_metrics['median_lag_days'] = self.df['reporting_lag'].median()
            timeliness_metrics['max_lag_days'] = self.df['reporting_lag'].max()
            
            # Count transactions reported late (>10 business days)
            late_reports = self.df[self.df['reporting_lag'] > 10]
            timeliness_metrics['late_report_percentage'] = (
                len(late_reports) / len(self.df) * 100
            )
            
            timeliness_score = max(0, 100 - timeliness_metrics['late_report_percentage'])
            self.quality_scores['timeliness'] = timeliness_score
        
        return timeliness_metrics
    
    def generate_quality_report(self) -> pd.DataFrame:
        """
        Generate comprehensive data quality report.
        
        Returns:
            DataFrame summarizing quality scores across dimensions
        """
        completeness = self.assess_completeness()
        accuracy = self.assess_accuracy()
        consistency = self.assess_consistency()
        timeliness = self.assess_timeliness()
        
        report = pd.DataFrame({
            'Quality Dimension': list(self.quality_scores.keys()),
            'Score (%)': list(self.quality_scores.values())
        })
        
        report['Status'] = report['Score (%)'].apply(
            lambda x: 'Excellent' if x >= 95 else 
                     'Good' if x >= 85 else 
                     'Fair' if x >= 75 else 
                     'Poor'
        )
        
        return report
    
    def visualize_quality_scores(self, save_path: str = None):
        """
        Create visualization of quality scores.
        
        Args:
            save_path: Optional path to save the figure
        """
        if not self.quality_scores:
            self.generate_quality_report()
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Bar chart of quality scores
        dimensions = list(self.quality_scores.keys())
        scores = list(self.quality_scores.values())
        
        colors = ['#2ecc71' if s >= 85 else '#f39c12' if s >= 75 else '#e74c3c' 
                  for s in scores]
        
        ax1.bar(dimensions, scores, color=colors, alpha=0.7)
        ax1.axhline(y=85, color='r', linestyle='--', alpha=0.5, label='Target (85%)')
        ax1.set_ylabel('Quality Score (%)', fontsize=12)
        ax1.set_title('Data Quality Assessment by Dimension', fontsize=14, fontweight='bold')
        ax1.set_ylim(0, 100)
        ax1.legend()
        
        for i, (dim, score) in enumerate(zip(dimensions, scores)):
            ax1.text(i, score + 2, f'{score:.1f}%', ha='center', fontsize=10)
        
        # Pie chart of overall quality
        overall_score = np.mean(scores)
        remaining = 100 - overall_score
        
        ax2.pie(
            [overall_score, remaining], 
            labels=['Quality Score', 'Improvement Needed'],
            colors=['#3498db', '#ecf0f1'],
            autopct='%1.1f%%',
            startangle=90
        )
        ax2.set_title(
            f'Overall Data Quality: {overall_score:.1f}%', 
            fontsize=14, 
            fontweight='bold'
        )
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def export_issues_log(self, filepath: str):
        """
        Export all identified issues to CSV.
        
        Args:
            filepath: Path where CSV should be saved
        """
        issues_df = pd.DataFrame({
            'Issue': self.issues,
            'Timestamp': datetime.now()
        })
        issues_df.to_csv(filepath, index=False)
        print(f"Issues log exported to {filepath}")


def calculate_quality_trends(df: pd.DataFrame, date_column: str = 'report_date') -> pd.DataFrame:
    """
    Calculate data quality trends over time.
    
    Args:
        df: Transaction dataset
        date_column: Column containing dates for trending
    
    Returns:
        DataFrame with quality trends by month
    """
    df[date_column] = pd.to_datetime(df[date_column])
    df['year_month'] = df[date_column].dt.to_period('M')
    
    trends = []
    
    for period in df['year_month'].unique():
        period_data = df[df['year_month'] == period]
        assessor = TransactionDataQuality(period_data)
        report = assessor.generate_quality_report()
        
        trend_row = {
            'period': str(period),
            'record_count': len(period_data),
            'overall_quality': report['Score (%)'].mean()
        }
        
        for _, row in report.iterrows():
            trend_row[row['Quality Dimension']] = row['Score (%)']
        
        trends.append(trend_row)
    
    return pd.DataFrame(trends)


if __name__ == "__main__":
    # Example usage
    print("Financial Transaction Data Quality Assessment Framework")
    print("=" * 60)
    print("\nThis module provides automated data quality assessment")
    print("for financial intelligence and regulatory reporting datasets.")
    print("\nKey capabilities:")
    print("  - Completeness assessment across required fields")
    print("  - Accuracy validation for amounts, dates, and formats")
    print("  - Consistency checking for logical relationships")
    print("  - Timeliness analysis of reporting lag")
    print("  - Trend analysis over time")
    print("  - Automated issue logging and reporting")
