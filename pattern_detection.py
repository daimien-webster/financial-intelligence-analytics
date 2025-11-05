"""
Suspicious Transaction Pattern Detection
Author: Daimien Webster
Purpose: Identify potential money laundering patterns in transaction data

This module implements pattern detection algorithms to identify suspicious
transaction behaviors that may indicate money laundering or structuring.
"""

import pandas as pd
import numpy as np
from datetime import timedelta
from typing import List, Dict, Tuple
import matplotlib.pyplot as plt
import seaborn as sns

class SuspiciousPatternDetector:
    """
    Detect suspicious patterns in financial transaction data.
    
    Implements detection for:
    - Structuring (transactions just below reporting thresholds)
    - Rapid movement (layering through multiple quick transfers)
    - Unusual velocity (high volume of transactions in short period)
    - Round amount clustering (preference for round numbers)
    """
    
    def __init__(self, df: pd.DataFrame, threshold_amount: float = 10000):
        """
        Initialize detector with transaction dataset.
        
        Args:
            df: DataFrame containing transaction data
            threshold_amount: Regulatory reporting threshold
        """
        self.df = df.copy()
        self.threshold = threshold_amount
        self.alerts = []
        
        # Ensure date columns are datetime
        if 'transaction_date' in self.df.columns:
            self.df['transaction_date'] = pd.to_datetime(self.df['transaction_date'])
    
    def detect_structuring(self, window_days: int = 30, tolerance: float = 0.1) -> pd.DataFrame:
        """
        Detect potential structuring (smurfing) patterns.
        
        Identifies accounts making multiple transactions just below
        the reporting threshold within a specified time window.
        
        Args:
            window_days: Time window to analyze
            tolerance: Percentage below threshold to consider (0.1 = 10%)
            
        Returns:
            DataFrame of accounts with potential structuring behavior
        """
        lower_bound = self.threshold * (1 - tolerance)
        
        # Filter transactions near threshold
        near_threshold = self.df[
            (self.df['amount'] >= lower_bound) & 
            (self.df['amount'] < self.threshold)
        ].copy()
        
        # Group by sender account and analyze patterns
        structuring_alerts = []
        
        for account in near_threshold['sender_account'].unique():
            account_txns = near_threshold[
                near_threshold['sender_account'] == account
            ].sort_values('transaction_date')
            
            if len(account_txns) >= 3:  # At least 3 transactions
                # Check if within time window
                date_range = (
                    account_txns['transaction_date'].max() - 
                    account_txns['transaction_date'].min()
                ).days
                
                if date_range <= window_days:
                    total_amount = account_txns['amount'].sum()
                    avg_amount = account_txns['amount'].mean()
                    
                    alert = {
                        'account': account,
                        'num_transactions': len(account_txns),
                        'total_amount': total_amount,
                        'avg_amount': avg_amount,
                        'date_range_days': date_range,
                        'risk_score': self._calculate_structuring_risk(
                            len(account_txns), total_amount, date_range
                        )
                    }
                    structuring_alerts.append(alert)
        
        alerts_df = pd.DataFrame(structuring_alerts)
        
        if len(alerts_df) > 0:
            alerts_df = alerts_df.sort_values('risk_score', ascending=False)
            self.alerts.extend(alerts_df.to_dict('records'))
        
        return alerts_df
    
    def detect_rapid_movement(self, max_hours: int = 24) -> pd.DataFrame:
        """
        Detect rapid movement of funds (potential layering).
        
        Identifies sequences where funds move through multiple accounts
        in rapid succession.
        
        Args:
            max_hours: Maximum time between transfers to consider rapid
            
        Returns:
            DataFrame of potential rapid movement chains
        """
        rapid_chains = []
        
        # Sort by date
        sorted_df = self.df.sort_values('transaction_date')
        
        # Group consecutive transactions
        for i in range(len(sorted_df) - 2):
            txn1 = sorted_df.iloc[i]
            txn2 = sorted_df.iloc[i + 1]
            txn3 = sorted_df.iloc[i + 2]
            
            # Check if forms a chain (receiver becomes sender)
            if (txn1['receiver_account'] == txn2['sender_account'] and 
                txn2['receiver_account'] == txn3['sender_account']):
                
                time_diff_1 = (txn2['transaction_date'] - txn1['transaction_date']).total_seconds() / 3600
                time_diff_2 = (txn3['transaction_date'] - txn2['transaction_date']).total_seconds() / 3600
                
                if time_diff_1 <= max_hours and time_diff_2 <= max_hours:
                    chain = {
                        'start_account': txn1['sender_account'],
                        'end_account': txn3['receiver_account'],
                        'chain_length': 3,
                        'total_amount': txn1['amount'],
                        'time_hours': time_diff_1 + time_diff_2,
                        'risk_score': self._calculate_velocity_risk(
                            3, time_diff_1 + time_diff_2
                        )
                    }
                    rapid_chains.append(chain)
        
        chains_df = pd.DataFrame(rapid_chains)
        
        if len(chains_df) > 0:
            chains_df = chains_df.sort_values('risk_score', ascending=False)
        
        return chains_df
    
    def detect_unusual_velocity(self, window_hours: int = 24, threshold_count: int = 10) -> pd.DataFrame:
        """
        Detect accounts with unusually high transaction velocity.
        
        Args:
            window_hours: Time window to analyze
            threshold_count: Minimum transactions to flag
            
        Returns:
            DataFrame of accounts with high velocity
        """
        velocity_alerts = []
        
        for account in self.df['sender_account'].unique():
            account_txns = self.df[
                self.df['sender_account'] == account
            ].sort_values('transaction_date')
            
            if len(account_txns) < threshold_count:
                continue
            
            # Calculate rolling window velocity
            for i in range(len(account_txns)):
                window_start = account_txns.iloc[i]['transaction_date']
                window_end = window_start + timedelta(hours=window_hours)
                
                window_txns = account_txns[
                    (account_txns['transaction_date'] >= window_start) &
                    (account_txns['transaction_date'] <= window_end)
                ]
                
                if len(window_txns) >= threshold_count:
                    alert = {
                        'account': account,
                        'transactions_in_window': len(window_txns),
                        'total_amount': window_txns['amount'].sum(),
                        'window_start': window_start,
                        'risk_score': len(window_txns) / threshold_count
                    }
                    velocity_alerts.append(alert)
                    break  # One alert per account
        
        return pd.DataFrame(velocity_alerts)
    
    def analyze_round_amounts(self) -> Dict[str, float]:
        """
        Analyze preference for round numbers (potential indicator).
        
        Returns:
            Dictionary with round number statistics
        """
        if 'amount' not in self.df.columns:
            return {}
        
        # Check various round number thresholds
        stats = {
            'exact_thousands': (self.df['amount'] % 1000 == 0).sum(),
            'exact_hundreds': (self.df['amount'] % 100 == 0).sum(),
            'exact_fifties': (self.df['amount'] % 50 == 0).sum(),
        }
        
        total = len(self.df)
        stats['pct_exact_thousands'] = (stats['exact_thousands'] / total) * 100
        stats['pct_exact_hundreds'] = (stats['exact_hundreds'] / total) * 100
        
        return stats
    
    def _calculate_structuring_risk(
        self, 
        num_txns: int, 
        total_amount: float, 
        days: int
    ) -> float:
        """Calculate risk score for structuring behavior."""
        # Higher score for more transactions, larger amounts, shorter periods
        frequency_score = min(num_txns / 10, 1.0) * 0.4
        amount_score = min(total_amount / (self.threshold * 5), 1.0) * 0.3
        velocity_score = min(30 / max(days, 1), 1.0) * 0.3
        
        return (frequency_score + amount_score + velocity_score) * 100
    
    def _calculate_velocity_risk(self, chain_length: int, hours: float) -> float:
        """Calculate risk score for rapid movement."""
        length_score = min(chain_length / 5, 1.0) * 0.6
        speed_score = min(24 / max(hours, 1), 1.0) * 0.4
        
        return (length_score + speed_score) * 100
    
    def generate_alert_summary(self) -> pd.DataFrame:
        """
        Generate summary of all alerts.
        
        Returns:
            DataFrame summarizing alerts by type and risk level
        """
        if not self.alerts:
            return pd.DataFrame()
        
        summary = pd.DataFrame(self.alerts)
        summary['risk_level'] = pd.cut(
            summary['risk_score'],
            bins=[0, 30, 60, 100],
            labels=['Low', 'Medium', 'High']
        )
        
        return summary
    
    def visualize_patterns(self, save_path: str = None):
        """
        Create visualizations of detected patterns.
        
        Args:
            save_path: Optional path to save figure
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Amount distribution with threshold line
        ax1 = axes[0, 0]
        amounts = self.df['amount'].dropna()
        ax1.hist(amounts[amounts < self.threshold * 2], bins=50, alpha=0.7, color='#3498db')
        ax1.axvline(self.threshold, color='r', linestyle='--', linewidth=2, 
                    label=f'Threshold: ${self.threshold:,.0f}')
        ax1.axvline(self.threshold * 0.9, color='orange', linestyle='--', linewidth=2,
                    label=f'90% Threshold: ${self.threshold * 0.9:,.0f}')
        ax1.set_xlabel('Transaction Amount ($)', fontsize=11)
        ax1.set_ylabel('Frequency', fontsize=11)
        ax1.set_title('Transaction Amount Distribution', fontsize=13, fontweight='bold')
        ax1.legend()
        
        # 2. Round number analysis
        ax2 = axes[0, 1]
        round_stats = self.analyze_round_amounts()
        categories = ['Exact $1000s', 'Exact $100s', 'Exact $50s']
        percentages = [
            round_stats.get('pct_exact_thousands', 0),
            round_stats.get('pct_exact_hundreds', 0),
            round_stats.get('pct_exact_fifties', 0)
        ]
        ax2.bar(categories, percentages, color=['#e74c3c', '#f39c12', '#95a5a6'], alpha=0.7)
        ax2.set_ylabel('Percentage of Transactions (%)', fontsize=11)
        ax2.set_title('Round Number Preference Analysis', fontsize=13, fontweight='bold')
        ax2.set_ylim(0, max(percentages) * 1.2)
        
        for i, v in enumerate(percentages):
            ax2.text(i, v + 0.5, f'{v:.1f}%', ha='center', fontsize=10)
        
        # 3. Transaction type distribution
        ax3 = axes[1, 0]
        if 'transaction_type' in self.df.columns:
            type_counts = self.df['transaction_type'].value_counts()
            ax3.pie(type_counts.values, labels=type_counts.index, autopct='%1.1f%%',
                    startangle=90, colors=sns.color_palette("Set3"))
            ax3.set_title('Transaction Type Distribution', fontsize=13, fontweight='bold')
        
        # 4. Temporal pattern
        ax4 = axes[1, 1]
        if 'transaction_date' in self.df.columns:
            self.df['hour'] = pd.to_datetime(self.df['transaction_date']).dt.hour
            hourly_counts = self.df['hour'].value_counts().sort_index()
            ax4.plot(hourly_counts.index, hourly_counts.values, marker='o', 
                     color='#2ecc71', linewidth=2, markersize=6)
            ax4.set_xlabel('Hour of Day', fontsize=11)
            ax4.set_ylabel('Transaction Count', fontsize=11)
            ax4.set_title('Transaction Temporal Pattern', fontsize=13, fontweight='bold')
            ax4.grid(alpha=0.3)
            ax4.set_xticks(range(0, 24, 3))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()


if __name__ == "__main__":
    print("Suspicious Transaction Pattern Detection Module")
    print("=" * 60)
    print("\nThis module implements pattern detection for:")
    print("  - Structuring (transactions just below thresholds)")
    print("  - Rapid movement (layering through accounts)")
    print("  - Unusual velocity (high transaction frequency)")
    print("  - Round amount analysis")
    print("\nAll analysis uses synthetic data for demonstration purposes.")
