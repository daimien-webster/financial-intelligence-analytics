"""
Synthetic Transaction Data Generator
Author: Daimien Webster
Purpose: Generate anonymized synthetic transaction data for testing and demonstration

This module creates realistic but entirely synthetic financial transaction data
suitable for demonstrating data quality assessment and pattern detection capabilities.
All data is generated randomly with no connection to real entities.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
import string

class SyntheticTransactionGenerator:
    """Generate synthetic transaction data with configurable characteristics."""
    
    def __init__(self, seed: int = 42):
        """
        Initialize generator with random seed for reproducibility.
        
        Args:
            seed: Random seed for consistent data generation
        """
        np.random.seed(seed)
        random.seed(seed)
        
        self.transaction_types = [
            'WIRE_TRANSFER', 'CASH_DEPOSIT', 'CASH_WITHDRAWAL', 
            'CARD_PAYMENT', 'CHEQUE', 'DIRECT_DEBIT', 'ATM_WITHDRAWAL'
        ]
        
    def generate_account_number(self) -> str:
        """Generate synthetic account number."""
        return ''.join(random.choices(string.ascii_uppercase + string.digits, k=12))
    
    def generate_transaction_amount(self, transaction_type: str) -> float:
        """
        Generate realistic transaction amount based on type.
        
        Args:
            transaction_type: Type of transaction
            
        Returns:
            Transaction amount
        """
        if transaction_type == 'CASH_DEPOSIT':
            # Include some structuring patterns (just below $10k)
            if random.random() < 0.08:  # 8% near threshold
                return round(random.uniform(9000, 9900), 2)
            else:
                return round(np.random.lognormal(6, 1.5), 2)
        
        elif transaction_type == 'WIRE_TRANSFER':
            return round(np.random.lognormal(8, 1.8), 2)
        
        elif transaction_type == 'ATM_WITHDRAWAL':
            # ATM withdrawals typically smaller
            return round(min(np.random.lognormal(4, 1), 1000), 2)
        
        else:
            return round(np.random.lognormal(5, 1.5), 2)
    
    def generate_dataset(
        self, 
        num_records: int = 10000,
        start_date: datetime = None,
        end_date: datetime = None,
        quality_issues_rate: float = 0.05
    ) -> pd.DataFrame:
        """
        Generate synthetic transaction dataset.
        
        Args:
            num_records: Number of transaction records to generate
            start_date: Start date for transactions
            end_date: End date for transactions
            quality_issues_rate: Proportion of records with intentional quality issues
            
        Returns:
            DataFrame containing synthetic transactions
        """
        if start_date is None:
            start_date = datetime.now() - timedelta(days=365)
        if end_date is None:
            end_date = datetime.now()
        
        date_range = (end_date - start_date).days
        
        transactions = []
        
        # Generate pool of account numbers for reuse (realistic)
        account_pool = [self.generate_account_number() for _ in range(500)]
        
        for i in range(num_records):
            # Generate transaction date
            trans_date = start_date + timedelta(
                days=random.randint(0, date_range),
                hours=random.randint(0, 23),
                minutes=random.randint(0, 59)
            )
            
            # Generate report date (typically 1-5 days after transaction)
            report_lag = np.random.exponential(scale=2)
            if random.random() < 0.1:  # 10% late reports
                report_lag = random.uniform(10, 30)
            
            report_date = trans_date + timedelta(days=report_lag)
            
            # Select transaction type
            trans_type = random.choice(self.transaction_types)
            
            # Generate amount
            amount = self.generate_transaction_amount(trans_type)
            
            # Select accounts from pool
            sender = random.choice(account_pool)
            receiver = random.choice(account_pool)
            
            # Introduce quality issues
            if random.random() < quality_issues_rate:
                issue_type = random.choice([
                    'missing_amount', 'missing_sender', 'same_account', 
                    'invalid_amount', 'future_date'
                ])
                
                if issue_type == 'missing_amount':
                    amount = np.nan
                elif issue_type == 'missing_sender':
                    sender = np.nan
                elif issue_type == 'same_account':
                    receiver = sender
                elif issue_type == 'invalid_amount':
                    amount = -abs(amount) if random.random() < 0.5 else 0
                elif issue_type == 'future_date':
                    trans_date = datetime.now() + timedelta(days=random.randint(1, 30))
            
            transaction = {
                'transaction_id': f'TXN{i+1:08d}',
                'transaction_date': trans_date,
                'report_date': report_date,
                'amount': amount,
                'sender_account': sender,
                'receiver_account': receiver,
                'transaction_type': trans_type,
                'currency': 'AUD',
                'channel': random.choice(['BRANCH', 'ONLINE', 'ATM', 'MOBILE']),
                'country_code': 'AU'
            }
            
            transactions.append(transaction)
        
        df = pd.DataFrame(transactions)
        
        # Introduce some missing data
        if quality_issues_rate > 0:
            missing_indices = np.random.choice(
                df.index, 
                size=int(len(df) * quality_issues_rate * 0.5),
                replace=False
            )
            df.loc[missing_indices, 'channel'] = np.nan
        
        return df


def generate_sample_datasets():
    """Generate multiple sample datasets for demonstration."""
    
    generator = SyntheticTransactionGenerator(seed=42)
    
    # High quality dataset
    print("Generating high quality dataset...")
    high_quality_df = generator.generate_dataset(
        num_records=5000,
        quality_issues_rate=0.02
    )
    high_quality_df.to_csv('/home/claude/transactions_high_quality.csv', index=False)
    print(f"Saved high quality dataset: {len(high_quality_df)} records")
    
    # Medium quality dataset
    print("\nGenerating medium quality dataset...")
    medium_quality_df = generator.generate_dataset(
        num_records=5000,
        quality_issues_rate=0.10
    )
    medium_quality_df.to_csv('/home/claude/transactions_medium_quality.csv', index=False)
    print(f"Saved medium quality dataset: {len(medium_quality_df)} records")
    
    # Low quality dataset
    print("\nGenerating low quality dataset...")
    low_quality_df = generator.generate_dataset(
        num_records=5000,
        quality_issues_rate=0.25
    )
    low_quality_df.to_csv('/home/claude/transactions_low_quality.csv', index=False)
    print(f"Saved low quality dataset: {len(low_quality_df)} records")
    
    print("\n" + "="*60)
    print("Sample datasets generated successfully!")
    print("\nDataset summary:")
    print(f"  - High quality: 2% quality issues")
    print(f"  - Medium quality: 10% quality issues")
    print(f"  - Low quality: 25% quality issues")
    print("\nAll data is entirely synthetic and anonymized.")


if __name__ == "__main__":
    generate_sample_datasets()
