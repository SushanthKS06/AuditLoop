"""
Razorpay Test-Mode Settlement Recon API Client

Pulls settlement reconciliation data from Razorpay's test-mode API.
Endpoint: GET https://api.razorpay.com/v1/settlements/recon/combined

Note: Test-mode settlements only populate after creating test payments
AND waiting for the (accelerated but non-zero) test settlement cycle.
If too few records are returned, use hybrid mode with synthetic top-up.
"""

import os
import base64
import requests
from typing import Optional
import pandas as pd
from dotenv import load_dotenv

load_dotenv()


class RazorpayReconClient:
    """Thin client for Razorpay Settlement Recon API."""
    
    BASE_URL = "https://api.razorpay.com/v1/settlements/recon/combined"
    
    def __init__(self, key_id: Optional[str] = None, key_secret: Optional[str] = None):
        self.key_id = key_id or os.getenv("RAZORPAY_KEY_ID")
        self.key_secret = key_secret or os.getenv("RAZORPAY_KEY_SECRET")
        
        if not self.key_id or not self.key_secret:
            raise ValueError(
                "Razorpay credentials not found. Set RAZORPAY_KEY_ID and "
                "RAZORPAY_KEY_SECRET in .env or environment variables."
            )
    
    def _get_auth_header(self) -> dict:
        """Generate HTTP Basic Auth header."""
        credentials = f"{self.key_id}:{self.key_secret}"
        encoded = base64.b64encode(credentials.encode()).decode()
        return {"Authorization": f"Basic {encoded}"}
    
    def fetch_recon(
        self,
        year: int,
        month: int,
        day: Optional[int] = None,
        count: int = 100,
        skip: int = 0
    ) -> pd.DataFrame:
        """
        Fetch settlement recon data from Razorpay API.
        
        Args:
            year: YYYY format (required)
            month: MM format (required)
            day: DD format (optional)
            count: Number of records (1-1000)
            skip: Pagination offset
            
        Returns:
            DataFrame with normalized columns for matching engine
        """
        params = {
            "year": str(year),
            "month": str(month).zfill(2),
            "count": min(max(count, 1), 1000),
            "skip": skip
        }
        
        if day:
            params["day"] = str(day).zfill(2)
        
        response = requests.get(
            self.BASE_URL,
            headers=self._get_auth_header(),
            params=params,
            timeout=30
        )
        
        if response.status_code == 401:
            raise ValueError("Invalid Razorpay credentials. Check your test-mode keys.")
        elif response.status_code == 404:
            # No settlements for this date range - return empty DataFrame
            return self._empty_dataframe()
        elif response.status_code != 200:
            raise ValueError(f"API error: {response.status_code} - {response.text}")
        
        data = response.json()
        
        if not data or len(data) == 0:
            return self._empty_dataframe()
        
        return self._normalize_to_dataframe(data)
    
    def _empty_dataframe(self) -> pd.DataFrame:
        """Return empty DataFrame with expected schema."""
        return pd.DataFrame(columns=[
            'entity_id', 'type', 'payment_id', 'order_id', 'amount',
            'fee', 'tax', 'currency', 'settled_amount', 'debit', 'credit',
            'settlement_id', 'settlement_utr', 'created_at', 'settled_at',
            'method', 'card_network', 'dispute_id', 'source'
        ])
    
    def _normalize_to_dataframe(self, records: list) -> pd.DataFrame:
        """
        Normalize API response to DataFrame matching our matching engine schema.
        
        The API returns fields like:
        - entity_id, type (payment/refund/transfer/adjustment)
        - debit, credit, amount, currency, fee, tax, settled
        - created_at, settled_at, settlement_id, settlement_utr
        - order_id, payment_id, method, card_network, dispute_id
        """
        normalized = []
        
        for record in records:
            row = {
                'entity_id': record.get('entity_id', ''),
                'type': record.get('type', ''),
                'payment_id': record.get('payment_id', ''),
                'order_id': record.get('order_id', ''),
                'amount': float(record.get('amount', 0)) / 100,  # Convert paise to rupees
                'fee': float(record.get('fee', 0)) / 100 if record.get('fee') else 0.0,
                'tax': float(record.get('tax', 0)) / 100 if record.get('tax') else 0.0,
                'currency': record.get('currency', 'INR'),
                'settled_amount': float(record.get('settled', 0)) / 100,
                'debit': float(record.get('debit', 0)) / 100 if record.get('debit') else None,
                'credit': float(record.get('credit', 0)) / 100 if record.get('credit') else None,
                'settlement_id': record.get('settlement_id', ''),
                'settlement_utr': record.get('settlement_utr', ''),
                'created_at': record.get('created_at', ''),
                'settled_at': record.get('settled_at', ''),
                'method': record.get('method', ''),
                'card_network': record.get('card_network', ''),
                'dispute_id': record.get('dispute_id', ''),
                'source': 'razorpay_test'  # Standardized source tag for live/test Razorpay API records
            }
            normalized.append(row)
        
        df = pd.DataFrame(normalized)
        return df


def fetch_settlements_cli():
    """CLI entry point for fetching settlements."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Fetch settlement recon data from Razorpay test-mode API"
    )
    parser.add_argument("--year", type=int, required=True, help="Year (YYYY)")
    parser.add_argument("--month", type=int, required=True, help="Month (MM)")
    parser.add_argument("--day", type=int, default=None, help="Day (DD, optional)")
    parser.add_argument("--output", type=str, default="data/settlements_live.csv",
                        help="Output CSV file path")
    parser.add_argument("--count", type=int, default=100, help="Max records to fetch")
    
    args = parser.parse_args()
    
    try:
        client = RazorpayReconClient()
        print(f"Fetching settlements for {args.year}-{args.month:02d}...")
        
        df = client.fetch_recon(
            year=args.year,
            month=args.month,
            day=args.day,
            count=args.count
        )
        
        if len(df) == 0:
            print("Warning: No settlements found for this date range.")
            print("This is expected for fresh test accounts with no transaction history.")
            print("Consider generating test payments first via the Payments API.")
        else:
            print(f"Fetched {len(df)} settlement records.")
        
        df.to_csv(args.output, index=False)
        print(f"Saved to {args.output}")
        
        # Report source breakdown
        source_counts = df['source'].value_counts() if 'source' in df.columns else {}
        if source_counts.any():
            print(f"Source breakdown: {source_counts.to_dict()}")
        
    except ValueError as e:
        print(f"Error: {e}")
        print("\nTo get test-mode credentials:")
        print("1. Go to https://dashboard.razorpay.com/")
        print("2. Enable 'Test Mode' toggle")
        print("3. Navigate to Settings -> API Keys")
        print("4. Copy key_id and key_secret to .env file")
        return 1
    except Exception as e:
        print(f"Unexpected error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(fetch_settlements_cli())
