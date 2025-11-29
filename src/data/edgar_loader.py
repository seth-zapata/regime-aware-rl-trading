"""
SEC EDGAR Data Loader

Downloads and parses SEC filings (10-K, 10-Q, 8-K) from the EDGAR database.
Extracts text sections for sentiment analysis using FinBERT.

Important: SEC requires a User-Agent header with contact information.
Rate limit: 10 requests per second.
"""

import os
import re
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

import pandas as pd
import requests
from bs4 import BeautifulSoup


@dataclass
class Filing:
    """Represents a single SEC filing."""
    accession_number: str
    filing_type: str
    filing_date: str
    company_name: str
    cik: str
    document_url: str
    description: str = ""


class EDGARLoader:
    """
    Load and parse SEC EDGAR filings.

    Downloads 10-K (annual), 10-Q (quarterly), and 8-K (current) reports.
    Extracts key text sections for sentiment analysis.

    Example:
        loader = EDGARLoader(email="your.email@example.com")
        filings = loader.get_filings('AAPL', filing_type='10-K', count=5)
        text = loader.extract_filing_text(filings[0])
    """

    BASE_URL = "https://www.sec.gov"
    SUBMISSIONS_URL = "https://data.sec.gov/submissions"
    RATE_LIMIT_DELAY = 0.1  # 100ms between requests (10 req/sec)

    # Key sections to extract from 10-K/10-Q filings
    SECTIONS_10K = {
        'item_1': r'item\s*1[.\s]*business',
        'item_1a': r'item\s*1a[.\s]*risk\s*factors',
        'item_7': r'item\s*7[.\s]*management.{0,50}discussion',
        'item_7a': r'item\s*7a[.\s]*quantitative',
    }

    def __init__(
        self,
        email: str = "research@example.com",
        cache_dir: Optional[Path] = None
    ):
        """
        Initialize EDGAR loader.

        Args:
            email: Contact email for SEC User-Agent header (required by SEC)
            cache_dir: Directory for caching filings. Defaults to data/raw/edgar/
        """
        self.email = email
        self.headers = {
            "User-Agent": f"Research Project {email}",
            "Accept-Encoding": "gzip, deflate",
        }

        if cache_dir is None:
            project_root = Path(__file__).parent.parent.parent
            cache_dir = project_root / 'data' / 'raw' / 'edgar'
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self._last_request_time = 0

    def _rate_limit(self):
        """Ensure we don't exceed SEC rate limits."""
        elapsed = time.time() - self._last_request_time
        if elapsed < self.RATE_LIMIT_DELAY:
            time.sleep(self.RATE_LIMIT_DELAY - elapsed)
        self._last_request_time = time.time()

    def _make_request(self, url: str) -> requests.Response:
        """Make rate-limited request to SEC."""
        self._rate_limit()
        response = requests.get(url, headers=self.headers, timeout=30)
        response.raise_for_status()
        return response

    def get_cik(self, ticker: str) -> str:
        """
        Get CIK (Central Index Key) for a ticker symbol.

        Args:
            ticker: Stock ticker symbol (e.g., 'AAPL')

        Returns:
            CIK as zero-padded 10-digit string
        """
        # SEC provides a ticker->CIK mapping
        url = f"{self.BASE_URL}/cgi-bin/browse-edgar?action=getcompany&CIK={ticker}&type=&dateb=&owner=include&count=1&output=atom"

        try:
            response = self._make_request(url)
            # Parse the response to extract CIK
            match = re.search(r'CIK=(\d+)', response.text)
            if match:
                cik = match.group(1).zfill(10)
                return cik
        except Exception:
            pass

        # Alternative: Try the company tickers JSON
        try:
            tickers_url = "https://www.sec.gov/files/company_tickers.json"
            response = self._make_request(tickers_url)
            data = response.json()

            for entry in data.values():
                if entry.get('ticker', '').upper() == ticker.upper():
                    return str(entry['cik_str']).zfill(10)
        except Exception:
            pass

        raise ValueError(f"Could not find CIK for ticker: {ticker}")

    def get_filings(
        self,
        ticker: str,
        filing_type: str = "10-K",
        count: int = 10,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> List[Filing]:
        """
        Get list of filings for a company.

        Args:
            ticker: Stock ticker symbol
            filing_type: Type of filing ('10-K', '10-Q', '8-K')
            count: Maximum number of filings to return
            start_date: Filter filings after this date (YYYY-MM-DD)
            end_date: Filter filings before this date (YYYY-MM-DD)

        Returns:
            List of Filing objects with metadata
        """
        cik = self.get_cik(ticker)

        # Fetch company submissions
        submissions_url = f"{self.SUBMISSIONS_URL}/CIK{cik}.json"

        try:
            response = self._make_request(submissions_url)
            data = response.json()
        except Exception as e:
            raise RuntimeError(f"Failed to fetch submissions for {ticker}: {e}")

        # Extract filing information
        filings = []
        recent = data.get('filings', {}).get('recent', {})

        if not recent:
            return filings

        form_types = recent.get('form', [])
        accession_numbers = recent.get('accessionNumber', [])
        filing_dates = recent.get('filingDate', [])
        primary_docs = recent.get('primaryDocument', [])

        company_name = data.get('name', ticker)

        for i, form in enumerate(form_types):
            if form != filing_type:
                continue

            filing_date = filing_dates[i]

            # Apply date filters
            if start_date and filing_date < start_date:
                continue
            if end_date and filing_date > end_date:
                continue

            accession = accession_numbers[i].replace('-', '')
            doc = primary_docs[i]

            doc_url = f"{self.BASE_URL}/Archives/edgar/data/{cik}/{accession}/{doc}"

            filing = Filing(
                accession_number=accession_numbers[i],
                filing_type=form,
                filing_date=filing_date,
                company_name=company_name,
                cik=cik,
                document_url=doc_url
            )
            filings.append(filing)

            if len(filings) >= count:
                break

        return filings

    def download_filing(self, filing: Filing, use_cache: bool = True) -> str:
        """
        Download the full text of a filing.

        Args:
            filing: Filing object with document URL
            use_cache: Whether to use cached version if available

        Returns:
            Raw HTML/text content of the filing
        """
        # Check cache
        cache_file = self.cache_dir / f"{filing.cik}_{filing.accession_number}.html"

        if use_cache and cache_file.exists():
            return cache_file.read_text(encoding='utf-8', errors='ignore')

        # Download
        try:
            response = self._make_request(filing.document_url)
            content = response.text

            # Cache
            if use_cache:
                cache_file.write_text(content, encoding='utf-8')

            return content

        except Exception as e:
            raise RuntimeError(f"Failed to download filing: {e}")

    def extract_filing_text(
        self,
        filing: Filing,
        sections: Optional[List[str]] = None,
        use_cache: bool = True
    ) -> Dict[str, str]:
        """
        Extract text sections from a filing.

        Args:
            filing: Filing object
            sections: Which sections to extract. Defaults to risk factors and MD&A.
            use_cache: Whether to use cached filing

        Returns:
            Dict mapping section names to extracted text
        """
        html_content = self.download_filing(filing, use_cache=use_cache)

        # Parse HTML
        soup = BeautifulSoup(html_content, 'html.parser')

        # Remove script and style elements
        for element in soup(['script', 'style', 'meta', 'link']):
            element.decompose()

        # Get text
        text = soup.get_text(separator=' ', strip=True)

        # Clean text
        text = re.sub(r'\s+', ' ', text)

        # If no specific sections requested, return full text
        if sections is None:
            sections = ['item_1a', 'item_7']  # Risk factors and MD&A

        result = {}

        # Try to extract specific sections
        for section in sections:
            if section in self.SECTIONS_10K:
                pattern = self.SECTIONS_10K[section]
                extracted = self._extract_section(text, pattern)
                if extracted:
                    result[section] = extracted

        # If no sections extracted, return truncated full text
        if not result:
            result['full_text'] = text[:50000]  # Limit to 50k chars

        return result

    def _extract_section(
        self,
        text: str,
        start_pattern: str,
        max_length: int = 20000
    ) -> Optional[str]:
        """
        Extract a section from filing text.

        Args:
            text: Full filing text
            start_pattern: Regex pattern for section start
            max_length: Maximum characters to extract

        Returns:
            Extracted section text or None if not found
        """
        # Find section start
        match = re.search(start_pattern, text.lower())
        if not match:
            return None

        start_idx = match.start()

        # Find next "item" header (section end)
        end_match = re.search(
            r'item\s*\d+[a-z]?[.\s]',
            text[start_idx + 100:start_idx + max_length].lower()
        )

        if end_match:
            end_idx = start_idx + 100 + end_match.start()
        else:
            end_idx = start_idx + max_length

        section_text = text[start_idx:end_idx]

        # Clean up
        section_text = re.sub(r'\s+', ' ', section_text).strip()

        return section_text if len(section_text) > 100 else None

    def get_filings_dataframe(
        self,
        ticker: str,
        filing_types: List[str] = ['10-K', '10-Q'],
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        extract_text: bool = False
    ) -> pd.DataFrame:
        """
        Get filings as a DataFrame with optional text extraction.

        Args:
            ticker: Stock ticker symbol
            filing_types: Types of filings to fetch
            start_date: Filter start date
            end_date: Filter end date
            extract_text: Whether to extract and include text content

        Returns:
            DataFrame with filing metadata and optionally text
        """
        all_filings = []

        for filing_type in filing_types:
            filings = self.get_filings(
                ticker,
                filing_type=filing_type,
                count=40,  # Get up to 40 per type (~10 years of quarterlies)
                start_date=start_date,
                end_date=end_date
            )
            all_filings.extend(filings)

        if not all_filings:
            return pd.DataFrame()

        # Convert to DataFrame
        records = []
        for filing in all_filings:
            record = {
                'ticker': ticker,
                'filing_type': filing.filing_type,
                'filing_date': filing.filing_date,
                'company_name': filing.company_name,
                'cik': filing.cik,
                'accession_number': filing.accession_number,
                'document_url': filing.document_url,
            }

            if extract_text:
                try:
                    text_sections = self.extract_filing_text(filing)
                    record['risk_factors'] = text_sections.get('item_1a', '')
                    record['mda'] = text_sections.get('item_7', '')
                    record['full_text'] = text_sections.get('full_text', '')[:10000]
                except Exception as e:
                    print(f"Warning: Failed to extract text from {filing.accession_number}: {e}")
                    record['risk_factors'] = ''
                    record['mda'] = ''
                    record['full_text'] = ''

            records.append(record)

        df = pd.DataFrame(records)
        df['filing_date'] = pd.to_datetime(df['filing_date'])
        df = df.sort_values('filing_date')

        return df


def main():
    """Example usage of EDGARLoader."""
    loader = EDGARLoader(email="research@example.com")

    # Get filings for Apple
    print("Fetching AAPL 10-K filings...")
    filings = loader.get_filings('AAPL', filing_type='10-K', count=3)

    print(f"\nFound {len(filings)} filings:")
    for f in filings:
        print(f"  {f.filing_date}: {f.filing_type} - {f.company_name}")

    # Extract text from most recent filing
    if filings:
        print("\nExtracting text from most recent filing...")
        text_sections = loader.extract_filing_text(filings[0])
        for section, text in text_sections.items():
            print(f"  {section}: {len(text)} characters")

    # Get as DataFrame
    print("\nFetching as DataFrame (without text)...")
    df = loader.get_filings_dataframe('AAPL', filing_types=['10-K', '10-Q'])
    print(f"Total filings: {len(df)}")
    print(df[['filing_date', 'filing_type']].head(10))


if __name__ == '__main__':
    main()
