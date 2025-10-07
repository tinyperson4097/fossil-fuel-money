#!/usr/bin/env python3
"""
Leadership Donor Cross-Check
Cross-checks individuals from leadership_fast.csv against donor names in cuomo.csv and fixthecity.csv
using Virginia normalization and fuzzy matching.
"""

import pandas as pd
import argparse
import logging
from google.cloud import bigquery
from typing import Dict, List, Optional, Tuple
import json
import re
from fuzzywuzzy import fuzz

# Import Virginia normalization functions
from virginia_normalization import normalize_name

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def parse_lastname_firstname_name(name_str: str) -> str:
    """
    Parse names in 'LastName, FirstName MiddleInitial' format and convert to 'FirstName LastName'.
    
    Args:
        name_str: Name in format like "Smith, John A" or "Smith, John"
        
    Returns:
        Name in format "John Smith" for normalization
    """
    if not name_str or pd.isna(name_str):
        return ''
    
    name_str = str(name_str).strip()
    
    # Check if it's in LastName, FirstName format
    if ',' in name_str:
        parts = name_str.split(',', 1)
        if len(parts) == 2:
            last_name = parts[0].strip()
            first_part = parts[1].strip()
            
            # Split first part to get first name and middle initial/name
            first_parts = first_part.split()
            if first_parts:
                first_name = first_parts[0]
                # Return as "FirstName LastName" (ignore middle names for normalization)
                return f"{first_name} {last_name}"
    
    # If not in comma format, return as-is
    return name_str


class LeadershipDonorCrossChecker:
    """Cross-checks leadership individuals against donor databases using Virginia normalization."""
    
    def __init__(self, fuzzy_threshold: int = 95):
        self.fuzzy_threshold = fuzzy_threshold
        self.donor_lookup = {}  # Normalized name -> donor info
        
        logger.info(f"Initialized cross-checker with fuzzy threshold: {fuzzy_threshold}")
    
    def load_donor_databases(self, cuomo_csv: str, fixthecity_csv: str):
        """Load and normalize donor databases for fast lookup."""
        logger.info("Loading donor databases...")
        
        # Process Cuomo CSV
        try:
            cuomo_df = pd.read_csv(cuomo_csv)
            logger.info(f"Loaded {len(cuomo_df)} records from {cuomo_csv}")
            
            if 'NAME' in cuomo_df.columns:
                for _, row in cuomo_df.iterrows():
                    name = row['NAME']
                    if pd.notna(name) and str(name).strip():
                        # Determine if individual based on C_CODE
                        c_code = row.get('C_CODE', '')
                        is_individual = (str(c_code).upper() == 'IND')
                        
                        # Parse and normalize name
                        if is_individual:
                            parsed_name = parse_lastname_firstname_name(str(name))
                            normalized_name = normalize_name(parsed_name, is_individual=True)
                        else:
                            normalized_name = normalize_name(str(name), is_individual=False)
                        
                        if normalized_name:
                            # Store donor info with source
                            donor_info = row.to_dict()
                            donor_info['source_file'] = 'cuomo.csv'
                            donor_info['original_name'] = str(name)
                            donor_info['normalized_name'] = normalized_name
                            
                            # Handle multiple donors with same normalized name
                            if normalized_name not in self.donor_lookup:
                                self.donor_lookup[normalized_name] = []
                            self.donor_lookup[normalized_name].append(donor_info)
                
                logger.info(f"Processed {len([name for name in cuomo_df['NAME'] if pd.notna(name)])} names from cuomo.csv")
            else:
                logger.error(f"NAME column not found in {cuomo_csv}")
                
        except Exception as e:
            logger.error(f"Error loading {cuomo_csv}: {e}")
        
        # Process FixTheCity CSV
        try:
            fixthecity_df = pd.read_csv(fixthecity_csv)
            logger.info(f"Loaded {len(fixthecity_df)} records from {fixthecity_csv}")
            
            if 'NAME' in fixthecity_df.columns:
                for _, row in fixthecity_df.iterrows():
                    name = row['NAME']
                    if pd.notna(name) and str(name).strip():
                        # Determine if individual based on C_CODE
                        c_code = row.get('C_CODE', '')
                        is_individual = (str(c_code).upper() == 'IND')
                        
                        # Parse and normalize name
                        if is_individual:
                            parsed_name = parse_lastname_firstname_name(str(name))
                            normalized_name = normalize_name(parsed_name, is_individual=True)
                        else:
                            normalized_name = normalize_name(str(name), is_individual=False)
                        
                        if normalized_name:
                            # Store donor info with source
                            donor_info = row.to_dict()
                            donor_info['source_file'] = 'fixthecity.csv'
                            donor_info['original_name'] = str(name)
                            donor_info['normalized_name'] = normalized_name
                            
                            # Handle multiple donors with same normalized name
                            if normalized_name not in self.donor_lookup:
                                self.donor_lookup[normalized_name] = []
                            self.donor_lookup[normalized_name].append(donor_info)
                
                logger.info(f"Processed {len([name for name in fixthecity_df['NAME'] if pd.notna(name)])} names from fixthecity.csv")
            else:
                logger.error(f"NAME column not found in {fixthecity_csv}")
                
        except Exception as e:
            logger.error(f"Error loading {fixthecity_csv}: {e}")
        
        logger.info(f"Total unique normalized donor names: {len(self.donor_lookup)}")
        
        # Show some examples
        if self.donor_lookup:
            examples = list(self.donor_lookup.keys())[:5]
            logger.info(f"Sample normalized donor names: {examples}")
    
    def check_leadership_donor_match(self, individual_name: str) -> Tuple[str, str, int, List[Dict]]:
        """
        Check if a leadership individual matches any donors.
        
        Args:
            individual_name: Name from leadership data
            
        Returns:
            Tuple of (match_status, matched_donor_name, fuzzy_score, matching_records)
        """
        if not individual_name or pd.isna(individual_name):
            return "NOPE", "NOPE", 0, []
        
        # Normalize the individual name (assume it's a person)
        individual_clean = str(individual_name).strip()
        if not individual_clean:
            return "NOPE", "NOPE", 0, []
        
        # Parse if needed and normalize as individual
        parsed_individual = parse_lastname_firstname_name(individual_clean)
        normalized_individual = normalize_name(parsed_individual, is_individual=True)
        
        if not normalized_individual:
            return "NOPE", "NOPE", 0, []
        
        # Check for exact match first
        if normalized_individual in self.donor_lookup:
            matching_records = self.donor_lookup[normalized_individual]
            return "EXACT_MATCH", matching_records[0]['original_name'], 100, matching_records
        
        # Check for fuzzy matches
        best_match = None
        best_score = 0
        best_records = []
        
        for normalized_donor_name, donor_records in self.donor_lookup.items():
            score = fuzz.ratio(normalized_individual, normalized_donor_name)
            if score >= self.fuzzy_threshold and score > best_score:
                best_score = score
                best_match = normalized_donor_name
                best_records = donor_records
        
        if best_match:
            logger.debug(f"Fuzzy match: '{individual_name}' -> '{best_records[0]['original_name']}' (score: {best_score})")
            return "FUZZY_MATCH", best_records[0]['original_name'], best_score, best_records
        
        logger.debug(f"No match found for: '{individual_name}' (normalized: '{normalized_individual}')")
        return "NOPE", "NOPE", 0, []


def process_leadership_csv(leadership_csv: str) -> pd.DataFrame:
    """Load and process leadership CSV file."""
    try:
        leadership_df = pd.read_csv(leadership_csv)
        logger.info(f"Loaded {len(leadership_df)} leadership records from {leadership_csv}")
        
        if 'individual' not in leadership_df.columns:
            logger.error(f"'individual' column not found in {leadership_csv}")
            return pd.DataFrame()
        
        return leadership_df
        
    except Exception as e:
        logger.error(f"Error loading {leadership_csv}: {e}")
        return pd.DataFrame()


def upload_to_bigquery(df: pd.DataFrame, project_id: str, dataset_id: str, table_id: str):
    """Upload DataFrame to BigQuery."""
    try:
        client = bigquery.Client(project=project_id)
        
        # Create dataset if it doesn't exist
        dataset_ref = client.dataset(dataset_id)
        try:
            client.get_dataset(dataset_ref)
            logger.info(f"Dataset {dataset_id} already exists")
        except:
            dataset = bigquery.Dataset(dataset_ref)
            dataset.location = "US"
            dataset = client.create_dataset(dataset)
            logger.info(f"Created dataset {dataset_id}")
        
        # Upload data
        table_ref = dataset_ref.table(table_id)
        
        job_config = bigquery.LoadJobConfig(
            write_disposition=bigquery.WriteDisposition.WRITE_TRUNCATE,
            autodetect=False,
            create_disposition=bigquery.CreateDisposition.CREATE_IF_NEEDED
        )
        
        # Define schema manually to ensure all columns are treated as strings
        schema = []
        for column in df.columns:
            # Clean column names for BigQuery
            clean_column_name = re.sub(r'[^a-zA-Z0-9_]', '_', str(column))
            schema.append(bigquery.SchemaField(clean_column_name, "STRING", mode="NULLABLE"))
        
        # Also clean the DataFrame column names to match schema
        df.columns = [re.sub(r'[^a-zA-Z0-9_]', '_', str(col)) for col in df.columns]
        
        job_config.schema = schema
        
        job = client.load_table_from_dataframe(df, table_ref, job_config=job_config)
        job.result()  # Wait for completion
        
        logger.info(f"Successfully uploaded {len(df)} records to {project_id}.{dataset_id}.{table_id}")
        
    except Exception as e:
        logger.error(f"Error uploading to BigQuery: {e}")
        raise


def main():
    parser = argparse.ArgumentParser(description='Cross-check leadership individuals against donor databases')
    parser.add_argument('--leadership-csv', type=str, required=True,
                       help='Path to leadership_fast.csv file')
    parser.add_argument('--cuomo-csv', type=str, required=True,
                       help='Path to cuomo.csv file')
    parser.add_argument('--fixthecity-csv', type=str, required=True,
                       help='Path to fixthecity.csv file')
    parser.add_argument('--project-id', type=str, default='va-campaign-finance',
                       help='Google Cloud project ID (default: va-campaign-finance)')
    parser.add_argument('--dataset', type=str, default='ny_elections',
                       help='BigQuery dataset name (default: ny_elections)')
    parser.add_argument('--table', type=str, default='leadership_donor_crosscheck',
                       help='BigQuery table name (default: leadership_donor_crosscheck)')
    parser.add_argument('--fuzzy-threshold', type=int, default=95,
                       help='Fuzzy matching threshold (default: 95)')
    parser.add_argument('--output-csv', type=str,
                       help='Optional: Save results to CSV file')
    parser.add_argument('--test-mode', action='store_true',
                       help='Test mode: only process first 100 leadership records')
    parser.add_argument('--debug', action='store_true',
                       help='Enable debug logging')
    
    args = parser.parse_args()
    
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        # Load leadership data
        leadership_df = process_leadership_csv(args.leadership_csv)
        if leadership_df.empty:
            logger.error("No leadership data to process")
            return 1
        
        # Initialize cross-checker and load donor databases
        cross_checker = LeadershipDonorCrossChecker(args.fuzzy_threshold)
        cross_checker.load_donor_databases(args.cuomo_csv, args.fixthecity_csv)
        
        # Test mode
        if args.test_mode:
            leadership_df = leadership_df.head(100)
            logger.info("Test mode: Processing only first 100 leadership records")
        
        logger.info(f"Cross-checking {len(leadership_df)} leadership individuals against donor databases...")
        
        # Process each leadership record
        results = []
        
        for idx, row in leadership_df.iterrows():
            individual_name = row['individual']
            
            logger.debug(f"Processing {idx + 1}/{len(leadership_df)}: {individual_name}")
            
            # Check for donor match
            match_status, matched_donor_name, fuzzy_score, matching_records = cross_checker.check_leadership_donor_match(individual_name)
            
            # Create result record with all original leadership data
            result = row.to_dict()
            
            # Add cross-check results
            result.update({
                'donor_match_status': match_status,
                'matched_donor_name': matched_donor_name,
                'donor_match_score': fuzzy_score,
                'donor_record_count': len(matching_records)
            })
            
            # Add donor information if matched
            if matching_records:
                # Take first matching donor record for main fields
                first_donor = matching_records[0]
                result.update({
                    'donor_source_file': first_donor['source_file'],
                    'donor_original_name': first_donor['original_name'],
                    'donor_normalized_name': first_donor['normalized_name']
                })
                
                # Add all matching donor records as JSON
                result['all_donor_matches'] = json.dumps([
                    {
                        'name': donor['original_name'],
                        'source': donor['source_file'],
                        'c_code': donor.get('C_CODE', ''),
                        'empname': donor.get('EMPNAME', ''),
                        'amount': donor.get('AMOUNT', ''),
                        'date': donor.get('DATE', '')
                    }
                    for donor in matching_records
                ])
            else:
                result.update({
                    'donor_source_file': 'NOPE',
                    'donor_original_name': 'NOPE',
                    'donor_normalized_name': 'NOPE',
                    'all_donor_matches': '[]'
                })
            
            results.append(result)
            
            # Progress logging
            if (idx + 1) % 100 == 0:
                match_count = sum(1 for r in results if r['donor_match_status'] != 'NOPE')
                logger.info(f"Progress: {idx + 1}/{len(leadership_df)} - {match_count} matches found")
        
        # Create results DataFrame
        results_df = pd.DataFrame(results)
        
        # Convert all columns to string for BigQuery compatibility
        for col in results_df.columns:
            results_df[col] = results_df[col].astype(str)
        
        # Replace 'nan' strings with None for proper NULL handling
        results_df = results_df.replace('nan', None)
        
        # Summary statistics
        total_matches = len(results_df[results_df['donor_match_status'] != 'NOPE'])
        exact_matches = len(results_df[results_df['donor_match_status'] == 'EXACT_MATCH'])
        fuzzy_matches = len(results_df[results_df['donor_match_status'] == 'FUZZY_MATCH'])
        
        logger.info(f"\n=== CROSS-CHECK COMPLETE ===")
        logger.info(f"Total leadership individuals: {len(results_df)}")
        logger.info(f"Total donor matches: {total_matches}")
        logger.info(f"Exact matches: {exact_matches}")
        logger.info(f"Fuzzy matches: {fuzzy_matches}")
        logger.info(f"Match rate: {(total_matches/len(results_df)*100):.1f}%")
        
        # Save to CSV if requested
        if args.output_csv:
            results_df.to_csv(args.output_csv, index=False)
            logger.info(f"Results saved to {args.output_csv}")
        
        # Upload to BigQuery
        logger.info("Uploading results to BigQuery...")
        upload_to_bigquery(results_df, args.project_id, args.dataset, args.table)
        
        return 0
        
    except Exception as e:
        logger.error(f"Script failed: {e}")
        return 1


if __name__ == "__main__":
    exit(main())