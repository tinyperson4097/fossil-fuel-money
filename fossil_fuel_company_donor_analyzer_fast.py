#!/usr/bin/env python3
"""
Fossil Fuel Donor Connection Analyzer
Processes donor lists from CSV files and identifies fossil fuel connections
by cross-checking against a fossil fuel company list.
"""

import pandas as pd
import argparse
import logging
from google.cloud import bigquery
from typing import Dict, List, Optional, Tuple
import json
import re
from fuzzywuzzy import fuzz

# Import name normalization functions
from name_normalization import normalize_name

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


class FossilFuelCrossChecker:
    """Cross-checks donors against fossil fuel company list using Virginia normalization."""
    
    def __init__(self, fflist_csv_path: str, fuzzy_threshold: int = 95):
        self.fuzzy_threshold = fuzzy_threshold
        self.fossil_fuel_companies = {}
        self.load_fossil_fuel_list(fflist_csv_path)
        
        logger.info(f"Initialized cross-checker with {len(self.fossil_fuel_companies)} fossil fuel companies")
        logger.info(f"Fuzzy matching threshold: {fuzzy_threshold}")
    
    def load_fossil_fuel_list(self, fflist_csv_path: str):
        """Load and normalize fossil fuel company list."""
        try:
            ff_df = pd.read_csv(fflist_csv_path)
            logger.info(f"Loaded {len(ff_df)} companies from {fflist_csv_path}")
            
            # Expected columns in fflist.csv
            company_col = None
            category_col = None
            
            # Try to find the company name column
            for col in ff_df.columns:
                if 'company' in col.lower() or 'name' in col.lower():
                    company_col = col
                    break
            
            # Try to find the category column
            for col in ff_df.columns:
                if 'category' in col.lower() or 'type' in col.lower():
                    category_col = col
                    break
            
            if not company_col:
                # Use first column as company name
                company_col = ff_df.columns[0]
                logger.warning(f"No company column found, using first column: {company_col}")
            
            if not category_col:
                # Use second column as category or create default
                if len(ff_df.columns) > 1:
                    category_col = ff_df.columns[1]
                    logger.warning(f"No category column found, using second column: {category_col}")
                else:
                    ff_df['category'] = 'Fossil Fuel'
                    category_col = 'category'
                    logger.warning("No category column found, using default 'Fossil Fuel'")
            
            # Normalize company names using Virginia normalization (isIndividual=False)
            for _, row in ff_df.iterrows():
                company_name = row[company_col]
                category = row[category_col] if pd.notna(row[category_col]) else 'Fossil Fuel'
                
                if pd.notna(company_name) and str(company_name).strip():
                    normalized_company = normalize_name(str(company_name), is_individual=False)
                    if normalized_company:
                        self.fossil_fuel_companies[normalized_company] = {
                            'original_name': str(company_name),
                            'category': str(category)
                        }
            
            logger.info(f"Normalized {len(self.fossil_fuel_companies)} fossil fuel companies")
            
            # Show some examples
            if self.fossil_fuel_companies:
                examples = list(self.fossil_fuel_companies.keys())[:3]
                logger.info(f"Sample normalized companies: {examples}")
                
        except Exception as e:
            logger.error(f"Error loading fossil fuel list from {fflist_csv_path}: {e}")
            self.fossil_fuel_companies = {}
    
    def check_fossil_fuel_match(self, name: str, employer: str, c_code: str) -> Tuple[str, str, int]:
        """
        Check if name or employer matches fossil fuel companies.
        
        Args:
            name: Donor name
            employer: Employer name  
            c_code: Contribution code (IND for individual, other for organizations)
            
        Returns:
            Tuple of (matched_company_name, category, fuzzy_score) or ("NOPE", "NOPE", 0)
        """
        
        # Determine if this is an individual based on C_CODE
        is_individual = (str(c_code).upper() == 'IND')
        
        # Normalize the donor name
        if name and pd.notna(name) and str(name).strip():
            if is_individual:
                # Parse LastName, FirstName format first
                parsed_name = parse_lastname_firstname_name(str(name))
                normalized_name = normalize_name(parsed_name, is_individual=True)
            else:
                normalized_name = normalize_name(str(name), is_individual=False)
        else:
            normalized_name = ''
        
        # Normalize the employer name (always as organization)
        if employer and pd.notna(employer) and str(employer).strip():
            normalized_employer = normalize_name(str(employer), is_individual=False)
        else:
            normalized_employer = ''
        
        # Check for exact matches first
        if normalized_name in self.fossil_fuel_companies:
            company_info = self.fossil_fuel_companies[normalized_name]
            return company_info['original_name'], company_info['category'], 100  # Exact match = 100% score
        
        if normalized_employer in self.fossil_fuel_companies:
            company_info = self.fossil_fuel_companies[normalized_employer]
            return company_info['original_name'], company_info['category'], 100  # Exact match = 100% score
        
        # Check for fuzzy matches
        best_match = None
        best_score = 0
        
        # Get list of all normalized company names for fuzzy matching
        company_names = list(self.fossil_fuel_companies.keys())
        
        # Check name against all companies
        if normalized_name:
            for company in company_names:
                score = fuzz.ratio(normalized_name, company)
                if score >= self.fuzzy_threshold and score > best_score:
                    best_score = score
                    best_match = company
        
        # Check employer against all companies  
        if normalized_employer:
            for company in company_names:
                score = fuzz.ratio(normalized_employer, company)
                if score >= self.fuzzy_threshold and score > best_score:
                    best_score = score
                    best_match = company
        
        if best_match:
            company_info = self.fossil_fuel_companies[best_match]
            logger.debug(f"Fuzzy match found: '{name}' or '{employer}' -> '{best_match}' (score: {best_score})")
            return company_info['original_name'], company_info['category'], best_score
        
        # Debug logging for no matches
        logger.debug(f"No fossil fuel match found for: NAME='{normalized_name}', EMPLOYER='{normalized_employer}'")
        
        return "NOPE", "NOPE", 0



def process_csv_files(cuomo_csv: str, fixthecity_csv: str) -> pd.DataFrame:
    """Process and merge the CSV files."""
    logger.info("Processing CSV files...")
    
    # Read CSV files
    try:
        cuomo_df = pd.read_csv(cuomo_csv)
        logger.info(f"Loaded {len(cuomo_df)} records from {cuomo_csv}")
    except Exception as e:
        logger.error(f"Error reading {cuomo_csv}: {e}")
        cuomo_df = pd.DataFrame()
    
    try:
        fixthecity_df = pd.read_csv(fixthecity_csv)
        logger.info(f"Loaded {len(fixthecity_df)} records from {fixthecity_csv}")
    except Exception as e:
        logger.error(f"Error reading {fixthecity_csv}: {e}")
        fixthecity_df = pd.DataFrame()
    
    # Combine dataframes
    combined_data = []
    
    # Process Cuomo CSV
    if not cuomo_df.empty:
        logger.info(f"Cuomo CSV columns: {list(cuomo_df.columns)}")
        if 'NAME' in cuomo_df.columns:
            for _, row in cuomo_df.iterrows():
                # Skip rows with empty NAME
                if pd.notna(row['NAME']) and str(row['NAME']).strip():
                    combined_data.append({
                        'name': row['NAME'],
                        'source_file': 'cuomo.csv',
                        'employer': row.get('EMPNAME', '') if 'EMPNAME' in row else '',
                        'original_data': row.to_dict()
                    })
            logger.info(f"Added {sum(1 for _, row in cuomo_df.iterrows() if pd.notna(row['NAME']) and str(row['NAME']).strip())} records from cuomo.csv")
        else:
            logger.error(f"NAME column not found in cuomo.csv. Available columns: {list(cuomo_df.columns)}")
    
    # Process FixTheCity CSV
    if not fixthecity_df.empty:
        logger.info(f"FixTheCity CSV columns: {list(fixthecity_df.columns)}")
        if 'NAME' in fixthecity_df.columns:
            for _, row in fixthecity_df.iterrows():
                # Skip rows with empty NAME
                if pd.notna(row['NAME']) and str(row['NAME']).strip():
                    combined_data.append({
                        'name': row['NAME'],
                        'source_file': 'fixthecity.csv', 
                        'employer': row.get('EMPNAME', '') if 'EMPNAME' in row else '',
                        'original_data': row.to_dict()
                    })
            logger.info(f"Added {sum(1 for _, row in fixthecity_df.iterrows() if pd.notna(row['NAME']) and str(row['NAME']).strip())} records from fixthecity.csv")
        else:
            logger.error(f"NAME column not found in fixthecity.csv. Available columns: {list(fixthecity_df.columns)}")
    
    # Create combined DataFrame
    if combined_data:
        result_df = pd.DataFrame(combined_data)
        
        # NO deduplication - keep all individual transactions/rows
        logger.info(f"Combined {len(result_df)} total transaction records")
        return result_df
    else:
        logger.error("No valid data found in CSV files")
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
            autodetect=False,  # Disable autodetect to avoid schema issues
            create_disposition=bigquery.CreateDisposition.CREATE_IF_NEEDED
        )
        
        # Define schema manually to ensure all columns are treated as strings
        schema = []
        for column in df.columns:
            # Clean column names for BigQuery (remove special characters)
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
    parser = argparse.ArgumentParser(description='Analyze donors for fossil fuel connections')
    parser.add_argument('--cuomo-csv', type=str, required=True,
                       help='Path to cuomo.csv file')
    parser.add_argument('--fixthecity-csv', type=str, required=True,
                       help='Path to fixthecity.csv file')
    parser.add_argument('--fflist-csv', type=str, required=True,
                       help='Path to fflist.csv file with fossil fuel companies')
    parser.add_argument('--project-id', type=str, default='va-campaign-finance',
                       help='Google Cloud project ID (default: va-campaign-finance)')
    parser.add_argument('--dataset', type=str, default='ny_elections',
                       help='BigQuery dataset name (default: ny_elections)')
    parser.add_argument('--table', type=str, default='cuomo',
                       help='BigQuery table name (default: cuomo)')
    parser.add_argument('--fuzzy-threshold', type=int, default=95,
                       help='Fuzzy matching threshold for fossil fuel companies (default: 95)')
    parser.add_argument('--output-csv', type=str,
                       help='Optional: Save results to CSV file')
    parser.add_argument('--test-mode', action='store_true',
                       help='Test mode: only process first 10 donors')
    parser.add_argument('--debug', action='store_true',
                       help='Enable debug logging')
    
    args = parser.parse_args()
    
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        # Process CSV files
        donors_df = process_csv_files(args.cuomo_csv, args.fixthecity_csv)
        
        if donors_df.empty:
            logger.error("No data to process")
            return 1
        
        # Initialize cross-checker for fossil fuel companies
        cross_checker = FossilFuelCrossChecker(args.fflist_csv, args.fuzzy_threshold)
        
        # Analyze donors
        results = []
        total_donors = len(donors_df)
        
        if args.test_mode:
            donors_df = donors_df.head(10)
            logger.info("Test mode: Processing only first 10 donors")
        
        logger.info(f"Analyzing {len(donors_df)} donors for fossil fuel connections...")
        
        for idx, row in donors_df.iterrows():
            name = row['name']
            employer = row.get('employer', '')
            original_data = row['original_data']
            
            # Get C_CODE from original data to determine if individual
            c_code = original_data.get('C_CODE', '')
            
            logger.info(f"Processing {idx + 1}/{len(donors_df)}: {name}")
            
            # Cross-check against fossil fuel company list
            fossil_fuel_co, fossil_fuel_category, fuzzy_score = cross_checker.check_fossil_fuel_match(
                name, employer, c_code
            )
            
            # Create normalized versions for new columns
            is_individual = (str(c_code).upper() == 'IND')
            
            # Normalize NAME
            if name and pd.notna(name) and str(name).strip():
                if is_individual:
                    parsed_name = parse_lastname_firstname_name(str(name))
                    name_normalized = normalize_name(parsed_name, is_individual=True)
                else:
                    name_normalized = normalize_name(str(name), is_individual=False)
            else:
                name_normalized = ''
            
            # Normalize EMPNAME (always as organization)
            if employer and pd.notna(employer) and str(employer).strip():
                empname_normalized = normalize_name(str(employer), is_individual=False)
            else:
                empname_normalized = ''
            
            # Normalize fossil_fuel_co (always as organization)
            if fossil_fuel_co and fossil_fuel_co != 'NOPE':
                fossil_fuel_co_normalized = normalize_name(fossil_fuel_co, is_individual=False)
            else:
                fossil_fuel_co_normalized = 'NOPE'
            
            # Create analysis result with all original columns plus cross-check data
            analysis = original_data.copy()  # Start with all original columns
            
            # Add our new columns including normalized versions and fuzzy score
            analysis.update({
                'fossil_fuel_co': fossil_fuel_co,
                'fossil_fuel_category': fossil_fuel_category,
                'fuzzy_match_score': fuzzy_score,
                'source_file': row['source_file'],
                'NAME_normalized': name_normalized,
                'EMPNAME_normalized': empname_normalized,
                'fossil_fuel_co_normalized': fossil_fuel_co_normalized
            })
            
            # Debug logging to verify normalization is working
            if args.debug:
                logger.debug(f"Name: '{name}' -> '{name_normalized}'")
                logger.debug(f"Employer: '{employer}' -> '{empname_normalized}'")
                logger.debug(f"Fossil fuel co: '{fossil_fuel_co}' -> '{fossil_fuel_co_normalized}' (score: {fuzzy_score})")
            
            results.append(analysis)
            
            # Progress logging
            if (idx + 1) % 10 == 0:
                crosscheck_count = sum(1 for r in results if r['fossil_fuel_co'] != 'NOPE')
                logger.info(f"Progress: {idx + 1}/{len(donors_df)} - {crosscheck_count} fossil fuel connections found")
        
        # Create results DataFrame
        results_df = pd.DataFrame(results)
        
        # Handle data types properly for BigQuery
        for col in results_df.columns:
            # Convert to string but handle special cases
            results_df[col] = results_df[col].astype(str)
            # Replace pandas artifacts with proper NULLs
            results_df[col] = results_df[col].replace(['nan', 'None', '<NA>', 'NaT'], None)
        
        # Ensure strings are properly trimmed and cleaned
        for col in results_df.columns:
            if results_df[col].dtype == 'object':
                results_df[col] = results_df[col].apply(lambda x: x.strip() if isinstance(x, str) and x is not None else x)
        
        # Debug: Check for BlackRock entries if debug mode
        if args.debug:
            blackrock_entries = results_df[results_df['EMPNAME'].str.contains('BLACKROCK', case=False, na=False)]
            if not blackrock_entries.empty:
                logger.debug(f"Found {len(blackrock_entries)} BlackRock entries:")
                for idx, row in blackrock_entries.head(3).iterrows():
                    logger.debug(f"  EMPNAME: '{row['EMPNAME']}' (type: {type(row['EMPNAME'])}, len: {len(str(row['EMPNAME']))})")
                    logger.debug(f"  EMPNAME repr: {repr(row['EMPNAME'])}")
            else:
                logger.debug("No BlackRock entries found in EMPNAME column")
                # Show sample EMPNAME values for debugging
                sample_empnames = results_df['EMPNAME'].dropna().head(5).tolist()
                logger.debug(f"Sample EMPNAME values: {sample_empnames}")
        
        # Add summary statistics
        total_connections = len(results_df[results_df['fossil_fuel_co'] != 'NOPE'])
        
        logger.info(f"\n=== ANALYSIS COMPLETE ===")
        logger.info(f"Total donors analyzed: {len(results_df)}")
        logger.info(f"Fossil fuel connections found: {total_connections}")
        logger.info(f"Connection rate: {(total_connections/len(results_df)*100):.1f}%")
        
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