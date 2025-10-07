#!/usr/bin/env python3
"""
Fossil Fuel Donor Connection Analyzer
Processes donor lists from CSV files and identifies fossil fuel connections
using LittleSis API and LinkedIn lookup.
"""

import pandas as pd
import requests
import time
import argparse
import logging
from google.cloud import bigquery
from typing import Dict, List, Optional, Tuple
import re
from urllib.parse import quote
import json

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class FossilFuelAnalyzer:
    """Analyzes donors for fossil fuel connections using LittleSis API and LinkedIn lookup."""
    
    def __init__(self, rate_limit_delay: float = 1.0):
        self.rate_limit_delay = rate_limit_delay
        self.littlesis_base_url = "https://littlesis.org/api"
        
        # Cache for API responses to avoid duplicate calls
        self.search_cache = {}
        self.entity_cache = {}
        
        # Fossil fuel related keywords for company identification
        self.fossil_fuel_keywords = [
            'oil', 'gas', 'petroleum', 'energy', 'coal', 'exxon', 'chevron', 'bp', 'shell',
            'conocophillips', 'valero', 'marathon', 'phillips', 'texaco', 'mobil', 'citgo',
            'sunoco', 'hess', 'occidental', 'anadarko', 'devon', 'chesapeake', 'kinder morgan',
            'enbridge', 'enterprise products', 'plains all american', 'spectra energy',
            'williams companies', 'oneok', 'tc energy', 'keystone', 'pipeline', 'refining',
            'drilling', 'fracking', 'hydraulic fracturing', 'lng', 'natural gas',
            'fossil fuel', 'carbon', 'hydrocarbon', 'upstream', 'downstream', 'midstream'
        ]
        
        logger.info(f"Initialized analyzer with {rate_limit_delay}s rate limit delay")
    
    def search_littlesis(self, name: str) -> List[Dict]:
        """Search LittleSis for entities by name."""
        if name in self.search_cache:
            return self.search_cache[name]
        
        try:
            # Clean the name for search - handle non-string values
            if pd.isna(name) or name is None:
                return []
            
            search_name = str(name).strip()
            if not search_name:
                return []
            
            url = f"{self.littlesis_base_url}/entities/search"
            params = {'q': search_name}
            
            logger.debug(f"Searching LittleSis for: {search_name}")
            response = requests.get(url, params=params, timeout=30)
            
            # Rate limiting
            time.sleep(self.rate_limit_delay)
            
            if response.status_code == 200:
                data = response.json()
                entities = data.get('data', [])
                self.search_cache[name] = entities
                return entities
            elif response.status_code == 404:
                logger.debug(f"No results found for: {search_name}")
                self.search_cache[name] = []
                return []
            elif response.status_code == 503:
                logger.warning("Rate limit exceeded, waiting longer...")
                time.sleep(5)
                return self.search_littlesis(name)  # Retry
            else:
                logger.warning(f"LittleSis search failed for {search_name}: {response.status_code}")
                return []
                
        except Exception as e:
            logger.error(f"Error searching LittleSis for {name}: {e}")
            return []
    
    def get_entity_connections(self, entity_id: int) -> List[Dict]:
        """Get connections for a specific entity."""
        if entity_id in self.entity_cache:
            return self.entity_cache[entity_id]
        
        try:
            url = f"{self.littlesis_base_url}/entities/{entity_id}/connections"
            
            logger.debug(f"Getting connections for entity {entity_id}")
            response = requests.get(url, timeout=30)
            
            # Rate limiting
            time.sleep(self.rate_limit_delay)
            
            if response.status_code == 200:
                data = response.json()
                connections = data.get('data', [])
                self.entity_cache[entity_id] = connections
                return connections
            else:
                logger.debug(f"Failed to get connections for entity {entity_id}: {response.status_code}")
                return []
                
        except Exception as e:
            logger.error(f"Error getting connections for entity {entity_id}: {e}")
            return []
    
    def check_fossil_fuel_connection(self, name: str, employer: str = "") -> Dict:
        """
        Check if a person has fossil fuel connections via LittleSis.
        
        Args:
            name: Person's name
            employer: Person's employer (if available)
            
        Returns:
            Dict with connection details
        """
        result = {
            'has_fossil_fuel_connection': False,
            'connection_type': None,
            'connection_details': [],
            'littlesis_entity_id': None,
            'search_method': 'littlesis'
        }
        
        # Search for the person in LittleSis
        entities = self.search_littlesis(name)
        
        if not entities:
            # If no direct match, try with employer if provided
            if employer and pd.notna(employer) and str(employer).strip():
                entities = self.search_littlesis(f"{name} {str(employer).strip()}")
        
        for entity in entities:
            entity_id = entity.get('id')
            entity_name = entity.get('attributes', {}).get('name', '')
            entity_blurb = entity.get('attributes', {}).get('blurb', '')
            entity_types = entity.get('attributes', {}).get('types', [])
            
            logger.debug(f"Checking entity: {entity_name} (ID: {entity_id})")
            
            # Check if the entity itself has fossil fuel keywords
            text_to_check = f"{entity_name} {entity_blurb}".lower()
            fossil_fuel_match = any(keyword in text_to_check for keyword in self.fossil_fuel_keywords)
            
            if fossil_fuel_match:
                result['has_fossil_fuel_connection'] = True
                result['connection_type'] = 'direct_entity'
                result['connection_details'].append({
                    'entity_name': entity_name,
                    'entity_id': entity_id,
                    'entity_blurb': entity_blurb,
                    'match_reason': 'fossil_fuel_keywords_in_entity'
                })
                result['littlesis_entity_id'] = entity_id
                continue
            
            # Check entity connections
            connections = self.get_entity_connections(entity_id)
            
            for connection in connections:
                conn_attributes = connection.get('attributes', {})
                conn_name = conn_attributes.get('name', '')
                conn_blurb = conn_attributes.get('blurb', '')
                conn_types = conn_attributes.get('types', [])
                
                # Check for fossil fuel connections
                conn_text = f"{conn_name} {conn_blurb}".lower()
                if any(keyword in conn_text for keyword in self.fossil_fuel_keywords):
                    result['has_fossil_fuel_connection'] = True
                    result['connection_type'] = 'connected_entity'
                    result['connection_details'].append({
                        'connected_entity_name': conn_name,
                        'connected_entity_blurb': conn_blurb,
                        'connected_entity_types': conn_types,
                        'relationship_category_id': conn_attributes.get('relationship_category_id'),
                        'through_entity': entity_name
                    })
                    if not result['littlesis_entity_id']:
                        result['littlesis_entity_id'] = entity_id
        
        return result
    
    def check_linkedin_fossil_fuel_connection(self, name: str, employer: str = "") -> Dict:
        """
        Check for fossil fuel connections via LinkedIn/employer lookup.
        This is a simplified version - in practice you'd use LinkedIn API or web scraping.
        """
        result = {
            'has_fossil_fuel_connection': False,
            'connection_type': None,
            'connection_details': [],
            'search_method': 'employer_lookup'
        }
        
        # Handle non-string employer values
        if pd.isna(employer) or employer is None:
            return result
        
        employer_str = str(employer).strip()
        if not employer_str:
            return result
        
        # Check if employer name contains fossil fuel keywords
        employer_lower = employer_str.lower()
        fossil_fuel_match = any(keyword in employer_lower for keyword in self.fossil_fuel_keywords)
        
        if fossil_fuel_match:
            result['has_fossil_fuel_connection'] = True
            result['connection_type'] = 'employer'
            result['connection_details'].append({
                'employer_name': employer,
                'match_reason': 'fossil_fuel_keywords_in_employer'
            })
        
        return result
    
    def analyze_donor(self, name: str, employer: str = "") -> Dict:
        """
        Analyze a donor for fossil fuel connections using both methods.
        
        Args:
            name: Donor's name
            employer: Donor's employer (if available)
            
        Returns:
            Dict with comprehensive analysis results
        """
        logger.debug(f"Analyzing donor: {name} (employer: {employer})")
        
        # Try LittleSis first
        littlesis_result = self.check_fossil_fuel_connection(name, employer)
        
        # If no connection found via LittleSis, try employer lookup
        if not littlesis_result['has_fossil_fuel_connection']:
            employer_result = self.check_linkedin_fossil_fuel_connection(name, employer)
            
            if employer_result['has_fossil_fuel_connection']:
                return {
                    'name': str(name) if pd.notna(name) else '',
                    'employer': str(employer) if pd.notna(employer) else '',
                    'has_fossil_fuel_connection': True,
                    'primary_connection_method': 'employer_lookup',
                    'connection_type': employer_result['connection_type'],
                    'connection_details': employer_result['connection_details'],
                    'littlesis_checked': True,
                    'littlesis_entity_id': None
                }
        
        return {
            'name': str(name) if pd.notna(name) else '',
            'employer': str(employer) if pd.notna(employer) else '',
            'has_fossil_fuel_connection': littlesis_result['has_fossil_fuel_connection'],
            'primary_connection_method': 'littlesis' if littlesis_result['has_fossil_fuel_connection'] else 'none',
            'connection_type': littlesis_result['connection_type'],
            'connection_details': littlesis_result['connection_details'],
            'littlesis_checked': True,
            'littlesis_entity_id': littlesis_result['littlesis_entity_id']
        }

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
    if not cuomo_df.empty and 'NAME' in cuomo_df.columns:
        for _, row in cuomo_df.iterrows():
            combined_data.append({
                'name': row['NAME'],
                'source_file': 'cuomo.csv',
                'employer': row.get('EMPNAME', '') if 'EMPNAME' in row else '',
                'original_data': row.to_dict()
            })
    
    # Process FixTheCity CSV
    if not fixthecity_df.empty and 'NAME' in fixthecity_df.columns:
        for _, row in fixthecity_df.iterrows():
            combined_data.append({
                'name': row['NAME'],
                'source_file': 'fixthecity.csv',
                'employer': row.get('EMPNAME', '') if 'EMPNAME' in row else '',
                'original_data': row.to_dict()
            })
    
    # Create combined DataFrame
    if combined_data:
        result_df = pd.DataFrame(combined_data)
        
        # Remove duplicates based on name (case-insensitive)
        result_df['name_lower'] = result_df['name'].str.lower().str.strip()
        result_df = result_df.drop_duplicates(subset=['name_lower'])
        result_df = result_df.drop('name_lower', axis=1)
        
        logger.info(f"Combined and deduplicated to {len(result_df)} unique donors")
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
            autodetect=True
        )
        
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
    parser.add_argument('--project-id', type=str, default='va-campaign-finance',
                       help='Google Cloud project ID (default: va-campaign-finance)')
    parser.add_argument('--dataset', type=str, default='ny_elections',
                       help='BigQuery dataset name (default: ny_elections)')
    parser.add_argument('--table', type=str, default='cuomo',
                       help='BigQuery table name (default: cuomo)')
    parser.add_argument('--rate-limit', type=float, default=1.0,
                       help='Rate limit delay in seconds (default: 1.0)')
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
        
        # Initialize analyzer
        analyzer = FossilFuelAnalyzer(rate_limit_delay=args.rate_limit)
        
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
            
            logger.info(f"Processing {idx + 1}/{len(donors_df)}: {name}")
            
            # Analyze donor
            analysis = analyzer.analyze_donor(name, employer)
            
            # Add original data
            analysis.update({
                'source_file': row['source_file'],
                'original_data_json': json.dumps(row['original_data'])
            })
            
            results.append(analysis)
            
            # Progress logging
            if (idx + 1) % 10 == 0:
                fossil_fuel_count = sum(1 for r in results if r['has_fossil_fuel_connection'])
                logger.info(f"Progress: {idx + 1}/{len(donors_df)} - {fossil_fuel_count} with fossil fuel connections found so far")
        
        # Create results DataFrame
        results_df = pd.DataFrame(results)
        
        # Add summary statistics
        total_connections = len(results_df[results_df['has_fossil_fuel_connection']])
        logger.info(f"\n=== ANALYSIS COMPLETE ===")
        logger.info(f"Total donors analyzed: {len(results_df)}")
        logger.info(f"Donors with fossil fuel connections: {total_connections}")
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