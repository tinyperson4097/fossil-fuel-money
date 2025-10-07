#!/usr/bin/env python3
"""
Fossil Fuel Leadership Extractor
Takes companies from fflist.csv and uses LittleSis API to find all leadership,
staff, and board members associated with each company.
"""

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry
import time
import argparse
import logging
from typing import Dict, List, Optional
import re

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class LittleSisLeadershipExtractor:
    """Extracts leadership and staff information from LittleSis for fossil fuel companies."""
    
    def __init__(self, rate_limit_delay: float = 1.0, max_retries: int = 3, timeout: int = 60):
        self.rate_limit_delay = rate_limit_delay
        self.max_retries = max_retries
        self.timeout = timeout
        self.littlesis_base_url = "https://littlesis.org/api"
        
        # Setup requests session with retry strategy
        self.session = requests.Session()
        retry_strategy = Retry(
            total=max_retries,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["HEAD", "GET", "OPTIONS"],  # Updated parameter name
            backoff_factor=1
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)
        
        # Cache for API responses to avoid duplicate calls
        self.search_cache = {}
        self.entity_cache = {}
        self.connections_cache = {}
        
        # Leadership/staff relationship categories from LittleSis
        self.leadership_categories = [
            1,   # Position (employment, staff positions)
            3,   # Family (sometimes includes family board members)
            9,   # Lobbying (lobbyists are often former staff)
            12   # Other (catch-all for leadership roles)
        ]
        
        logger.info(f"Initialized extractor with {rate_limit_delay}s rate limit delay, {max_retries} max retries, {timeout}s timeout")
    
    def search_littlesis(self, company_name: str) -> List[Dict]:
        """Search LittleSis for company entities by name with robust error handling."""
        if company_name in self.search_cache:
            return self.search_cache[company_name]
        
        # Clean the company name for search
        if pd.isna(company_name) or company_name is None:
            return []
        
        search_name = str(company_name).strip()
        if not search_name:
            return []
        
        url = f"{self.littlesis_base_url}/entities/search"
        params = {'q': search_name}
        
        for attempt in range(self.max_retries + 1):
            try:
                logger.debug(f"Searching LittleSis for company: {search_name} (attempt {attempt + 1}/{self.max_retries + 1})")
                
                response = self.session.get(url, params=params, timeout=self.timeout)
                
                # Rate limiting after each request
                time.sleep(self.rate_limit_delay)
                
                if response.status_code == 200:
                    data = response.json()
                    entities = data.get('data', [])
                    self.search_cache[company_name] = entities
                    return entities
                elif response.status_code == 404:
                    logger.debug(f"No results found for: {search_name}")
                    self.search_cache[company_name] = []
                    return []
                elif response.status_code == 503:
                    logger.warning(f"Rate limit exceeded for {search_name}, waiting longer...")
                    time.sleep(10)  # Longer wait for rate limits
                    continue
                elif response.status_code in [429, 500, 502, 504]:
                    logger.warning(f"Server error {response.status_code} for {search_name}, retrying...")
                    time.sleep(2 ** attempt)  # Exponential backoff
                    continue
                else:
                    logger.warning(f"LittleSis search failed for {search_name}: {response.status_code}")
                    break
                    
            except requests.exceptions.Timeout:
                logger.warning(f"Timeout searching for {search_name} (attempt {attempt + 1})")
                if attempt < self.max_retries:
                    time.sleep(5)
                    continue
                else:
                    logger.error(f"Max retries exceeded for timeout on {search_name}")
            except requests.exceptions.ConnectionError as e:
                logger.warning(f"Connection error for {search_name} (attempt {attempt + 1}): {e}")
                if attempt < self.max_retries:
                    time.sleep(10)  # Longer wait for connection errors
                    continue
                else:
                    logger.error(f"Max retries exceeded for connection error on {search_name}")
            except Exception as e:
                logger.error(f"Unexpected error searching for {company_name}: {e}")
                break
        
        # Cache empty result to avoid repeated failures
        self.search_cache[company_name] = []
        return []
    
    def get_entity_relationships(self, entity_id: int) -> List[Dict]:
        """Get relationships for a specific entity with robust error handling."""
        if entity_id in self.connections_cache:
            return self.connections_cache[entity_id]
        
        url = f"{self.littlesis_base_url}/entities/{entity_id}/relationships"
        
        for attempt in range(self.max_retries + 1):
            try:
                logger.debug(f"Getting relationships for entity {entity_id} (attempt {attempt + 1})")
                response = self.session.get(url, timeout=self.timeout)
                
                time.sleep(self.rate_limit_delay)
                
                if response.status_code == 200:
                    data = response.json()
                    relationships = data.get('data', [])
                    self.connections_cache[entity_id] = relationships
                    return relationships
                elif response.status_code == 404:
                    logger.debug(f"No relationships found for entity {entity_id}")
                    self.connections_cache[entity_id] = []
                    return []
                elif response.status_code in [429, 500, 502, 503, 504]:
                    logger.warning(f"Server error {response.status_code} for entity {entity_id}, retrying...")
                    time.sleep(2 ** attempt)
                    continue
                else:
                    logger.warning(f"Failed to get relationships for entity {entity_id}: {response.status_code}")
                    break
                    
            except requests.exceptions.Timeout:
                logger.warning(f"Timeout getting relationships for entity {entity_id} (attempt {attempt + 1})")
                if attempt < self.max_retries:
                    time.sleep(5)
                    continue
            except requests.exceptions.ConnectionError as e:
                logger.warning(f"Connection error for entity {entity_id} (attempt {attempt + 1}): {e}")
                if attempt < self.max_retries:
                    time.sleep(10)
                    continue
            except Exception as e:
                logger.error(f"Error getting relationships for entity {entity_id}: {e}")
                break
        
        self.connections_cache[entity_id] = []
        return []
    
    def get_entity_details(self, entity_id: int) -> Dict:
        """Get detailed information about a specific entity with robust error handling."""
        if entity_id in self.entity_cache:
            return self.entity_cache[entity_id]
        
        url = f"{self.littlesis_base_url}/entities/{entity_id}"
        
        for attempt in range(self.max_retries + 1):
            try:
                logger.debug(f"Getting entity details for {entity_id} (attempt {attempt + 1})")
                response = self.session.get(url, timeout=self.timeout)
                
                time.sleep(self.rate_limit_delay)
                
                if response.status_code == 200:
                    data = response.json()
                    entity_data = data.get('data', {})
                    self.entity_cache[entity_id] = entity_data
                    return entity_data
                elif response.status_code == 404:
                    logger.debug(f"Entity {entity_id} not found")
                    self.entity_cache[entity_id] = {}
                    return {}
                elif response.status_code in [429, 500, 502, 503, 504]:
                    logger.warning(f"Server error {response.status_code} for entity {entity_id}, retrying...")
                    time.sleep(2 ** attempt)
                    continue
                else:
                    logger.warning(f"Failed to get entity details for {entity_id}: {response.status_code}")
                    break
                    
            except requests.exceptions.Timeout:
                logger.warning(f"Timeout getting entity details for {entity_id} (attempt {attempt + 1})")
                if attempt < self.max_retries:
                    time.sleep(5)
                    continue
            except requests.exceptions.ConnectionError as e:
                logger.warning(f"Connection error for entity {entity_id} (attempt {attempt + 1}): {e}")
                if attempt < self.max_retries:
                    time.sleep(10)
                    continue
            except Exception as e:
                logger.error(f"Error getting entity details for {entity_id}: {e}")
                break
        
        self.entity_cache[entity_id] = {}
        return {}
    
    def extract_leadership_for_company(self, company_name: str) -> List[Dict]:
        """
        Extract all leadership, staff, and board members for a specific company.
        
        Args:
            company_name: Name of the fossil fuel company
            
        Returns:
            List of dictionaries with individual names and roles
        """
        logger.info(f"Extracting leadership for: {company_name}")
        
        leadership_results = []
        
        # Search for the company in LittleSis
        company_entities = self.search_littlesis(company_name)
        
        if not company_entities:
            logger.warning(f"No LittleSis entities found for: {company_name}")
            return []
        
        # Process each potential company match
        for entity in company_entities:
            entity_id = entity.get('id')
            entity_name = entity.get('attributes', {}).get('name', '')
            entity_types = entity.get('attributes', {}).get('types', [])
            
            logger.debug(f"Checking entity: {entity_name} (ID: {entity_id}) - Types: {entity_types}")
            
            # Skip if this doesn't look like a company/organization
            if not any(org_type in entity_types for org_type in ['Organization', 'Business', 'Public Company', 'Private Company']):
                logger.debug(f"Skipping {entity_name} - not a company type")
                continue
            
            # Get all relationships for this company
            relationships = self.get_entity_relationships(entity_id)
            
            logger.debug(f"Found {len(relationships)} relationships for {entity_name}")
            
            # Process each relationship to find leadership roles
            for relationship in relationships:
                rel_attributes = relationship.get('attributes', {})
                category_id = rel_attributes.get('category_id')
                
                # Focus on Position relationships (category 1) which include employment/leadership
                if category_id == 1:  # Position category
                    # Get the connected person's information
                    entity1_id = rel_attributes.get('entity1_id')
                    entity2_id = rel_attributes.get('entity2_id')
                    
                    # Determine which entity is the person (not the company we're analyzing)
                    person_id = entity1_id if entity1_id != entity_id else entity2_id
                    
                    if person_id:
                        person_details = self.get_entity_details(person_id)
                        person_attributes = person_details.get('attributes', {})
                        person_name = person_attributes.get('name', '')
                        person_types = person_attributes.get('types', [])
                        
                        # Only include if this is actually a person
                        if 'Person' in person_types and person_name:
                            role_description = rel_attributes.get('description1', '')
                            start_date = rel_attributes.get('start_date', '')
                            end_date = rel_attributes.get('end_date', '')
                            is_current = rel_attributes.get('is_current')
                            
                            # Determine if this is a leadership role
                            role_lower = role_description.lower() if role_description else ''
                            leadership_keywords = [
                                'ceo', 'president', 'chairman', 'director', 'executive', 
                                'vice president', 'vp', 'chief', 'board', 'officer',
                                'manager', 'head', 'senior', 'lead', 'founder'
                            ]
                            
                            is_leadership = any(keyword in role_lower for keyword in leadership_keywords)
                            
                            leadership_results.append({
                                'company': company_name,
                                'individual': person_name,
                                'role': role_description,
                                'start_date': start_date,
                                'end_date': end_date,
                                'is_current': is_current,
                                'is_leadership': is_leadership,
                                'littlesis_person_id': person_id,
                                'littlesis_company_id': entity_id,
                                'company_matched_name': entity_name
                            })
                            
                            logger.debug(f"Found: {person_name} - {role_description} at {entity_name}")
        
        logger.info(f"Found {len(leadership_results)} leadership/staff connections for {company_name}")
        return leadership_results
    
    def extract_all_leadership(self, companies_list: List[str]) -> List[Dict]:
        """
        Extract leadership for all companies in the list.
        
        Args:
            companies_list: List of company names from fflist.csv
            
        Returns:
            Combined list of all leadership/staff connections
        """
        all_leadership = []
        
        for i, company_name in enumerate(companies_list, 1):
            logger.info(f"Processing company {i}/{len(companies_list)}: {company_name}")
            
            try:
                company_leadership = self.extract_leadership_for_company(company_name)
                all_leadership.extend(company_leadership)
                
                # Progress logging
                if i % 5 == 0:
                    logger.info(f"Progress: {i}/{len(companies_list)} companies processed. Total connections found: {len(all_leadership)}")
                    
            except Exception as e:
                logger.error(f"Error processing {company_name}: {e}")
                continue
        
        return all_leadership


def load_fossil_fuel_companies(fflist_csv_path: str) -> List[str]:
    """Load company names from fflist.csv."""
    try:
        ff_df = pd.read_csv(fflist_csv_path)
        logger.info(f"Loaded {len(ff_df)} companies from {fflist_csv_path}")
        
        # Find the company name column
        company_col = None
        for col in ff_df.columns:
            if 'company' in col.lower() or 'name' in col.lower():
                company_col = col
                break
        
        if not company_col:
            # Use first column as company name
            company_col = ff_df.columns[0]
            logger.warning(f"No company column found, using first column: {company_col}")
        
        # Extract company names, removing duplicates and empty values
        companies = ff_df[company_col].dropna().astype(str).str.strip()
        companies = companies[companies != ''].unique().tolist()
        
        logger.info(f"Found {len(companies)} unique company names")
        return companies
        
    except Exception as e:
        logger.error(f"Error loading fossil fuel companies from {fflist_csv_path}: {e}")
        return []


def main():
    parser = argparse.ArgumentParser(description='Extract fossil fuel company leadership from LittleSis')
    parser.add_argument('--fflist-csv', type=str, required=True,
                       help='Path to fflist.csv file with fossil fuel companies')
    parser.add_argument('--output-csv', type=str, required=True,
                       help='Path to output CSV file for leadership data')
    parser.add_argument('--rate-limit', type=float, default=2.0,
                       help='Rate limit delay in seconds (default: 2.0)')
    parser.add_argument('--max-retries', type=int, default=3,
                       help='Maximum number of retries for failed requests (default: 3)')
    parser.add_argument('--timeout', type=int, default=60,
                       help='Request timeout in seconds (default: 60)')
    parser.add_argument('--test-mode', action='store_true',
                       help='Test mode: only process first 5 companies')
    parser.add_argument('--debug', action='store_true',
                       help='Enable debug logging')
    
    args = parser.parse_args()
    
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        # Load fossil fuel companies
        companies = load_fossil_fuel_companies(args.fflist_csv)
        
        if not companies:
            logger.error("No companies found to process")
            return 1
        
        if args.test_mode:
            companies = companies[:5]
            logger.info("Test mode: Processing only first 5 companies")
        
        # Initialize extractor
        extractor = LittleSisLeadershipExtractor(
            rate_limit_delay=args.rate_limit,
            max_retries=args.max_retries,
            timeout=args.timeout
        )
        
        # Extract leadership for all companies
        logger.info(f"Starting leadership extraction for {len(companies)} companies...")
        all_leadership = extractor.extract_all_leadership(companies)
        
        # Create results DataFrame
        if all_leadership:
            results_df = pd.DataFrame(all_leadership)
            
            # Save to CSV
            results_df.to_csv(args.output_csv, index=False)
            logger.info(f"Successfully saved {len(results_df)} leadership connections to {args.output_csv}")
            
            # Summary statistics
            unique_companies = results_df['company'].nunique()
            unique_individuals = results_df['individual'].nunique()
            leadership_count = len(results_df[results_df['is_leadership']])
            
            print(f"\n✅ Leadership Extraction Complete!")
            print(f"Companies with connections found: {unique_companies}")
            print(f"Total individual connections: {len(results_df)}")
            print(f"Unique individuals: {unique_individuals}")
            print(f"Leadership roles: {leadership_count}")
            print(f"Non-leadership staff: {len(results_df) - leadership_count}")
            
            # Show top companies by connection count
            print(f"\n🏢 Top 10 Companies by Connection Count:")
            top_companies = results_df['company'].value_counts().head(10)
            for company, count in top_companies.items():
                print(f"  {company}: {count} connections")
        
        else:
            logger.warning("No leadership connections found")
            # Create empty CSV with headers
            empty_df = pd.DataFrame(columns=[
                'company', 'individual', 'role', 'start_date', 'end_date', 
                'is_current', 'is_leadership', 'littlesis_person_id', 
                'littlesis_company_id', 'company_matched_name'
            ])
            empty_df.to_csv(args.output_csv, index=False)
            print("No connections found - empty CSV created")
        
        return 0
        
    except Exception as e:
        logger.error(f"Script failed: {e}")
        return 1


if __name__ == "__main__":
    exit(main())