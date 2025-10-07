#!/usr/bin/env python3
"""
Fast Fossil Fuel Leadership Extractor
Ultra-optimized version using async/await, connection pooling, intelligent caching,
and concurrent processing for maximum speed.
"""

import pandas as pd
import aiohttp
import asyncio
import time
import argparse
import logging
from typing import Dict, List, Optional, Set
import json
import pickle
from pathlib import Path
import hashlib

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class FastLittleSisExtractor:
    """Ultra-fast async LittleSis extractor with intelligent caching and concurrent processing."""
    
    def __init__(
        self, 
        concurrent_companies: int = 10,
        concurrent_requests: int = 20, 
        rate_limit: float = 0.1,
        cache_dir: str = "./littlesis_cache",
        resume_file: str = "./extraction_progress.json"
    ):
        self.concurrent_companies = concurrent_companies
        self.concurrent_requests = concurrent_requests  
        self.rate_limit = rate_limit
        self.cache_dir = Path(cache_dir)
        self.resume_file = Path(resume_file)
        self.littlesis_base_url = "https://littlesis.org/api"
        
        # Create cache directory
        self.cache_dir.mkdir(exist_ok=True)
        
        # In-memory caches
        self.search_cache = {}
        self.entity_cache = {}
        self.relationships_cache = {}
        
        # Progress tracking
        self.processed_companies = set()
        self.failed_companies = set()
        
        # Statistics
        self.stats = {
            'companies_processed': 0,
            'api_calls_made': 0,
            'cache_hits': 0,
            'leadership_found': 0,
            'start_time': time.time()
        }
        
        logger.info(f"Initialized fast extractor:")
        logger.info(f"  Concurrent companies: {concurrent_companies}")
        logger.info(f"  Concurrent requests: {concurrent_requests}")
        logger.info(f"  Rate limit: {rate_limit}s")
        logger.info(f"  Cache directory: {cache_dir}")
    
    def create_session(self) -> aiohttp.ClientSession:
        """Create optimized aiohttp session with aggressive timeouts."""
        timeout = aiohttp.ClientTimeout(total=15, connect=5)  # Much more aggressive timeouts
        connector = aiohttp.TCPConnector(
            limit=self.concurrent_requests * 3,
            limit_per_host=self.concurrent_requests * 2,
            keepalive_timeout=10,
            enable_cleanup_closed=True,
            ttl_dns_cache=300,  # Cache DNS for 5 minutes
            use_dns_cache=True
        )
        
        return aiohttp.ClientSession(
            connector=connector,
            timeout=timeout,
            headers={'User-Agent': 'FossilFuelLeadershipExtractor/2.0'}
        )
    
    def get_cache_key(self, cache_type: str, identifier: str) -> str:
        """Generate cache key for different types of data."""
        return hashlib.md5(f"{cache_type}_{identifier}".encode()).hexdigest()
    
    async def load_cache(self, cache_type: str, identifier: str) -> Optional[Dict]:
        """Load data from disk cache if available."""
        cache_key = self.get_cache_key(cache_type, identifier)
        cache_file = self.cache_dir / f"{cache_key}.pkl"
        
        if cache_file.exists():
            try:
                with open(cache_file, 'rb') as f:
                    data = pickle.load(f)
                self.stats['cache_hits'] += 1
                return data
            except Exception as e:
                logger.debug(f"Cache read error for {cache_key}: {e}")
        
        return None
    
    async def save_cache(self, cache_type: str, identifier: str, data: Dict):
        """Save data to disk cache."""
        cache_key = self.get_cache_key(cache_type, identifier)
        cache_file = self.cache_dir / f"{cache_key}.pkl"
        
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(data, f)
        except Exception as e:
            logger.debug(f"Cache write error for {cache_key}: {e}")
    
    async def load_progress(self):
        """Load previous progress to enable resume."""
        if self.resume_file.exists():
            try:
                with open(self.resume_file, 'r') as f:
                    progress = json.load(f)
                    self.processed_companies = set(progress.get('processed', []))
                    self.failed_companies = set(progress.get('failed', []))
                logger.info(f"Loaded progress: {len(self.processed_companies)} processed, {len(self.failed_companies)} failed")
            except Exception as e:
                logger.warning(f"Could not load progress: {e}")
    
    async def save_progress(self):
        """Save current progress."""
        try:
            progress = {
                'processed': list(self.processed_companies),
                'failed': list(self.failed_companies),
                'timestamp': time.time()
            }
            with open(self.resume_file, 'w') as f:
                json.dump(progress, f)
        except Exception as e:
            logger.warning(f"Could not save progress: {e}")
    
    async def rate_limited_request(self, session: aiohttp.ClientSession, url: str, params: Dict = None) -> Optional[Dict]:
        """Make rate-limited HTTP request with FAIL-FAST approach."""
        # Rate limiting
        await asyncio.sleep(self.rate_limit)
        
        try:
            self.stats['api_calls_made'] += 1
            async with session.get(url, params=params) as response:
                if response.status == 200:
                    return await response.json()
                elif response.status == 404:
                    return {'data': []}  # No results found
                elif response.status == 429:
                    # Rate limited - wait briefly and fail
                    logger.debug(f"Rate limited on {url} - skipping")
                    await asyncio.sleep(1)
                    return None
                elif response.status >= 500:
                    # Server error - fail immediately, don't retry
                    logger.debug(f"Server error {response.status} on {url} - skipping")
                    return None
                else:
                    logger.debug(f"HTTP {response.status} for {url} - skipping")
                    return None
                    
        except asyncio.TimeoutError:
            logger.debug(f"Timeout for {url} - skipping")
            return None
        except Exception as e:
            logger.debug(f"Request error for {url}: {e} - skipping")
            return None
    
    async def search_company(self, session: aiohttp.ClientSession, company_name: str) -> List[Dict]:
        """Search for company with caching."""
        # Check in-memory cache first
        if company_name in self.search_cache:
            return self.search_cache[company_name]
        
        # Check disk cache
        cached_data = await self.load_cache('search', company_name)
        if cached_data is not None:
            self.search_cache[company_name] = cached_data
            return cached_data
        
        # Make API call
        url = f"{self.littlesis_base_url}/entities/search"
        params = {'q': company_name}
        
        logger.debug(f"Searching for: {company_name}")
        result = await self.rate_limited_request(session, url, params)
        
        if result:
            entities = result.get('data', [])
            # Cache results
            self.search_cache[company_name] = entities
            await self.save_cache('search', company_name, entities)
            return entities
        
        return []
    
    async def get_entity_relationships(self, session: aiohttp.ClientSession, entity_id: int) -> List[Dict]:
        """Get entity relationships with caching."""
        # Check in-memory cache first
        if entity_id in self.relationships_cache:
            return self.relationships_cache[entity_id]
        
        # Check disk cache
        cached_data = await self.load_cache('relationships', str(entity_id))
        if cached_data is not None:
            self.relationships_cache[entity_id] = cached_data
            return cached_data
        
        # Make API call
        url = f"{self.littlesis_base_url}/entities/{entity_id}/relationships"
        
        logger.debug(f"Getting relationships for entity: {entity_id}")
        result = await self.rate_limited_request(session, url)
        
        if result:
            relationships = result.get('data', [])
            # Cache results
            self.relationships_cache[entity_id] = relationships
            await self.save_cache('relationships', str(entity_id), relationships)
            return relationships
        
        return []
    
    async def get_entity_details(self, session: aiohttp.ClientSession, entity_id: int) -> Dict:
        """Get entity details with caching."""
        # Check in-memory cache first
        if entity_id in self.entity_cache:
            return self.entity_cache[entity_id]
        
        # Check disk cache
        cached_data = await self.load_cache('entity', str(entity_id))
        if cached_data is not None:
            self.entity_cache[entity_id] = cached_data
            return cached_data
        
        # Make API call
        url = f"{self.littlesis_base_url}/entities/{entity_id}"
        
        logger.debug(f"Getting entity details: {entity_id}")
        result = await self.rate_limited_request(session, url)
        
        if result:
            entity_data = result.get('data', {})
            # Cache results
            self.entity_cache[entity_id] = entity_data
            await self.save_cache('entity', str(entity_id), entity_data)
            return entity_data
        
        return {}
    
    async def extract_leadership_for_company(self, session: aiohttp.ClientSession, company_name: str) -> List[Dict]:
        """Extract leadership for a single company with optimizations."""
        
        # Skip if already processed
        if company_name in self.processed_companies:
            logger.debug(f"Skipping already processed company: {company_name}")
            return []
        
        if company_name in self.failed_companies:
            logger.debug(f"Skipping previously failed company: {company_name}")
            return []
        
        # Skip very short company names (likely to be noise)
        if len(company_name.strip()) < 3:
            logger.debug(f"Skipping too-short company name: {company_name}")
            self.failed_companies.add(company_name)
            return []
        
        try:
            logger.debug(f"Processing company: {company_name}")
            leadership_results = []
            
            # Step 1: Search for company
            entities = await self.search_company(session, company_name)
            
            if not entities:
                logger.debug(f"No entities found for: {company_name}")
                self.failed_companies.add(company_name)
                return []
            
            # Step 2: Filter to organization-type entities only
            org_entities = []
            for entity in entities:
                entity_types = entity.get('attributes', {}).get('types', [])
                if any(org_type in entity_types for org_type in ['Organization', 'Business', 'Public Company', 'Private Company']):
                    org_entities.append(entity)
            
            if not org_entities:
                logger.debug(f"No organization entities found for: {company_name}")
                self.failed_companies.add(company_name)
                return []
            
            # Step 3: Process relationships for each organization entity concurrently
            relationship_tasks = []
            for entity in org_entities:
                entity_id = entity.get('id')
                if entity_id:
                    task = self.process_entity_relationships(session, entity, company_name)
                    relationship_tasks.append(task)
            
            # Wait for all relationship processing to complete
            if relationship_tasks:
                results = await asyncio.gather(*relationship_tasks, return_exceptions=True)
                
                for result in results:
                    if isinstance(result, list):
                        leadership_results.extend(result)
                    elif isinstance(result, Exception):
                        logger.error(f"Error processing relationships for {company_name}: {result}")
            
            self.processed_companies.add(company_name)
            self.stats['companies_processed'] += 1
            self.stats['leadership_found'] += len(leadership_results)
            
            logger.debug(f"Found {len(leadership_results)} leadership connections for {company_name}")
            return leadership_results
            
        except Exception as e:
            logger.error(f"Error processing company {company_name}: {e}")
            self.failed_companies.add(company_name)
            return []
    
    async def process_entity_relationships(self, session: aiohttp.ClientSession, entity: Dict, company_name: str) -> List[Dict]:
        """Process relationships for a single entity."""
        leadership_results = []
        entity_id = entity.get('id')
        entity_name = entity.get('attributes', {}).get('name', '')
        
        # Get relationships
        relationships = await self.get_entity_relationships(session, entity_id)
        
        # Filter to Position relationships only (category 1)
        position_relationships = [
            rel for rel in relationships 
            if rel.get('attributes', {}).get('category_id') == 1
        ]
        
        if not position_relationships:
            return []
        
        # Process person details concurrently
        person_tasks = []
        for relationship in position_relationships:
            rel_attributes = relationship.get('attributes', {})
            entity1_id = rel_attributes.get('entity1_id')
            entity2_id = rel_attributes.get('entity2_id')
            
            # Determine which entity is the person
            person_id = entity1_id if entity1_id != entity_id else entity2_id
            
            if person_id:
                task = self.process_person_details(
                    session, person_id, relationship, entity_name, company_name
                )
                person_tasks.append(task)
        
        # Wait for all person processing to complete
        if person_tasks:
            results = await asyncio.gather(*person_tasks, return_exceptions=True)
            
            for result in results:
                if isinstance(result, dict):
                    leadership_results.append(result)
                elif isinstance(result, Exception):
                    logger.debug(f"Error processing person: {result}")
        
        return leadership_results
    
    async def process_person_details(
        self, 
        session: aiohttp.ClientSession, 
        person_id: int, 
        relationship: Dict, 
        entity_name: str, 
        company_name: str
    ) -> Optional[Dict]:
        """Process details for a single person."""
        try:
            # Get person details
            person_details = await self.get_entity_details(session, person_id)
            person_attributes = person_details.get('attributes', {})
            person_name = person_attributes.get('name', '')
            person_types = person_attributes.get('types', [])
            
            # Only include if this is actually a person
            if 'Person' not in person_types or not person_name:
                return None
            
            # Extract relationship details
            rel_attributes = relationship.get('attributes', {})
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
            
            return {
                'company': company_name,
                'individual': person_name,
                'role': role_description,
                'start_date': start_date,
                'end_date': end_date,
                'is_current': is_current,
                'is_leadership': is_leadership,
                'littlesis_person_id': person_id,
                'littlesis_company_id': rel_attributes.get('entity1_id') if rel_attributes.get('entity2_id') == person_id else rel_attributes.get('entity2_id'),
                'company_matched_name': entity_name
            }
            
        except Exception as e:
            logger.debug(f"Error processing person {person_id}: {e}")
            return None
    
    async def extract_all_leadership(self, companies: List[str]) -> List[Dict]:
        """Extract leadership for all companies with maximum concurrency."""
        await self.load_progress()
        
        # Filter out already processed companies
        remaining_companies = [c for c in companies if c not in self.processed_companies and c not in self.failed_companies]
        
        logger.info(f"Processing {len(remaining_companies)} remaining companies (skipped {len(companies) - len(remaining_companies)} already processed)")
        
        if not remaining_companies:
            logger.info("All companies already processed!")
            return []
        
        all_leadership = []
        
        session = self.create_session()
        try:
            # Process companies in batches
            for i in range(0, len(remaining_companies), self.concurrent_companies):
                batch = remaining_companies[i:i + self.concurrent_companies]
                
                logger.info(f"Processing batch {i//self.concurrent_companies + 1}/{(len(remaining_companies) + self.concurrent_companies - 1)//self.concurrent_companies}: {len(batch)} companies")
                batch_start = time.time()
                
                # Process batch concurrently
                tasks = [
                    self.extract_leadership_for_company(session, company_name)
                    for company_name in batch
                ]
                
                batch_results = await asyncio.gather(*tasks, return_exceptions=True)
                
                # Collect results
                for result in batch_results:
                    if isinstance(result, list):
                        all_leadership.extend(result)
                    elif isinstance(result, Exception):
                        logger.error(f"Batch processing error: {result}")
                
                # Save progress periodically
                await self.save_progress()
                
                # Progress update
                processed_count = self.stats['companies_processed']
                total_leadership = len(all_leadership)
                elapsed_time = time.time() - self.stats['start_time']
                batch_time = time.time() - batch_start
                rate = processed_count / elapsed_time if elapsed_time > 0 else 0
                batch_rate = len(batch) / batch_time if batch_time > 0 else 0
                
                logger.info(f"Batch completed in {batch_time:.1f}s ({batch_rate:.1f} companies/sec)")
                logger.info(f"Overall: {processed_count}/{len(companies)} companies, {total_leadership} leadership, {rate:.1f} companies/sec")
                logger.info(f"API: {self.stats['api_calls_made']} calls, {self.stats['cache_hits']} cache hits, {(self.stats['cache_hits']/(self.stats['api_calls_made']+self.stats['cache_hits'])*100):.1f}% cache rate")
        
        finally:
            await session.close()
        
        return all_leadership
    
    async def cleanup(self):
        """Cleanup resources."""
        await self.save_progress()


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
            company_col = ff_df.columns[0]
            logger.warning(f"No company column found, using first column: {company_col}")
        
        # Extract company names
        companies = ff_df[company_col].dropna().astype(str).str.strip()
        companies = companies[companies != ''].unique().tolist()
        
        logger.info(f"Found {len(companies)} unique company names")
        return companies
        
    except Exception as e:
        logger.error(f"Error loading companies from {fflist_csv_path}: {e}")
        return []


async def main():
    parser = argparse.ArgumentParser(description='Fast fossil fuel leadership extraction')
    parser.add_argument('--fflist-csv', type=str, required=True,
                       help='Path to fflist.csv file with fossil fuel companies')
    parser.add_argument('--output-csv', type=str, required=True,
                       help='Path to output CSV file for leadership data')
    parser.add_argument('--concurrent-companies', type=int, default=10,
                       help='Number of companies to process concurrently (default: 10)')
    parser.add_argument('--concurrent-requests', type=int, default=20,
                       help='Number of concurrent HTTP requests (default: 20)')
    parser.add_argument('--rate-limit', type=float, default=0.1,
                       help='Rate limit delay in seconds (default: 0.1)')
    parser.add_argument('--cache-dir', type=str, default='./littlesis_cache',
                       help='Directory for caching API results')
    parser.add_argument('--resume-file', type=str, default='./extraction_progress.json',
                       help='File for saving/loading progress')
    parser.add_argument('--test-mode', action='store_true',
                       help='Test mode: only process first 5 companies')
    parser.add_argument('--debug', action='store_true',
                       help='Enable debug logging')
    
    args = parser.parse_args()
    
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        start_time = time.time()
        
        # Load companies
        companies = load_fossil_fuel_companies(args.fflist_csv)
        if not companies:
            logger.error("No companies found to process")
            return 1
        
        if args.test_mode:
            companies = companies[:5]
            logger.info("Test mode: Processing only first 5 companies")
        
        # Initialize fast extractor
        extractor = FastLittleSisExtractor(
            concurrent_companies=args.concurrent_companies,
            concurrent_requests=args.concurrent_requests,
            rate_limit=args.rate_limit,
            cache_dir=args.cache_dir,
            resume_file=args.resume_file
        )
        
        # Extract leadership
        logger.info(f"Starting fast leadership extraction for {len(companies)} companies...")
        all_leadership = await extractor.extract_all_leadership(companies)
        
        # Cleanup
        await extractor.cleanup()
        
        # Save results
        if all_leadership:
            results_df = pd.DataFrame(all_leadership)
            results_df.to_csv(args.output_csv, index=False)
            
            elapsed_time = time.time() - start_time
            unique_companies = results_df['company'].nunique()
            unique_individuals = results_df['individual'].nunique()
            leadership_count = len(results_df[results_df['is_leadership']])
            
            logger.info(f"✅ Fast Extraction Complete!")
            logger.info(f"Total time: {elapsed_time:.1f} seconds")
            logger.info(f"Processing rate: {len(companies)/elapsed_time:.1f} companies/second")
            logger.info(f"Companies with connections: {unique_companies}")
            logger.info(f"Total connections: {len(results_df)}")
            logger.info(f"Unique individuals: {unique_individuals}")
            logger.info(f"Leadership roles: {leadership_count}")
            logger.info(f"API calls made: {extractor.stats['api_calls_made']}")
            logger.info(f"Cache hits: {extractor.stats['cache_hits']}")
            logger.info(f"Results saved to: {args.output_csv}")
            
        else:
            logger.warning("No leadership connections found")
            empty_df = pd.DataFrame(columns=[
                'company', 'individual', 'role', 'start_date', 'end_date', 
                'is_current', 'is_leadership', 'littlesis_person_id', 
                'littlesis_company_id', 'company_matched_name'
            ])
            empty_df.to_csv(args.output_csv, index=False)
        
        return 0
        
    except Exception as e:
        logger.error(f"Script failed: {e}")
        return 1


if __name__ == "__main__":
    exit(asyncio.run(main()))