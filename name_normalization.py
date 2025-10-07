#!/usr/bin/env python3
"""
Virginia Campaign Finance Data Normalization Module

Shared normalization functions for Virginia campaign finance data processing.
This module provides consistent normalization across all processors.
"""

import re
import pandas as pd
import logging

logger = logging.getLogger(__name__)

# Pre-compile regex patterns and constants for performance
TITLE_PATTERNS = [
    # Political titles
    re.compile(r'\bDELEGATE\b'), re.compile(r'\bDEL\.?\b'),
    re.compile(r'\bSENATOR\b'), re.compile(r'\bSEN\.?\b'),
    re.compile(r'\bGOVERNOR\b'), re.compile(r'\bGOV\.?\b'),
    re.compile(r'\bLIEUTENANT GOVERNOR\b'), re.compile(r'\bLT\.? GOV\.?\b'), re.compile(r'\bLIEUT\.? GOV\.?\b'),
    re.compile(r'\bATTORNEY GENERAL\b'), re.compile(r'\bAG\b'), re.compile(r'\bA\.G\.?\b'),
    re.compile(r'\bMAYOR\b'), re.compile(r'\bSHERIFF\b'),
    
    # Personal honorifics
    re.compile(r'\bTHE HONORABLE\b'), re.compile(r'\bHONORABLE\b'), re.compile(r'\bHON\.?\b'),
    re.compile(r'\bMR\.?\b'), re.compile(r'\bMRS\.?\b'), re.compile(r'\bMS\.?\b'), re.compile(r'\bMISS\.?\b'),
    re.compile(r'\bDR\.?\b'), re.compile(r'\bDOCTOR\b'),
    re.compile(r'\bPROF\.?\b'), re.compile(r'\bPROFESSOR\b'),
    re.compile(r'\bREV\.?\b'), re.compile(r'\bREVEREND\b'),
    
    # Military titles
    re.compile(r'\bCAPT\.?\b'), re.compile(r'\bCAPTAIN\b'),
    re.compile(r'\bCOL\.?\b'), re.compile(r'\bCOLONEL\b'),
    re.compile(r'\bMAJ\.?\b'), re.compile(r'\bMAJOR\b'),
    re.compile(r'\bLT\.?\b'), re.compile(r'\bLIEUTENANT\b'),
    re.compile(r'\bGEN\.?\b'), re.compile(r'\bGENERAL\b'),
    
    # Professional titles
    re.compile(r'\bESQ\.?\b'), re.compile(r'\bESQUIRE\b'),
]

# Pre-compile other regex patterns
SPACES_PATTERN = re.compile(r'\s+')
LEADING_NONWORD_PATTERN = re.compile(r'^\W+')
TRAILING_NONWORD_PATTERN = re.compile(r'\W+$')
PAC_PATTERN = re.compile(r'\b(POLITICAL\s+ACTION\s+COMMITTEE)\b')
ASSOCIATION_PATTERN = re.compile(r'\b(ASSOCIATION)\b')
ASSN_PATTERN = re.compile(r'\b(ASSN)\b')
VIRGINIA_PATTERN = re.compile(r'\bVIRGINIA\b')
HIGHWAY_PATTERN = re.compile(r'\bHIGHWAY\b')

# Pre-define exact Dominion matches as a set for O(1) lookup
EXACT_DOMINION_MATCHES = {
    'DOMINION',
    'DOMINION PAC', 
    'DOMINION ENERGY PAC'
    'DOMINION POLITICAL ACTION COMMITTEE',
    'DOMINION POLITICAL ACTION COMMITTEE - VA',
    'DOMINION POLITICAL ACTION COMMITTEE- VA LAST NAME LEFT BLANK',
    'DOMINION EMPLOYEES PAC',
    'DOMINION VA POWER',
    'DOMINION VA. POWER', 
    'DOMINION ENERGY INC. PAC',
    'DOMINION ENERGY INC PAC - VA',
    'DOMINION ENERGY INC. PAC - VA',
    'DOMINION ENERGY INC. PAC - VIRGINIA',
    'DOMINION ENERGY INC PAC - VIRGINIA',
    'DOMINION PAC - VA',
    'DOMINION POWER',
    'DOMINION PAC VA',
    'DOMINION POLITICAL ACTION COMMITTE - VA',
    'DOMINION PAC-VA',
    'DOMINION POWER PAC',
    'DOMINION POLITICAL ACTION COMITTEE',
    'DOMINION PAC OF VA',
    'DOMINION PAC-VA',
    'DOMINION POLITICAL ACCTION COMMITEE',
    'DOMINION PAC VA',
    'DOMINION-PAC-VA',
    'DOMINION ENERGY, INC.',
    'DOMINION ENERGY INC.',
    'DOMINION ENERGY INC',
    'DOMINION RESOURCES INC. PAC - VA',
    'DOMINION RESOURCES',
    'DOMINIONENERGY',  # no space
    'DOMINION ENGERGY',  # misspelling
}


def normalize_name(name: str, is_individual: bool = None) -> str:
    """Enhanced name normalization with title/honorific removal and middle name standardization."""
    if not name:
        return ''
    
    # Convert to uppercase, remove extra spaces, and basic cleanup
    normalized = str(name).upper().strip()
    normalized = SPACES_PATTERN.sub(' ', normalized)  # Replace multiple spaces with single space
    
    # For individuals only: Remove titles and normalize to first/last name
    if is_individual:
        # Remove titles (but keep family suffixes) using pre-compiled patterns
        for pattern in TITLE_PATTERNS:
            normalized = pattern.sub('', normalized)
        
        # Clean up punctuation and extra spaces using pre-compiled patterns
        normalized = LEADING_NONWORD_PATTERN.sub('', normalized)
        normalized = TRAILING_NONWORD_PATTERN.sub('', normalized)
        normalized = SPACES_PATTERN.sub(' ', normalized).strip()
        
        # Normalize to first and last name only (remove middle names/initials for matching)
        normalized = extract_first_last_name(normalized)
    
    # For all entities (individuals and companies): Apply general normalizations using pre-compiled patterns
    normalized = PAC_PATTERN.sub('PAC', normalized)
    normalized = ASSOCIATION_PATTERN.sub('ASSOC', normalized)
    normalized = ASSN_PATTERN.sub('ASSOC', normalized)
    normalized = VIRGINIA_PATTERN.sub('VA', normalized)
    normalized = HIGHWAY_PATTERN.sub('HWY', normalized)
    
    # For non-individuals: strip all punctuation and remove ending words PAC/INC
    if not is_individual:
        # Strip all punctuation
        normalized = re.sub(r'[^\w\s]', '', normalized)
        
        # Remove ending words PAC and INC
        normalized = re.sub(r'\bPAC$', '', normalized).strip()
        normalized = re.sub(r'\bINC$', '', normalized).strip()
    
    # Check for exact Dominion matches using pre-defined set (O(1) lookup)
    if normalized.strip() in EXACT_DOMINION_MATCHES:
        normalized = 'DOMINION ENERGY'
    elif normalized.startswith('DOMINION ENERGY '):
        # Clean up existing "DOMINION ENERGY" variations that have extra words
        normalized = 'DOMINION ENERGY'
    
    # Final cleanup
    normalized = SPACES_PATTERN.sub(' ', normalized).strip()
    
    return normalized


def extract_first_last_name(name: str) -> str:
    """Extract first and last name, removing middle names/initials for consistent matching."""
    if not name:
        return ''
    
    # Normalize hyphens - remove them to handle hyphenated vs non-hyphenated variations
    # This makes 'MICHELLE-ANN' = 'MICHELLE ANN' and 'LOPES-MALDONADO' = 'LOPES MALDONADO'
    normalized_name = name.replace('-', ' ')
    
    # Split into parts
    parts = normalized_name.split()
    if len(parts) < 2:
        return name  # Return as-is if less than 2 parts
    
    # Identify suffixes (JR, SR, III, IV, V)
    suffixes = ['JR', 'SR', 'III', 'IV', 'V', 'JUNIOR', 'SENIOR']
    suffix_parts = []
    name_parts = []
    
    # Separate suffixes from name parts
    for part in parts:
        if part in suffixes:
            suffix_parts.append(part)
        else:
            name_parts.append(part)
    
    if len(name_parts) < 2:
        return name  # Return original if we can't identify first/last
    
    # Extract first and last name (ignore middle parts)
    first_name = name_parts[0]
    last_name = name_parts[-1]
    
    # Log potential nickname matches for manual review
    #log_potential_nickname_matches(first_name, last_name)
    
    # Rebuild: first + last + suffixes
    result_parts = [first_name, last_name] + suffix_parts
    return ' '.join(result_parts)


def log_potential_nickname_matches(first_name: str, last_name: str):
    """Log potential nickname/name variations for manual review."""
    # Common nickname patterns that might need manual review
    potential_nicknames = {
        'PATRICK': ['PAT'], 'PAT': ['PATRICK'],
        'DANIEL': ['DAN'], 'DAN': ['DANIEL'], 
        'MICHAEL': ['MIKE'], 'MIKE': ['MICHAEL'],
        'ROBERT': ['BOB', 'ROB'], 'BOB': ['ROBERT'], 'ROB': ['ROBERT'],
        'WILLIAM': ['BILL', 'WILL'], 'BILL': ['WILLIAM'], 'WILL': ['WILLIAM'],
        'RICHARD': ['RICK', 'DICK'], 'RICK': ['RICHARD'], 'DICK': ['RICHARD'],
        'ELIZABETH': ['LIZ', 'BETH'], 'LIZ': ['ELIZABETH'], 'BETH': ['ELIZABETH'],
        'CHRISTOPHER': ['CHRIS'], 'CHRIS': ['CHRISTOPHER'],
        'MATTHEW': ['MATT'], 'MATT': ['MATTHEW'],
        'ANTHONY': ['TONY'], 'TONY': ['ANTHONY'],
        'JOSEPH': ['JOE'], 'JOE': ['JOSEPH'],
        'JAMES': ['JIM'], 'JIM': ['JAMES']
    }
    
    # Common surname variations that might need manual review  
    potential_surname_variations = {
        'LOPEZ': ['LOPES'], 'LOPES': ['LOPEZ'],
        'JOHNSON': ['JOHNSTON'], 'JOHNSTON': ['JOHNSON'],
        'GARCIA': ['GARCIA'], # Placeholder for accent variations
        'RODRIGUEZ': ['RODRIQUEZ'], 'RODRIQUEZ': ['RODRIGUEZ']
    }
    
    # Check if this name has potential nickname matches
    if first_name in potential_nicknames:
        logger.info(f"POTENTIAL_NICKNAME_MATCH: '{first_name} {last_name}' - could match: {[f'{alt} {last_name}' for alt in potential_nicknames[first_name]]}")
    
    # Check if this surname has potential variations
    if last_name in potential_surname_variations:
        logger.info(f"POTENTIAL_SURNAME_VARIATION: '{first_name} {last_name}' - could match: {[f'{first_name} {alt}' for alt in potential_surname_variations[last_name]]}")


def normalize_office_sought(office_sought: str) -> str:
    """Normalize office_sought to standard categories."""
    if pd.isna(office_sought):
        return None
    
    office = str(office_sought).lower().strip()
    
    # Remove district names from office_sought_normal
    # Extract base office by removing district-specific parts
    office_clean = re.sub(r'\s*-\s*.*$', '', office)  # Remove everything after dash
    office_clean = re.sub(r'\b(prince william county|blue ridge district|arlington county|at large)\b', '', office_clean).strip()
    office_clean = re.sub(r'\s+', ' ', office_clean)  # Clean up multiple spaces
    
    # Handle abbreviations and specific mappings first
    if office_clean in ['hod', 'h.o.d.']:
        return 'delegate'
    elif office_clean in ['ag', 'a.g.']:
        return 'attorney general'
    elif office_clean in ['gov', 'governor']:
        return 'governor'
    elif any(abbrev in office_clean for abbrev in ['lt gov', 'lt. gov', 'lieutenant gov', 'lieut gov', 'lieu gov']):
        return 'lieutenant governor'
    elif 'delegate' in office_clean or 'hod' in office_clean:
        return 'delegate'
    elif 'senator' in office_clean or 'senate' in office_clean:
        return 'senator'
    elif 'governor' in office_clean and 'lieutenant' not in office_clean and 'lt' not in office_clean:
        return 'governor'
    elif any(term in office_clean for term in ['lieutenant', 'lt']) and 'governor' in office_clean:
        return 'lieutenant governor'
    elif ('attorney' in office_clean and 'general' in office_clean) or office_clean in ['ag', 'a.g.']:
        return 'attorney general'
    elif 'treasurer' in office_clean:
        return 'treasurer'
    elif 'secretary' in office_clean and 'commonwealth' in office_clean:
        return 'secretary of the commonwealth'
    elif ('member' in office_clean and 'county board' in office_clean) or ('supervisor' in office_clean or 'county board' in office_clean) and ('chair' in office_clean or 'chairman' in office_clean):
        return 'chair board of supervisors'
    elif ('member' in office_clean and 'board' in office_clean) or 'supervisor' in office_clean or 'county board' in office_clean:
        return 'member board of supervisors'
    elif 'school' in office_clean and 'board' in office_clean and ('chair' in office_clean or 'chairman' in office_clean):
        return 'chair school board'
    elif 'school' in office_clean and 'board' in office_clean:
        return 'school board'
    elif 'city council' in office_clean or 'town council' in office_clean:
        return 'city council'
    elif 'mayor' in office_clean:
        return 'mayor'
    elif 'sheriff' in office_clean:
        return 'sheriff'
    elif 'clerk' in office_clean and 'court' in office_clean:
        return 'clerk of court'
    elif 'commonwealth' in office_clean and 'attorney' in office_clean:
        return 'commonwealth attorney'
    else:
        return office_clean


def determine_government_level(office_sought_normal: str, district: str) -> str:
    """Determine the level of government based on office and district."""
    district_str = str(district).lower().strip() if district and pd.notna(district) else ''
    
    # Federal level
    if 'congressional' in district_str:
        return 'federal'
    
    # State level offices
    if office_sought_normal:
        state_offices = {
            'delegate', 'senator', 'governor', 'lieutenant governor', 
            'attorney general', 'treasurer', 'secretary of the commonwealth'
        }
        
        if office_sought_normal in state_offices:
            return 'state'
    
    # Everything else is local
    return 'local'


def normalize_district(district: str, candidate_city: str = None, level: str = None, office_sought: str = None) -> str:
    """Extract numerical part of district with no leading zeros."""
    # Get normalized office for special handling
    office_sought_normal = normalize_office_sought(office_sought) if office_sought else None
    
    # Check if office_sought contains "at large" or similar variations
    at_large = False
    if office_sought and not pd.isna(office_sought):
        office_lower = str(office_sought).lower()
        if any(term in office_lower for term in ['at large', 'at-large', 'atlarge', ' al ', ' al,', ' al.', 'at large']):
            at_large = True
    
    # Check if district contains at-large variations
    if district and not pd.isna(district):
        district_lower = str(district).lower().strip()
        if any(term in district_lower for term in ['at large', 'at-large', 'atlarge', ' al ', ' al,', ' al.', 'at large']):
            at_large = True
    
    suffix = (' - ' + office_sought.split('-', 1)[1].strip()) if office_sought and '-' in office_sought else ''
    
    # Special handling for mayors - always district 0
    if office_sought_normal == 'mayor':
        if level == 'local' and candidate_city and not pd.isna(candidate_city):
            return f"{candidate_city.strip()} (0)".title()
        return "0"
    
    # Special handling for at-large positions - always district 0
    if at_large:
        if level == 'local' and candidate_city and not pd.isna(candidate_city):
            return f"{candidate_city.strip()} (0)".title()
        return "0"
    
    # Handle empty/null district
    if pd.isna(district) or str(district).strip() == '':
        if level == 'local' and candidate_city and not pd.isna(candidate_city):
            # For LOCAL entries with blank district: "City Name (0)"
            return f"{candidate_city.strip()} (0)".title()
        elif candidate_city and not pd.isna(candidate_city):
            return candidate_city.strip().title()
        return None
    
    district_str = str(district).strip()
    
    # Extract numbers from the district string
    numbers = re.findall(r'\d+', district_str)
    
    if numbers:
        # Take the first number found and remove leading zeros
        district_normal = str(int(numbers[0]))
        # For LOCAL entries: put city name before district
        if level == 'local' and candidate_city and not pd.isna(candidate_city):
            district_normal = f"{candidate_city.strip()} ({district_normal})"
            district_normal += suffix
        return district_normal.title()
    
    # For entries with no numbers/letters: use 0
    if level == 'local' and candidate_city and not pd.isna(candidate_city):
        # Check if district has any letters or numbers
        if not re.search(r'[a-zA-Z0-9]', district_str):
            return f"{candidate_city.strip()} (0)".title()
        else:
            return f"{candidate_city.strip()} ({district_str})".title()
    
    return district_str.title() if district_str else None


def determine_primary_or_general(election_cycle: str) -> str:
    """Determine if election is primary or general based on election cycle."""
    if pd.isna(election_cycle):
        return None
    
    election_str = str(election_cycle).strip()
    if election_str.startswith('11/'):
        return 'general'
    else:
        return 'primary'