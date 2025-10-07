#!/usr/bin/env python3
"""
Fossil Fuel Donor Matcher - Simple Web Interface
Matches uploaded donor CSVs against fossil fuel companies and leadership.
"""

from flask import Flask, render_template, request, send_file, jsonify, Response
import pandas as pd
import os
import tempfile
import json
from datetime import datetime
from werkzeug.utils import secure_filename
from fuzzywuzzy import fuzz
import time
import queue
import threading

# Import our matching logic
from name_normalization import normalize_name

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB max
app.config['UPLOAD_FOLDER'] = tempfile.gettempdir()

# Load reference data at startup
FOSSIL_FUEL_COMPANIES = {}
FOSSIL_FUEL_LEADERSHIP = {}

# First letter indexes for fast fuzzy matching
COMPANIES_BY_FIRST_LETTER = {}
LEADERSHIP_BY_FIRST_LETTER = {}

# Caching for match results
MATCH_CACHE = {}

# Progress tracking
PROGRESS_QUEUE = queue.Queue()
CURRENT_PROGRESS = {'processed': 0, 'total': 0, 'matches': 0}

def load_reference_data():
    """Load fossil fuel companies and leadership data."""
    global FOSSIL_FUEL_COMPANIES, FOSSIL_FUEL_LEADERSHIP, COMPANIES_BY_FIRST_LETTER, LEADERSHIP_BY_FIRST_LETTER

    # Load companies
    try:
        companies_df = pd.read_csv('fossil_fuel_data/fossil_fuel_companies_list.csv')
        company_col = 'Company Name' if 'Company Name' in companies_df.columns else companies_df.columns[0]
        category_col = 'Category' if 'Category' in companies_df.columns else (companies_df.columns[1] if len(companies_df.columns) > 1 else None)

        for _, row in companies_df.iterrows():
            company_name = row[company_col]
            if pd.notna(company_name) and str(company_name).strip():
                normalized = normalize_name(str(company_name), is_individual=False)
                if normalized:
                    FOSSIL_FUEL_COMPANIES[normalized] = {
                        'original_name': str(company_name),
                        'category': str(row[category_col]) if category_col and pd.notna(row[category_col]) else 'Fossil Fuel'
                    }
                    # Index by first letter for faster fuzzy matching
                    first_letter = normalized[0] if normalized else ''
                    if first_letter not in COMPANIES_BY_FIRST_LETTER:
                        COMPANIES_BY_FIRST_LETTER[first_letter] = []
                    COMPANIES_BY_FIRST_LETTER[first_letter].append(normalized)
        print(f"✓ Loaded {len(FOSSIL_FUEL_COMPANIES)} fossil fuel companies")
    except Exception as e:
        print(f"✗ Error loading companies: {e}")

    # Load leadership
    try:
        leadership_df = pd.read_csv('fossil_fuel_data/fossil_fuel_leadership.csv')
        for _, row in leadership_df.iterrows():
            individual = row['individual']
            if pd.notna(individual) and str(individual).strip():
                normalized = normalize_name(str(individual), is_individual=True)
                if normalized:
                    if normalized not in FOSSIL_FUEL_LEADERSHIP:
                        FOSSIL_FUEL_LEADERSHIP[normalized] = []
                    FOSSIL_FUEL_LEADERSHIP[normalized].append({
                        'original_name': str(individual),
                        'company': str(row['company']),
                        'role': str(row['role']) if pd.notna(row['role']) else '',
                        'is_leadership': row.get('is_leadership', False)
                    })
                    # Index by first letter for faster fuzzy matching
                    first_letter = normalized[0] if normalized else ''
                    if first_letter not in LEADERSHIP_BY_FIRST_LETTER:
                        LEADERSHIP_BY_FIRST_LETTER[first_letter] = []
                    LEADERSHIP_BY_FIRST_LETTER[first_letter].append(normalized)
        print(f"✓ Loaded {len(FOSSIL_FUEL_LEADERSHIP)} fossil fuel leadership individuals")
    except Exception as e:
        print(f"✗ Error loading leadership: {e}")

def parse_lastname_firstname_name(name_str: str) -> str:
    """Parse 'LastName, FirstName' format to 'FirstName LastName'."""
    if not name_str or pd.isna(name_str):
        return ''

    name_str = str(name_str).strip()
    if ',' in name_str:
        parts = name_str.split(',', 1)
        if len(parts) == 2:
            last_name = parts[0].strip()
            first_part = parts[1].strip()
            first_parts = first_part.split()
            if first_parts:
                first_name = first_parts[0]
                return f"{first_name} {last_name}"

    return name_str

def check_fossil_fuel_match(name: str, employer: str, c_code: str, fuzzy_threshold: int = 95, match_type: str = 'both'):
    """Check if donor matches fossil fuel companies or leadership.

    Args:
        match_type: 'both', 'companies', or 'leadership'
    """
    # Create cache key
    cache_key = f"{name}|{employer}|{c_code}|{fuzzy_threshold}|{match_type}"
    if cache_key in MATCH_CACHE:
        return MATCH_CACHE[cache_key]

    result = {
        'company_match': False,
        'company_name': None,
        'company_category': None,
        'company_score': 0,
        'leadership_match': False,
        'leadership_name': None,
        'leadership_company': None,
        'leadership_role': None,
        'leadership_score': 0
    }

    # Determine if individual
    is_individual = (str(c_code).upper() == 'IND')

    # Normalize names
    if name and pd.notna(name) and str(name).strip():
        if is_individual:
            parsed_name = parse_lastname_firstname_name(str(name))
            normalized_name = normalize_name(parsed_name, is_individual=True)
        else:
            normalized_name = normalize_name(str(name), is_individual=False)
    else:
        normalized_name = ''

    if employer and pd.notna(employer) and str(employer).strip():
        normalized_employer = normalize_name(str(employer), is_individual=False)
    else:
        normalized_employer = ''

    # Check company matches (if enabled)
    if match_type in ('both', 'companies'):
        # Exact match
        if normalized_name in FOSSIL_FUEL_COMPANIES:
            company_info = FOSSIL_FUEL_COMPANIES[normalized_name]
            result['company_match'] = True
            result['company_name'] = company_info['original_name']
            result['company_category'] = company_info['category']
            result['company_score'] = 100
        elif normalized_employer in FOSSIL_FUEL_COMPANIES:
            company_info = FOSSIL_FUEL_COMPANIES[normalized_employer]
            result['company_match'] = True
            result['company_name'] = company_info['original_name']
            result['company_category'] = company_info['category']
            result['company_score'] = 100
        else:
            # Fuzzy match using first-letter index
            best_match = None
            best_score = 0

            # Get companies that start with same letter
            candidates_to_check = set()
            if normalized_name:
                first_letter = normalized_name[0] if normalized_name else ''
                candidates_to_check.update(COMPANIES_BY_FIRST_LETTER.get(first_letter, []))
            if normalized_employer:
                first_letter = normalized_employer[0] if normalized_employer else ''
                candidates_to_check.update(COMPANIES_BY_FIRST_LETTER.get(first_letter, []))

            # Only check companies with matching first letter
            for company_normalized in candidates_to_check:
                company_info = FOSSIL_FUEL_COMPANIES[company_normalized]

                if normalized_name:
                    score = fuzz.ratio(normalized_name, company_normalized)
                    if score >= fuzzy_threshold and score > best_score:
                        best_score = score
                        best_match = company_info
                        # Early exit: if we found a near-perfect match, stop searching
                        if score >= 98:
                            break

                if normalized_employer:
                    score = fuzz.ratio(normalized_employer, company_normalized)
                    if score >= fuzzy_threshold and score > best_score:
                        best_score = score
                        best_match = company_info
                        # Early exit: if we found a near-perfect match, stop searching
                        if score >= 98:
                            break

            if best_match:
                result['company_match'] = True
                result['company_name'] = best_match['original_name']
                result['company_category'] = best_match['category']
                result['company_score'] = best_score

    # Check leadership matches (only for individuals and if enabled)
    if match_type in ('both', 'leadership') and is_individual and normalized_name:
        # Exact match
        if normalized_name in FOSSIL_FUEL_LEADERSHIP:
            leadership_info = FOSSIL_FUEL_LEADERSHIP[normalized_name][0]  # Take first match
            result['leadership_match'] = True
            result['leadership_name'] = leadership_info['original_name']
            result['leadership_company'] = leadership_info['company']
            result['leadership_role'] = leadership_info['role']
            result['leadership_score'] = 100
        else:
            # Fuzzy match using first-letter index
            best_match = None
            best_score = 0

            # Get leadership that start with same letter
            first_letter = normalized_name[0] if normalized_name else ''
            candidates_to_check = LEADERSHIP_BY_FIRST_LETTER.get(first_letter, [])

            # Only check leadership with matching first letter
            for leader_normalized in candidates_to_check:
                leader_records = FOSSIL_FUEL_LEADERSHIP[leader_normalized]
                score = fuzz.ratio(normalized_name, leader_normalized)
                if score >= fuzzy_threshold and score > best_score:
                    best_score = score
                    best_match = leader_records[0]
                    # Early exit: if we found a near-perfect match, stop searching
                    if score >= 98:
                        break

            if best_match:
                result['leadership_match'] = True
                result['leadership_name'] = best_match['original_name']
                result['leadership_company'] = best_match['company']
                result['leadership_role'] = best_match['role']
                result['leadership_score'] = best_score

    # Cache the result
    MATCH_CACHE[cache_key] = result
    return result

def process_donor_files(files, fuzzy_threshold=95, match_type='both'):
    """Process uploaded donor CSV files and return matches."""
    all_matches = []
    stats = {
        'total_donors': 0,
        'company_matches': 0,
        'leadership_matches': 0,
        'any_match': 0,
        'files_processed': 0
    }

    for file in files:
        try:
            df = pd.read_csv(file)
            stats['files_processed'] += 1

            # Verify required columns
            if 'NAME' not in df.columns:
                continue

            for idx, row in df.iterrows():
                name = row.get('NAME', '')
                if not name or pd.isna(name) or not str(name).strip():
                    continue

                stats['total_donors'] += 1

                # Update progress tracking
                CURRENT_PROGRESS['processed'] = stats['total_donors']
                CURRENT_PROGRESS['matches'] = stats['any_match']

                # Progress logging every 100 donors
                if stats['total_donors'] % 100 == 0:
                    print(f"Processed {stats['total_donors']} donors, {stats['any_match']} matches found...")

                employer = row.get('EMPNAME', '')
                c_code = row.get('C_CODE', '')

                # Check for matches
                match_result = check_fossil_fuel_match(name, employer, c_code, fuzzy_threshold, match_type)

                if match_result['company_match'] or match_result['leadership_match']:
                    stats['any_match'] += 1
                    if match_result['company_match']:
                        stats['company_matches'] += 1
                    if match_result['leadership_match']:
                        stats['leadership_matches'] += 1

                    # Create result record
                    result = {
                        'source_file': file.filename,
                        'donor_name': str(name),
                        'donor_employer': str(employer) if pd.notna(employer) else '',
                        'donor_type': str(c_code) if pd.notna(c_code) else '',
                        'amount': str(row.get('AMNT', '')) if 'AMNT' in row else '',
                        'date': str(row.get('DATE', '')) if 'DATE' in row else '',
                        'city': str(row.get('CITY', '')) if 'CITY' in row else '',
                        'state': str(row.get('STATE', '')) if 'STATE' in row else '',
                        'occupation': str(row.get('OCCUPATION', '')) if 'OCCUPATION' in row else '',
                        'company_match': 'YES' if match_result['company_match'] else 'NO',
                        'matched_company': match_result['company_name'] or '',
                        'company_category': match_result['company_category'] or '',
                        'company_match_score': match_result['company_score'],
                        'leadership_match': 'YES' if match_result['leadership_match'] else 'NO',
                        'matched_leadership': match_result['leadership_name'] or '',
                        'leadership_company': match_result['leadership_company'] or '',
                        'leadership_role': match_result['leadership_role'] or '',
                        'leadership_match_score': match_result['leadership_score']
                    }

                    all_matches.append(result)

        except Exception as e:
            print(f"Error processing {file.filename}: {e}")
            continue

    # Print cache statistics
    print(f"Cache size: {len(MATCH_CACHE)} unique donor/employer combinations cached")

    return all_matches, stats

@app.route('/')
def index():
    """Render main page."""
    return render_template('index.html')

@app.route('/process', methods=['POST'])
def process():
    """Process uploaded CSV files."""
    if 'files[]' not in request.files:
        return jsonify({'error': 'No files uploaded'}), 400

    files = request.files.getlist('files[]')

    if len(files) > 20:
        return jsonify({'error': 'Maximum 20 files allowed'}), 400

    fuzzy_threshold = int(request.form.get('fuzzy_threshold', 95))
    match_type = request.form.get('match_type', 'both')

    # Clear cache for new processing run
    MATCH_CACHE.clear()
    CURRENT_PROGRESS['processed'] = 0
    CURRENT_PROGRESS['total'] = 0
    CURRENT_PROGRESS['matches'] = 0
    print(f"Starting processing with match_type={match_type}, threshold={fuzzy_threshold}")

    # Count total donors first
    total_count = 0
    for file in files:
        try:
            df = pd.read_csv(file)
            if 'NAME' in df.columns:
                total_count += len(df[df['NAME'].notna()])
        except:
            pass
    CURRENT_PROGRESS['total'] = total_count
    print(f"Total donors to process: {total_count}")

    # Reset file pointers
    for file in files:
        file.seek(0)

    # Process files
    matches, stats = process_donor_files(files, fuzzy_threshold, match_type)

    if not matches:
        return jsonify({
            'success': True,
            'stats': stats,
            'message': 'No matches found'
        })

    # Create results CSV
    results_df = pd.DataFrame(matches)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_filename = f'fossil_fuel_matches_{timestamp}.csv'
    output_path = os.path.join(app.config['UPLOAD_FOLDER'], output_filename)
    results_df.to_csv(output_path, index=False)

    return jsonify({
        'success': True,
        'stats': stats,
        'download_url': f'/download/{output_filename}',
        'filename': output_filename
    })

@app.route('/download/<filename>')
def download(filename):
    """Download results file."""
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(filename))
    return send_file(filepath, as_attachment=True, download_name=filename)

@app.route('/progress')
def progress():
    """Get current processing progress."""
    return jsonify(CURRENT_PROGRESS)

if __name__ == '__main__':
    print("Loading reference data...")
    load_reference_data()
    print("Starting server...")
    app.run(debug=True, port=5001)
