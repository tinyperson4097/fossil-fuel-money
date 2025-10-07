# 🔥 Fossil Fuel Donor Matcher

A simple web application to cross-check donor CSV files against fossil fuel companies and their leadership.

## Features

- ✅ Upload up to 20 donor CSV files at once
- ✅ Matches donors against fossil fuel companies list
- ✅ Matches donors against fossil fuel leadership database
- ✅ Uses fuzzy matching with configurable threshold
- ✅ Name normalization for accurate matching
- ✅ Download results as CSV
- ✅ Simple, clean web interface

## Quick Start

### 1. Clone the Repository

```bash
git clone <repository-url>
cd fossil-fuel-money
```

### 2. Install Dependencies

**Using Conda (recommended):**

```bash
conda env create -f environment.yml
conda activate fossil-fuel-matcher
```

**Using pip:**

```bash
pip install -r requirements.txt
```

Required packages:
- Flask (web framework)
- pandas (CSV processing)
- fuzzywuzzy (fuzzy string matching)
- python-Levenshtein (faster fuzzy matching)

### 3. Run the Application

```bash
python app.py
```

The application will start on `http://localhost:5001`

### 4. Use the Application

1. Open your browser to `http://localhost:5001`
2. Click or drag-and-drop donor CSV files (up to 20)
3. Choose what to match against:
   - **Both** (default) - Check companies AND leadership (slower)
   - **Companies Only** - Only check fossil fuel companies (faster)
   - **Leadership Only** - Only check leadership individuals (faster)
4. Optionally adjust the fuzzy match threshold (default: 95)
5. Click "Process Files"
6. Download the results CSV with all matches

## Input CSV Format

Your donor CSV files should have columns similar to:
- `NAME` (required) - Donor name
- `EMPNAME` - Employer name (optional but recommended)
- `C_CODE` - Donor type (IND for individual, CORP for corporation, etc.)
- `AMNT` - Donation amount
- `DATE` - Donation date
- `CITY`, `STATE` - Location
- `OCCUPATION` - Donor occupation

Example format (from NYC campaign finance data):
```csv
NAME,C_CODE,EMPNAME,AMNT,DATE,CITY,STATE,OCCUPATION
"Ackman, William A",IND,,,250000.00,4/7/2025,,,
AECOM Technology Corporation,CORP,,,125000.00,6/26/2025,,,
```

## Output CSV Format

The results CSV includes:
- Original donor information (name, employer, amount, date, etc.)
- **Company Match**: YES/NO, matched company name, category, match score
- **Leadership Match**: YES/NO, matched individual, their company, role, match score

## Reference Data

The application uses these reference files:
- `fossil_fuel_data/fossil_fuel_companies_list.csv` - List of fossil fuel companies
- `fossil_fuel_data/fossil_fuel_leadership.csv` - Leadership/staff from fossil fuel companies

Make sure these files are present before running the app.

## Matching Logic

### Name Normalization
- Removes titles (Mr., Mrs., Dr., etc.)
- Standardizes to FIRST LAST format (removes middle names)
- Handles "LastName, FirstName" format
- Removes punctuation from organization names
- Normalizes common abbreviations (PAC, INC, ASSOC, etc.)

### Matching Process
1. **Exact Match**: Normalized names match exactly (100% score)
2. **Fuzzy Match**: Similar names above threshold (default 95%)
3. **Company Matching**: Checks donor name AND employer against company list
4. **Leadership Matching**: Checks individual donors against leadership database

### Performance Optimization
- **Caching**: Results are cached - if the same donor appears multiple times, the match is instant after the first check
- **Selective Matching**: Choose to match only companies or only leadership to speed up processing
- **Progress Logging**: Check your terminal for progress updates every 100 donors

### Fuzzy Match Threshold
- **95-100**: Very strict, fewer false positives
- **90-94**: Balanced
- **80-89**: More lenient, may catch more variations but more false positives

**Speed Tip**: If processing is slow, try:
1. Select "Companies Only" or "Leadership Only" instead of "Both"
2. Lower the fuzzy threshold slightly (e.g., 90) for faster matching
3. Check terminal for progress - if it's processing, it will show every 100 donors

## Project Structure

```
fossil-fuel-money/
├── app.py                          # Main Flask application
├── name_normalization.py           # Name normalization utilities
├── requirements.txt                # Python dependencies
├── templates/
│   └── index.html                  # Web interface
├── fossil_fuel_data/
│   ├── fossil_fuel_companies_list.csv
│   └── fossil_fuel_leadership.csv
└── ny_election_data/               # Sample donor data
    ├── cuomo.csv
    └── fixthecity.csv
```

## Troubleshooting

### Import Error: `name_normalization`
Make sure you run `python app.py` from the project root directory.

### No Matches Found
- Check that your CSV has a `NAME` column
- Try lowering the fuzzy match threshold (e.g., 90 or 85)
- Verify the reference data files are present and populated

### Port Already in Use
Change the port in `app.py`:
```python
app.run(debug=True, port=5001)  # Change 5001 to another port
```

## License

MIT License - feel free to use and modify as needed.
