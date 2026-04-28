"""
Weekly Update Script for Premier League Card Prediction
========================================================

This script automates weekly data collection:
1. Downloads the latest Premier League CSV from football-data.co.uk  
2. Appends new matches to the existing Excel workbook (RAW data)
3. Runs the prediction model (which handles ALL cleaning)
4. Logs all activity

ALL data cleaning happens in 02_match_level_model.py
"""

import pandas as pd
from datetime import datetime
import requests
from io import StringIO
import sys
from pathlib import Path


class WeeklyUpdater:
    """Handles weekly data updates and model execution"""
    
    def __init__(self, base_path):
        self.base_path = Path(base_path)
        self.data_path = self.base_path / 'data' / 'match_level_data'
        self.excel_file = self.data_path / 'match_data_combined_raw.xlsx'
        self.current_season = '2025-2026'
        self.log_file = self.base_path / 'automation_log.txt'
        
        # URL for current season (E0 = Premier League)
        self.data_url = "https://www.football-data.co.uk/mmz4281/2526/E0.csv"
        
    def log(self, message):
        """Write message to console and log file"""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        log_message = f"[{timestamp}] {message}"
        
        print(log_message)
        
        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write(log_message + '\n')
    
    def download_current_season(self):
        """Download the latest CSV from football-data.co.uk"""
        self.log("Downloading current season data...")
        
        try:
            response = requests.get(self.data_url, timeout=30)
            response.raise_for_status()
            
            # Read CSV
            raw_data = pd.read_csv(StringIO(response.text))
            
            # Parse dates (so Excel stores them correctly)
            raw_data['Date'] = pd.to_datetime(raw_data['Date'], format='%d/%m/%Y', errors='coerce')
            
            self.log(f"Downloaded {len(raw_data)} matches")
            return raw_data
            
        except requests.exceptions.RequestException as e:
            self.log(f"ERROR downloading data: {e}")
            sys.exit(1)
    
    def load_existing_data(self):
        """Load existing Excel file"""
        self.log(f"Loading existing data...")
        
        try:
            existing = pd.read_excel(self.excel_file, sheet_name=self.current_season)
            existing['Date'] = pd.to_datetime(existing['Date'])
            
            self.log(f"Loaded {len(existing)} existing matches")
            return existing
            
        except FileNotFoundError:
            self.log(f"ERROR: Excel file not found at {self.excel_file}")
            sys.exit(1)
        except ValueError:
            self.log(f"ERROR: Sheet '{self.current_season}' not found")
            sys.exit(1)
    
    def identify_new_matches(self, downloaded_df, existing_df):
        """Find new matches by comparing Date + HomeTeam + AwayTeam"""
        self.log("Identifying new matches...")
        
        # Create comparison key
        downloaded_df['_key'] = (
            downloaded_df['Date'].dt.strftime('%Y-%m-%d') + '_' +
            downloaded_df['HomeTeam'] + '_' +
            downloaded_df['AwayTeam']
        )
        
        existing_df['_key'] = (
            existing_df['Date'].dt.strftime('%Y-%m-%d') + '_' +
            existing_df['HomeTeam'] + '_' +
            existing_df['AwayTeam']
        )
        
        # Find new matches
        existing_keys = set(existing_df['_key'].values)
        new_matches = downloaded_df[~downloaded_df['_key'].isin(existing_keys)].copy()
        new_matches = new_matches.drop(columns=['_key'])
        
        if len(new_matches) == 0:
            self.log("No new matches found")
            return None
        
        self.log(f"Found {len(new_matches)} new matches:")
        for _, match in new_matches.iterrows():
            self.log(f"  - {match['Date'].strftime('%Y-%m-%d')}: "
                    f"{match['HomeTeam']} vs {match['AwayTeam']}")
        
        return new_matches
    
    def append_to_excel(self, new_matches):
        """Append new matches to Excel"""
        self.log("Appending to Excel...")
        
        try:
            # Load all sheets
            with pd.ExcelFile(self.excel_file) as xls:
                all_sheets = {sheet: pd.read_excel(xls, sheet_name=sheet) 
                             for sheet in xls.sheet_names}
            
            # Append new matches
            all_sheets[self.current_season] = pd.concat(
                [all_sheets[self.current_season], new_matches],
                ignore_index=True
            )
            
            # Sort by date
            all_sheets[self.current_season]['Date'] = pd.to_datetime(
                all_sheets[self.current_season]['Date']
            )
            all_sheets[self.current_season] = (
                all_sheets[self.current_season]
                .sort_values('Date')
                .reset_index(drop=True)
            )
            
            # Write back
            with pd.ExcelWriter(self.excel_file, engine='openpyxl') as writer:
                for sheet_name, df in all_sheets.items():
                    df.to_excel(writer, sheet_name=sheet_name, index=False)
            
            self.log(f"Appended {len(new_matches)} matches")
            
        except Exception as e:
            self.log(f"ERROR writing to Excel: {e}")
            sys.exit(1)
    
    def run_model(self):
        """Run the prediction model"""
        self.log("Running model...")
        
        import subprocess
        
        model_script = self.base_path / '02_match_level_model.py'
        
        if not model_script.exists():
            self.log("ERROR: Model script not found")
            return
        
        try:
            result = subprocess.run(
                ['python', str(model_script)],
                capture_output=True,
                text=True,
                cwd=str(self.base_path),
                timeout=1200  # 20 minutes
            )
            
            if result.returncode == 0:
                self.log("Model completed successfully")
                
                lines = result.stdout.split('\n')
                
                # === OVERALL METRICS ===
                if "MODEL PERFORMANCE" in result.stdout:
                    self.log("")
                    self.log("=" * 60)
                    self.log("OVERALL MODEL PERFORMANCE")
                    self.log("=" * 60)
                    
                    in_section = False
                    for line in lines:
                        if "MODEL PERFORMANCE" in line and "=" in line:
                            in_section = True
                            continue
                        if in_section and line.strip().startswith("=") and len(line.strip()) > 30:
                            break
                        if in_section and line.strip():
                            if any(keyword in line for keyword in 
                                  ["Matches:", "MAE:", "RMSE:", "Bias:", "Log-Loss:"]):
                                self.log(line.strip())
                    
                    self.log("=" * 60)
                
                # === LATEST WEEK METRICS ===
                if "LATEST WEEK PERFORMANCE" in result.stdout:
                    self.log("")
                    self.log("=" * 60)
                    self.log("LATEST WEEK PERFORMANCE")
                    self.log("=" * 60)
                    
                    in_latest = False
                    skip_next_equals = False
                    
                    for i, line in enumerate(lines):
                        # Start capturing
                        if "LATEST WEEK PERFORMANCE" in line and "=" in line:
                            in_latest = True
                            continue
                        
                        # Stop at closing equals (the one after predictions)
                        if in_latest and line.strip() == "=" * 60:
                            if skip_next_equals:
                                # This is the final closing line
                                self.log("=" * 60)
                                break
                            else:
                                # This is after predictions, set flag for next one
                                skip_next_equals = True
                                continue
                        
                        # Log everything in between
                        if in_latest and line.strip():
                            self.log(line.strip())
                
                self.log("")
                
                # Save full output for detailed review
                output_file = self.base_path / 'last_model_run.log'
                with open(output_file, 'w', encoding='utf-8') as f:
                    f.write(result.stdout)
                self.log("Full output saved to: last_model_run.log")
                
            else:
                self.log("Model execution failed")
                
                # Save error for debugging
                error_file = self.base_path / 'model_error.log'
                with open(error_file, 'w', encoding='utf-8') as f:
                    f.write(f"STDOUT:\n{result.stdout}\n\nSTDERR:\n{result.stderr}")
                self.log("Error details saved to: model_error.log")
                    
        except subprocess.TimeoutExpired:
            self.log("Model timed out after 20 minutes")
        except Exception as e:
            self.log(f"ERROR running model: {e}")
    
    def run_update(self):
        """Main workflow"""
        self.log("=" * 60)
        self.log("WEEKLY UPDATE STARTED")
        self.log("=" * 60)
        
        downloaded_data = self.download_current_season()
        existing_data = self.load_existing_data()
        new_matches = self.identify_new_matches(downloaded_data, existing_data)
        
        if new_matches is not None:
            self.append_to_excel(new_matches)
        
        # ALWAYS run model (even if no new data)
        self.run_model()
        
        self.log("=" * 60)
        if new_matches is not None:
            self.log(f"UPDATE COMPLETE - Added {len(new_matches)} matches")
        else:
            self.log("UPDATE COMPLETE - No new data, ran model anyway")
        self.log("=" * 60)


if __name__ == "__main__":
    BASE_PATH = -----
    
    updater = WeeklyUpdater(BASE_PATH)
    updater.run_update()