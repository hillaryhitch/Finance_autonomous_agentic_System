import os
import requests

# Document URLs
documents = {
    'financial_reports': {
        'Safaricom_Annual_Report_2024.pdf': 'https://www.safaricom.co.ke/annualreport_2024/wp-content/uploads/2024/07/safaricom-annual-report.pdf',
        'Safaricom_Annual_Report_2025.pdf': 'https://www.safaricom.co.ke/images/Downloads/Annual-2025-Reporrt.pdf',
        'FY25_Results_Booklet.pdf': 'https://www.safaricom.co.ke/images/Downloads/FY25-Earnings-Booklet_May-9-2025.pdf'
    },
    'investor_presentations': {
        'FY24_Investor_Presentation.pdf': 'https://www.safaricom.co.ke/images/Downloads/FY24-Investor-Presentation-9th-May-2024.pdf'
    },
    'market_analysis': {
        'H1_FY25_NCBA_Analysis.pdf': 'https://investment-bank.ncbagroup.com/wp-content/uploads/2024/11/Safaricom-1H2025-Update.pdf'
    }
}

# Base directory
base_dir = 'finance_data/safaricom_docs'

# Download function
def download_documents():
    for category, files in documents.items():
        category_path = os.path.join(base_dir, category)
        os.makedirs(category_path, exist_ok=True)
        
        for filename, url in files.items():
            filepath = os.path.join(category_path, filename)
            print(f"Downloading {filename}...")
            
            try:
                response = requests.get(url, timeout=30, verify=False)
                response.raise_for_status()
                
                with open(filepath, 'wb') as f:
                    f.write(response.content)
                print(f"✓ Downloaded: {filepath}")
            except Exception as e:
                print(f"✗ Failed to download {filename}: {e}")

if __name__ == "__main__":
    download_documents()