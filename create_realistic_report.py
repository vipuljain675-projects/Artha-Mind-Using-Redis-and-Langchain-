"""
Creates a comprehensive, realistic financial annual report PDF
for a fictional Indian conglomerate - modelled on real-world reporting style.
"""
from fpdf import FPDF

class FinancialReport(FPDF):
    def __init__(self, company, period):
        super().__init__()
        self.company = company
        self.period = period

    def header(self):
        self.set_fill_color(15, 23, 42)
        self.rect(0, 0, 210, 20, 'F')
        self.set_font("Helvetica", "B", 11)
        self.set_text_color(255, 255, 255)
        self.cell(0, 20, f"  {self.company}  |  {self.period} Annual Report  |  CONFIDENTIAL", align="L")
        self.set_text_color(0, 0, 0)
        self.ln(8)

    def footer(self):
        self.set_y(-13)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(120, 120, 120)
        self.cell(0, 10, f"Page {self.page_no()} | IndiaCorp Industries Ltd. | For Internal Use Only", align="C")

    def section_title(self, title):
        self.ln(4)
        self.set_font("Helvetica", "B", 13)
        self.set_fill_color(240, 248, 255)
        self.set_text_color(10, 60, 120)
        self.cell(0, 9, f"  {title}", fill=True, new_x="LMARGIN", new_y="NEXT")
        self.set_text_color(0, 0, 0)
        self.ln(2)

    def sub_title(self, title):
        self.set_font("Helvetica", "B", 10)
        self.set_text_color(30, 100, 60)
        self.cell(0, 7, title, new_x="LMARGIN", new_y="NEXT")
        self.set_text_color(0, 0, 0)

    def body(self, text):
        self.set_font("Helvetica", size=9.5)
        self.multi_cell(0, 5.5, text)
        self.ln(2)

    def table_row(self, cols, bold=False, highlight=False):
        if highlight:
            self.set_fill_color(230, 244, 255)
        else:
            self.set_fill_color(255, 255, 255)
        style = "B" if bold else ""
        self.set_font("Helvetica", style, 9)
        widths = [90, 35, 35, 30]
        row_data = cols + [""] * (4 - len(cols))
        for i, (w, txt) in enumerate(zip(widths, row_data)):
            align = "L" if i == 0 else "R"
            self.cell(w, 6, txt, border=1, align=align, fill=highlight or bold)
        self.ln()


# ─── Build PDF ───────────────────────────────────────────────────────────────
pdf = FinancialReport("IndiaCorp Industries Ltd.", "FY 2023-24")
pdf.set_auto_page_break(auto=True, margin=15)
pdf.set_margins(12, 22, 12)

# ── PAGE 1: Cover + Executive Summary ──────────────────────────────────────
pdf.add_page()

# Cover Block
pdf.ln(5)
pdf.set_font("Helvetica", "B", 22)
pdf.set_text_color(10, 60, 120)
pdf.cell(0, 12, "IndiaCorp Industries Ltd.", new_x="LMARGIN", new_y="NEXT", align="C")
pdf.set_font("Helvetica", "B", 14)
pdf.set_text_color(40, 40, 40)
pdf.cell(0, 8, "Annual Report & Financial Statements - FY 2023-24", new_x="LMARGIN", new_y="NEXT", align="C")
pdf.set_font("Helvetica", size=10)
pdf.set_text_color(100, 100, 100)
pdf.cell(0, 7, "BSE: 500325  |  NSE: INDCORP  |  ISIN: INE002A01018", new_x="LMARGIN", new_y="NEXT", align="C")
pdf.set_text_color(0, 0, 0)
pdf.ln(6)

pdf.section_title("1. LETTER FROM THE CHAIRMAN & MANAGING DIRECTOR")
pdf.body("""Dear Shareholders,

It is with great pride that I present IndiaCorp Industries' Annual Report for the financial year ended March 31, 2024. This has been a landmark year for our organisation - one defined by record-breaking revenues, strategic global expansions, and a firm commitment to sustainable business practices.

FY 2023-24 saw IndiaCorp cross the historic milestone of Rs.10,00,000 Crore (Rs.10 Trillion) in consolidated gross revenue for the first time in our 48-year history. This achievement reflects the dedication of our 2.7 lakh employees, the trust of our 47 crore customers, and the strategic vision of our Board.

Our core businesses - Energy, Petrochemicals, Retail, and Digital Services - all recorded their highest-ever revenues and profits. The Digital Services segment, under the JioConnect brand, crossed 500 million subscribers and generated an EBITDA of Rs.1,45,780 Crore, establishing itself as one of Asia's most profitable telecom-tech businesses.

We have allocated Rs.2,00,000 Crore towards capital expenditure over the next three years, a testament to our conviction in India's economic trajectory and our ambition to lead in green energy, semiconductors, and next-generation retail.

I thank you for your continued faith in IndiaCorp's journey.

Warm regards,
Mukesh N. Ambani
Chairman & Managing Director""")

pdf.section_title("2. COMPANY OVERVIEW")
pdf.body("""IndiaCorp Industries Limited is India's largest private sector company by revenue and market capitalisation. Incorporated in 1973, the company operates across six major business verticals:

  1. Oil-to-Chemicals (O2C): Refining, petrochemicals, fuel retail
  2. Oil & Gas Exploration: Domestic E&P, including the KG-D6 deepwater block
  3. Retail (IndiaStore): India's largest retail chain with 19,500+ stores across Tier 1-4 cities
  4. Digital Services (JioConnect): 500M+ subscriber telecom, OTT, fintech ecosystem
  5. Media & Entertainment (IndiaBroadcast): 70+ TV channels, streaming platform (JioVis)
  6. New Energy: Solar giga-factories, green hydrogen, battery storage

Registered Office: IndiaCorp House, Maker Chambers IV, 222 Nariman Point, Mumbai - 400021
CIN: L17110MH1973PLC019786
Statutory Auditors: Price Waterhouse Coopers (PwC) LLP
Credit Rating: AAA/Stable (CRISIL), AAA/Stable (ICRA)""")

# ── PAGE 2: Financial Highlights + Consolidated P&L ───────────────────────
pdf.add_page()
pdf.section_title("3. FINANCIAL PERFORMANCE AT A GLANCE - FY 2023-24")

pdf.sub_title("Key Consolidated Financial Metrics (Rs. Crore unless stated):")
pdf.ln(2)

# Tables
headers = ["Metric", "FY 2023-24", "FY 2022-23", "YoY Change"]
pdf.table_row(headers, bold=True)
rows = [
    ("Gross Revenue", "10,09,855", "8,76,020", "+15.3%"),
    ("Net Revenue (ex-excise)", "9,01,774", "7,82,499", "+15.2%"),
    ("EBITDA", "2,01,283", "1,58,600", "+26.9%"),
    ("EBITDA Margin (%)", "22.3%", "20.3%", "+200 bps"),
    ("Profit Before Tax (PBT)", "1,25,846", "98,230", "+28.1%"),
    ("Net Profit (PAT)", "79,020", "66,702", "+18.5%"),
    ("Net Profit Margin (%)", "8.8%", "8.5%", "+30 bps"),
    ("Earnings Per Share (EPS)", "Rs.117.4", "Rs.99.1", "+18.5%"),
    ("Dividend Per Share (DPS)", "Rs.10.0", "Rs.9.0", "+11.1%"),
    ("Return on Equity (ROE)", "11.2%", "10.4%", "+80 bps"),
    ("Return on Capital Employed (ROCE)", "9.8%", "8.7%", "+110 bps"),
]
for i, r in enumerate(rows):
    pdf.table_row(list(r), highlight=(i % 2 == 0))

pdf.ln(4)
pdf.section_title("4. CONSOLIDATED PROFIT & LOSS STATEMENT")
pdf.sub_title("For the Year Ended March 31, 2024 (Rs. Crore)")
pdf.ln(2)

pdf.table_row(["Particulars", "FY 2023-24", "FY 2022-23", "Change (%)"], bold=True)
pl_rows = [
    ("I. Revenue from Operations", "9,01,774", "7,82,499", "+15.2%"),
    ("   a) Products", "7,20,540", "6,30,120", "+14.3%"),
    ("   b) Services", "1,81,234", "1,52,379", "+18.9%"),
    ("II. Other Income", "22,580", "18,230", "+23.9%"),
    ("III. Total Income (I + II)", "9,24,354", "8,00,729", "+15.4%"),
    ("IV. Expenses", "", "", ""),
    ("   Cost of Materials", "5,10,230", "4,60,100", "+10.9%"),
    ("   Purchase of Stock-in-Trade", "98,450", "85,640", "+15.0%"),
    ("   Employee Benefits Expense", "52,110", "44,300", "+17.6%"),
    ("   Finance Costs", "18,920", "16,400", "+15.4%"),
    ("   Depreciation & Amortisation", "56,517", "43,770", "+29.1%"),
    ("   Other Expenses", "62,281", "51,289", "+21.4%"),
    ("V. PROFIT BEFORE TAX (III - IV)", "1,25,846", "98,230", "+28.1%"),
    ("VI. Tax Expense", "26,826", "21,528", "+24.6%"),
    ("   Current Tax", "19,430", "15,200", "+27.8%"),
    ("   Deferred Tax", "7,396", "6,328", "+16.9%"),
    ("VII. NET PROFIT AFTER TAX", "79,020", "66,702", "+18.5%"),
    ("VIII. Other Comprehensive Income (OCI)", "3,240", "-1,820", "N/M"),
    ("IX. TOTAL COMPREHENSIVE INCOME", "82,260", "64,882", "+26.8%"),
    ("   Attributable to Equity Shareholders", "78,340", "64,100", "+22.2%"),
    ("   Attributable to Non-Controlling Interest", "3,920", "782", "+401.3%"),
]
for i, r in enumerate(pl_rows):
    bold = "NET PROFIT" in r[0] or "TOTAL" in r[0] or "PROFIT BEFORE" in r[0]
    pdf.table_row(list(r), bold=bold, highlight=(not bold and i % 2 == 0))

# ── PAGE 3: Balance Sheet ──────────────────────────────────────────────────
pdf.add_page()
pdf.section_title("5. CONSOLIDATED BALANCE SHEET")
pdf.sub_title("As at March 31, 2024 (Rs. Crore)")
pdf.ln(2)

pdf.table_row(["Particulars", "FY 2023-24", "FY 2022-23", "Change (%)"], bold=True)
bs_rows = [
    ("EQUITY & LIABILITIES", "", "", ""),
    ("Share Capital", "6,766", "6,765", "+0.0%"),
    ("Other Equity (Reserves & Surplus)", "7,46,929", "6,70,434", "+11.4%"),
    ("Total Equity Attributable to Shareholders", "7,53,695", "6,77,199", "+11.3%"),
    ("Non-Controlling Interests (NCI)", "1,48,922", "62,100", "+139.8%"),
    ("TOTAL EQUITY", "9,02,617", "7,39,299", "+22.1%"),
    ("Non-Current Liabilities", "", "", ""),
    ("Long-Term Borrowings", "2,41,690", "2,17,840", "+10.9%"),
    ("Deferred Tax Liabilities (Net)", "49,230", "42,800", "+15.0%"),
    ("Other Non-Current Liabilities", "18,442", "15,600", "+18.2%"),
    ("Current Liabilities", "", "", ""),
    ("Short-Term Borrowings", "25,490", "42,130", "-39.5%"),
    ("Trade Payables", "1,89,340", "1,52,300", "+24.3%"),
    ("Other Current Liabilities", "98,720", "87,560", "+12.7%"),
    ("TOTAL LIABILITIES", "6,22,912", "5,58,230", "+11.6%"),
    ("TOTAL EQUITY + LIABILITIES", "15,25,529", "12,97,529", "+17.6%"),
    ("ASSETS", "", "", ""),
    ("Non-Current Assets", "", "", ""),
    ("Property, Plant & Equipment (Net)", "6,42,180", "5,28,900", "+21.4%"),
    ("Capital Work-in-Progress (CWIP)", "1,78,340", "1,42,590", "+25.1%"),
    ("Goodwill & Intangible Assets", "98,440", "85,200", "+15.5%"),
    ("Long-Term Investments", "1,12,830", "89,420", "+26.2%"),
    ("Current Assets", "", "", ""),
    ("Inventories", "1,89,320", "1,54,100", "+22.9%"),
    ("Trade Receivables", "98,340", "82,660", "+19.0%"),
    ("Cash & Cash Equivalents", "1,43,650", "1,01,780", "+41.1%"),
    ("Other Current Assets", "62,429", "12,879", "+384.7%"),
    ("TOTAL ASSETS", "15,25,529", "12,97,529", "+17.6%"),
]
for i, r in enumerate(bs_rows):
    bold = r[0] in ("TOTAL EQUITY", "TOTAL ASSETS", "TOTAL LIABILITIES", "TOTAL EQUITY + LIABILITIES", "EQUITY & LIABILITIES", "ASSETS")
    pdf.table_row(list(r), bold=bold, highlight=(not bold and i % 2 == 0))

# ── PAGE 4: Segment-wise Performance ──────────────────────────────────────
pdf.add_page()
pdf.section_title("6. SEGMENT-WISE FINANCIAL PERFORMANCE")
pdf.sub_title("Revenue by Business Segment (Rs. Crore)")
pdf.ln(2)

pdf.table_row(["Business Segment", "FY 2023-24", "FY 2022-23", "YoY Growth"], bold=True)
seg_rows = [
    ("Oil-to-Chemicals (O2C)", "6,11,420", "5,58,200", "+9.5%"),
    ("Oil & Gas Exploration (E&P)", "18,340", "14,220", "+28.9%"),
    ("Retail (IndiaStore)", "3,06,805", "2,60,364", "+17.8%"),
    ("Digital Services (JioConnect)", "1,43,780", "1,18,500", "+21.3%"),
    ("Media & Entertainment", "22,040", "18,100", "+21.8%"),
    ("New Energy", "4,980", "820", "+507.3%"),
    ("Inter-Segment Eliminations", "-1,05,591", "-87,705", "N/M"),
    ("CONSOLIDATED REVENUE", "9,01,774", "7,82,499", "+15.2%"),
]
for i, r in enumerate(seg_rows):
    bold = r[0] == "CONSOLIDATED REVENUE"
    pdf.table_row(list(r), bold=bold, highlight=(not bold and i % 2 == 0))

pdf.ln(4)
pdf.sub_title("EBITDA by Business Segment (Rs. Crore)")
pdf.ln(2)

pdf.table_row(["Business Segment", "EBITDA FY24", "EBITDA FY23", "EBITDA Margin FY24"], bold=True)
ebitda_rows = [
    ("Oil-to-Chemicals (O2C)", "79,820", "61,900", "13.1%"),
    ("Oil & Gas Exploration (E&P)", "13,920", "10,540", "75.9%"),
    ("Retail (IndiaStore)", "22,840", "17,900", "7.4%"),
    ("Digital Services (JioConnect)", "1,45,780", "1,09,600", "101.4%*"),
    ("Media & Entertainment", "4,120", "3,100", "18.7%"),
    ("New Energy", "-4,200", "-1,210", "Negative"),
    ("Corporate & Unallocated", "-60,997", "-43,230", "N/M"),
    ("CONSOLIDATED EBITDA", "2,01,283", "1,58,600", "22.3%"),
]
for i, r in enumerate(ebitda_rows):
    bold = r[0] == "CONSOLIDATED EBITDA"
    pdf.table_row(list(r), bold=bold, highlight=(not bold and i % 2 == 0))

pdf.ln(2)
pdf.body("* JioConnect EBITDA margin exceeds 100% relative to reported segment revenue due to inter-company eliminations and capitalised tower infrastructure income reclassification per Ind AS 115.")

pdf.section_title("7. CASH FLOW ANALYSIS")
pdf.sub_title("Consolidated Cash Flow Statement Summary (Rs. Crore)")
pdf.ln(2)

pdf.table_row(["Cash Flow Category", "FY 2023-24", "FY 2022-23", "Change"], bold=True)
cf_rows = [
    ("Cash from Operating Activities", "1,74,280", "1,41,580", "+23.1%"),
    ("Cash used in Investing Activities", "-2,09,450", "-1,86,320", "+12.4%"),
    ("Cash from / (used in) Financing", "77,040", "28,650", "+169.0%"),
    ("Net Change in Cash & Equivalents", "41,870", "-16,090", "N/M"),
    ("Opening Cash Balance", "1,01,780", "1,17,870", ""),
    ("Closing Cash Balance", "1,43,650", "1,01,780", "+41.1%"),
    ("Free Cash Flow (OCF - Capex)", "-35,170", "-44,740", "Improvement"),
]
for i, r in enumerate(cf_rows):
    bold = r[0] in ("Free Cash Flow (OCF - Capex)",)
    pdf.table_row(list(r), bold=bold, highlight=(not bold and i % 2 == 0))

# ── PAGE 5: Debt + Ratios + ESG ───────────────────────────────────────────
pdf.add_page()
pdf.section_title("8. DEBT PROFILE & CREDIT METRICS")
pdf.body("""IndiaCorp maintains one of the strongest balance sheets in the Indian corporate sector. Total gross debt as on March 31, 2024 stands at Rs.2,67,180 Crore - an increase of Rs.7,210 Crore from FY23, primarily driven by capital raising for the New Energy Giga-complex in Jamnagar.

Net Debt (Gross Debt minus Cash & Liquid Investments): Rs.1,07,490 Crore (vs. Rs.1,44,820 Crore in FY23)
Net Debt / EBITDA: 0.53x (FY23: 0.91x) - a dramatic deleveraging
Net Debt / Equity: 0.12x (FY23: 0.22x)
Interest Coverage Ratio: 10.6x (EBITDA / Finance Costs)

The company successfully raised Rs.90,000 Crore via a Rights Issue at Rs.2,185 per share in Q2 FY24, fully subscribed at 1.3x. An additional USD 5 Billion (approx. Rs.41,600 Crore) was raised through Green Bonds listed on London Stock Exchange and Singapore Exchange to fund the New Energy vertical.

CRISIL upgraded IndiaCorp's long-term outlook from 'Stable' to 'Positive' in November 2023, citing improved free cash flow visibility from the Retail and Digital verticals.""")

pdf.section_title("9. KEY FINANCIAL RATIOS")
pdf.ln(2)

pdf.table_row(["Ratio", "FY 2023-24", "FY 2022-23", "Industry Avg."], bold=True)
ratio_rows = [
    ("P/E Ratio (on EPS of Rs.117.4)", "26.8x", "24.1x", "22.0x"),
    ("Price-to-Book (P/B)", "2.8x", "2.5x", "2.2x"),
    ("EV/EBITDA", "17.2x", "16.4x", "14.8x"),
    ("Current Ratio", "1.34", "1.12", "1.20"),
    ("Quick Ratio", "0.98", "0.84", "0.95"),
    ("Debt-to-Equity", "0.30", "0.35", "0.45"),
    ("Gross Profit Margin", "43.2%", "41.1%", "38.5%"),
    ("Operating Profit Margin (EBIT/Rev)", "18.7%", "17.3%", "15.2%"),
    ("Net Profit Margin", "8.8%", "8.5%", "7.1%"),
    ("Return on Assets (ROA)", "5.2%", "5.1%", "4.4%"),
    ("Return on Equity (ROE)", "11.2%", "10.4%", "9.8%"),
    ("Asset Turnover Ratio", "0.59", "0.60", "0.62"),
    ("Inventory Turnover", "4.76x", "5.10x", "4.90x"),
    ("Days Payable Outstanding (DPO)", "51 days", "49 days", "45 days"),
]
for i, r in enumerate(ratio_rows):
    pdf.table_row(list(r), highlight=(i % 2 == 0))

pdf.ln(4)
pdf.section_title("10. ESG & SUSTAINABILITY PERFORMANCE")
pdf.body("""IndiaCorp placed sustainability at the core of its FY24 strategy:

ENVIRONMENT:
- Carbon Intensity: Reduced by 17.3% per unit of GVA vs. FY19 baseline (target: 30% by FY30)
- Renewable Energy: 4.2 GW installed solar capacity (FY23: 2.8 GW); targeting 100 GW by FY30
- Water Recycling: 92.4% of total water withdrawal recycled across manufacturing sites
- Zero Liquid Discharge: Achieved at all 14 petrochemical manufacturing facilities
- Green Hydrogen: Pilot production of 1,200 tonnes in FY24; commercial plant under commissioning

SOCIAL:
- Total Workforce: 2,71,480 employees (14.3% female representation, up from 12.1%)
- CSR Spend: Rs.1,582 Crore (Section 135 obligation: Rs.932 Crore - exceeded by 70%)
- Rural Education: Jio Digital Classrooms deployed in 1,22,000 schools across 28 states
- Healthcare: IndiaCorp Foundation operated 7 district hospitals and 320 health centres

GOVERNANCE:
- Independent Directors: 9 of 14 Board members are independent (64%)
- Audit Committee: 100% independent; meets quarterly
- Whistleblower Cases FY24: 148 raised, 144 resolved, 4 under investigation
- SEBI BRSR Core: 'A+' rating for FY24""")

# ── PAGE 6: Risks + Outlook ───────────────────────────────────────────────
pdf.add_page()
pdf.section_title("11. RISK MANAGEMENT FRAMEWORK")
pdf.body("""IndiaCorp follows a Board-mandated enterprise risk management framework. Key risks identified for FY 2024-25:

1. COMMODITY PRICE RISK (High)
   Gross refining margins (GRM) are exposed to crude oil price volatility. Jamnagar refinery GRM in FY24 was USD 9.3/bbl vs. USD 11.2/bbl in FY23. Every USD 1/bbl change in GRM impacts EBITDA by ~Rs.3,800 Crore. Hedging policy covers 40-60% of rolling 3-month crude exposure via commodity swaps.

2. CURRENCY RISK (Medium-High)
   ~62% of revenues from O2C are USD-denominated exports. A 1% rupee depreciation against the dollar positively impacts EBITDA by ~Rs.1,200 Crore annually. USD/INR forward contracts and cross-currency swaps hedge 85% of net USD exposure.

3. REGULATORY & GEOPOLITICAL RISK (Medium)
   KG-D6 gas pricing is subject to Ministry of Petroleum review. The current APM ceiling price of USD 9.92/MMBtu is at risk in the FY25 revision cycle. US sanctions on Russian crude could raise feedstock costs by 3-4%.

4. TELECOM SECTOR RISK (Medium)
   JioConnect faces potential spectrum auction pricing pressure in the upcoming 26 GHz mmWave band allocation. Competition from Airtel's converged b2b offerings may impact ARPU growth in Q1-Q2 FY25.

5. NEW ENERGY EXECUTION RISK (Medium-Low)
   The 5 GW solar module giga-factory in Jamnagar is 14 months behind schedule due to supply chain delays in polysilicon procurement from Southeast Asia. Commissioning now expected by Q3 FY25 (original: Q1 FY25).

6. CYBERSECURITY RISK (High)
   As a consequence of digitalisation across retail (465 million loyalty members) and telecom (506 million subscribers), IndiaCorp faces elevated data breach exposure. ISO 27001 certification achieved across 8 critical data centres; CERT-In compliance fully maintained.""")

pdf.section_title("12. MANAGEMENT DISCUSSION & ANALYSIS - OUTLOOK FY 2024-25")
pdf.body("""MACRO ENVIRONMENT:
India's GDP is projected to grow at 7.0-7.2% in FY25 (IMF estimate), making it the fastest-growing major economy globally. Private consumption recovery, robust government capex (Rs.11.1 Lakh Crore budgeted), and sustained FDI inflows create a strong tailwind for IndiaCorp's domestic-facing businesses.

O2C OUTLOOK:
GRMs are expected to recover to USD 10.5-11.0/bbl in H2 FY25 as Russian crude supply normalises and Chinese demand recovers post-COVID destocking. Petchem ethylene spreads, currently at decade-lows, are expected to bottom out by Q2 FY25.

RETAIL OUTLOOK:
IndiaStore targets 25,000 stores by March 2025 (from 19,531 as of March 2024). Fashion & Lifestyle GMV is expected to grow 28-32% driven by the acquisition of Ed-a-Mamma and integration of Just Dial's hyperlocal discovery platform. B2B Commerce JioMart Partner is targeting Rs.50,000 Crore GMV by FY26.

DIGITAL SERVICES OUTLOOK:
JioConnect ARPU (Average Revenue Per User) is targeted at Rs.230 by FY25 (FY24: Rs.181.7), driven by premium 5G plan uptake. 5G rollout to be complete in all 1 lakh towns with population >10,000 by Q2 FY25. JioFinance NBFC is targeting a loan book of Rs.25,000 Crore by FY25-end.

GUIDANCE (Non-binding):
- Revenue Growth: 12-16% YoY
- EBITDA Growth: 18-22% YoY
- Capex (FY25): Rs.1,75,000 - Rs.2,00,000 Crore
- Net Debt / EBITDA: Target < 0.5x by March 2025""")

pdf.section_title("13. STATUTORY AUDITOR'S NOTE (EXTRACT)")
pdf.body("""TO THE MEMBERS OF INDIACORP INDUSTRIES LIMITED

This is an extract of our report dated May 25, 2024. The full Auditor's Report is available on pages 180-224 of the full Annual Report.

Opinion:
In our opinion and to the best of our information and explanation given to us, the consolidated financial statements give a true and fair view in conformity with Ind AS, of the consolidated state of affairs of the Company as at March 31, 2024, and its consolidated profit, consolidated other comprehensive income, consolidated changes in equity and consolidated cash flows for the year ended on that date.

Key Audit Matters:
1. Revenue Recognition for Telecom Contracts (JioConnect): We tested revenue allocation models under Ind AS 115 for bundled plans. Appropriate disclosures made.
2. Valuation of KG-D6 Block Hydrocarbon Reserves: We involved our internal specialists to assess management's reserve estimates. ONGC Videsh confirmation cross-referenced.
3. Impairment Testing of Retail Goodwill (Rs.42,800 Crore): Recoverable amount supported by robust DCF model with WACC of 10.2% and terminal growth of 4.5%.

PricewaterhouseCoopers LLP | Chartered Accountants
ICAI Firm Registration Number: 012754N/N500016
Mumbai, May 25, 2024""")

# Save
out = "reports/IndiaCorp_FY2024_Annual_Report.pdf"
pdf.output(out)
print(f"✅ Realistic report saved to: {out}")
print(f"   Pages: {pdf.page}")
