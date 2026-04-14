from fpdf import FPDF

class PDF(FPDF):
    def header(self):
        self.set_font("Helvetica", "B", 15)
        self.cell(0, 10, "GlobalTech Industries - Q3 2023 Earnings Report", new_x="LMARGIN", new_y="NEXT", align="C")
        self.ln(10)

    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.cell(0, 10, f"Page {self.page_no()}", align="C")

pdf = PDF()
pdf.add_page()
pdf.set_font("Helvetica", size=12)

content = """
Company Name: GlobalTech Industries
Report Period: Q3 2023

1. Executive Summary
GlobalTech Industries experienced robust growth in the third quarter of 2023. We saw significant expansion in our cloud infrastructure footprint and consumer hardware division.

2. Financial Highlights
- Total Revenue for the quarter was $42.5 Billion, representing a 14% year-over-year revenue growth.
- Net Income reported was $8.2 Billion. Our net margin now stands at approximately 19.3%.
- EBITDA reached $12.4 Billion, driven by operational efficiencies.
- Earnings Per Share (EPS) for the quarter was $1.45.
- Operating Cash Flow was strong at $10.1 Billion.

3. Balance Sheet Summary
- Total Assets as of the end of Q3 2023 were $140.5 Billion.
- Total Debt stands at $28.0 Billion.
- Our Debt-to-Equity ratio is a healthy 0.45.
- Return on Equity (ROE) remains outstanding at 28.5%.

4. Key Highlights & Risks
Key Highlight: The successful launch of the 'Nexus Server Rack' product line exceeded expectations, contributing $4.2B in unprecedented enterprise sales.
Risk Flag: Global supply chain constraints regarding semiconductor microchips continue to pose a moderate delay risk for the upcoming Q4 consumer hardware pipeline.

5. Future Outlook
We remain highly optimistic about Q4. While supply chains remain tight, our diversified revenue streams and high free cash flow position us to weather macroeconomic headwinds while continuing aggressive R&D investments.
"""

for line in content.split("\n"):
    pdf.multi_cell(0, 10, txt=line)

pdf.output("reports/GlobalTech_Q3_2023_Earnings.pdf")
print("PDF created successfully!")
