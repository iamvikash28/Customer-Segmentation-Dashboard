# 📊 Customer-Segmentation-Dashboard-Power-BI
A professional RFM-based Customer Segmentation Dashboard built in Power BI, replicating a dark-themed HTML prototype. The dashboard segments customers into 7 behavioral groups using Recency, Frequency, and Monetary (RFM) analysis to drive targeted marketing actions.

---
## 🖼️ Dashboard Preview

The dashboard features:
- 5 KPI cards with dynamic delta indicators
- Revenue by Segment bar chart
- Customer Distribution donut chart
- Recency vs Spend bubble chart
- RFM Score Distribution stacked bar chart
- Segment Performance table with mini progress bars

---

## 📁 Repository Structure

```
├── dashboard.html                        # Original HTML prototype (reference design)
├── SegmentData.csv                       # Source data for all 7 customer segments
├── CustomerSegmentation_PBI_Theme.json   # Power BI dark theme (import-ready)
├── CustomerSegmentation.pbix            # Power BI report file
└── README.md                            # Project documentation
```

---
## 📊 Dashboard Visuals

### KPI Cards (5)
| KPI | Value | Delta |
|---|---|---|
| Total Customers | 9,021 | ▲ +8.2% vs last year |
| Total Revenue | $5,592K | ▲ +12.4% vs last year |
| Avg Order Value | $105 | ▲ +3.1% vs last year |
| Avg Recency (days) | 543d | ▼ +6 days vs last year |
| Champions Revenue Share | 41% | ▲ +2.1pp vs last year |

### Charts
- **Revenue by Segment** — Horizontal bar chart, each bar colored by segment
- **Customer Distribution** — Donut chart (62% inner radius) showing customer count share
- **Recency vs Spend** — Bubble chart where bubble size = avg frequency
- **RFM Score Distribution** — Stacked bar chart showing R, F, M scores per segment

### Segment Performance Table
Columns: Segment pill · Customers · Revenue · Revenue Share % (with mini bar) · Avg AOV · Avg Recency · Action

---
## 🚀 Getting Started

### Prerequisites
- Power BI Desktop (latest version recommended)
- Basic knowledge of Power BI visuals and DAX

### Setup Steps

**1. Clone the repository**
```bash
git clone https://github.com/iamvikash28/Customer-Segmentation-Dashboard-Power-BI.git
cd Customer-Segmentation-Dashboard-Power-BI
```
**2. Import the Theme**
1. Open Power BI Desktop
2. Go to **View → Themes → Browse for themes**
3. Select `CustomerSegmentation_PBI_Theme.json`

**3. Load the Data**
1. Go to **Home → Get Data → Text/CSV**
2. Select `segments_labeled.csv`
3. Click **Load**
4. Set column data types in Power Query:

| Column | Type |
|---|---|
| Segment, Action | Text |
| Customers, Revenue, Avg AOV, Avg Recency | Whole Number |
| Avg Frequency, R Score, F Score, M Score | Decimal Number |

**4. Open the Report**
- Open `dashboard.pbix` in Power BI Desktop

---

## 🧮 Key DAX Measures

```dax
-- Total Customers
Total Customers = SUM('SegmentData'[Customers])

-- Total Revenue
Total Revenue = SUM('SegmentData'[Revenue])

-- Weighted Avg Order Value
Avg Order Value =
DIVIDE(
    SUMX('SegmentData', 'SegmentData'[Avg AOV] * 'SegmentData'[Customers]),
    SUM('SegmentData'[Customers])
)

-- Weighted Avg Recency
Avg Recency Days =
DIVIDE(
    SUMX('SegmentData', 'SegmentData'[Avg Recency] * 'SegmentData'[Customers]),
    SUM('SegmentData'[Customers])
)

-- Champions Revenue Share
Champions Revenue Share =
DIVIDE(
    CALCULATE(SUM('SegmentData'[Revenue]), 'SegmentData'[Segment] = "Champions"),
    CALCULATE(SUM('SegmentData'[Revenue]), ALL('SegmentData'))
)

-- Revenue Share % (per segment)
Revenue Share % =
DIVIDE(
    SUM('SegmentData'[Revenue]),
    CALCULATE(SUM('SegmentData'[Revenue]), ALL('SegmentData'))
)

-- Avg Recency with "d" suffix (display)
Avg Recency Display =
FORMAT(AVERAGE('SegmentData'[Avg Recency]), "0") & "d"

-- Segment bar color (for conditional formatting)
Revenue Share Bar Color =
SWITCH(
    SELECTEDVALUE('SegmentData'[Segment]),
    "Champions",             "#378ADD",
    "Loyal customers",       "#1D9E75",
    "Potential loyalists",   "#7F77DD",
    "New customers",         "#EF9F27",
    "At-risk customers",     "#D85A30",
    "Frequent low-spenders", "#D4537E",
    "Lost / dormant",        "#888780",
    "#378ADD"
)
```
---
## 📦 Data Source

The `segments_labeled.csv` contains 7 rows (one per segment) with the following fields:

| Field | Description |
|---|---|
| Segment | Customer segment name |
| Customers | Number of customers in segment |
| Revenue | Total revenue generated ($) |
| Avg AOV | Average order value ($) |
| Avg Recency | Average days since last purchase |
| Avg Frequency | Average number of purchases |
| R Score | Recency score (1–5) |
| F Score | Frequency score (1–5) |
| M Score | Monetary score (1–5) |
| Action | Recommended marketing action |

---
## 👤 Author

**Vikash Verma**
Aspiring Data Analyst | Excel · SQL · Power BI · Python

---
