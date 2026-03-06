# Amazon Product Review Summarizer

A web app that lets users browse products and view **sentiment breakdown** (positive / neutral / negative) and a **summary** of customer reviews in a single click. Helps shoppers decide faster and reduce purchase time.

## Features

- **Amazon-style UI**: Product grid with search, images, ratings, and prices
- **Sentiment breakdown**: Counts of positive, neutral, and negative reviews (from star ratings)
- **Review summary**: Summary from the most relevant reviews for each product
- **Popup details**: Click "See review summary" to open sentiment and summary in a modal
- **Expandable comments**: Customer review text available in a collapsed section

## Prerequisites

- **Python 3.8 or higher**
- **~2–3 GB free disk space** (for ML models; downloaded on first run)

## Setup

### 1. Clone the repository

```bash
git clone <your-repo-url>
cd Mtech-Capstone-Project
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv venv
```

- **Windows:**  
  `venv\Scripts\activate`
- **macOS / Linux:**  
  `source venv/bin/activate`

### 3. Install dependencies

```bash
pip install -r requirement.txt
```

Installation may take a few minutes (PyTorch, Transformers, Sentence-Transformers).

### 4. Add your data

Place your CSV files so the app can find them. By default it looks for:

| File | Default path |
|------|----------------|
| Product metadata | `src/chinmay/Appliances_meta.csv` |
| Reviews | `src/chinmay/Appliances_reviews.csv` |

**Metadata CSV** should include columns such as: `title`, `parent_asin`, `average_rating`, `rating_number`, `price`, `images` (optional).

**Reviews CSV** should include: `parent_asin`, `text`, `rating` (1–5 stars).

To use different paths, set environment variables before running:

```bash
export META_CSV="path/to/your_meta.csv"
export REVIEWS_CSV="path/to/your_reviews.csv"
```

## Run the app

From the project root:

```bash
streamlit run app.py
```

The app will open in your browser (usually `http://localhost:8501`). If it doesn’t, open that URL manually.

## Usage

1. Use the **search box** to filter products by name.
2. Click **"See review summary"** on any product card.
3. In the popup you’ll see:
   - Sentiment counts (positive / neutral / negative)
   - Overall summary
   - Quick takeaway
   - Expandable **Customer comments** with full review text

## Project structure (relevant parts)

```
Mtech-Capstone-Project/
├── app.py              # Streamlit app (UI + sentiment + summary)
├── requirement.txt     # Python dependencies
├── README.md            # This file
└── src/chinmay/        # Default location for CSVs
    ├── Appliances_meta.csv
    └── Appliances_reviews.csv
```

## Troubleshooting

- **"Could not load data"**  
  Check that the CSV paths exist and that `META_CSV` / `REVIEWS_CSV` point to the correct files if you set them.

- **First run is slow**  
  The first run downloads embedding and summarization models (~1–2 GB). Later runs use the cache and are faster.

- **Out of memory**  
  The app uses CPU by default. If you run out of RAM, reduce `MAX_REVIEWS_PER_PRODUCT` or `MAX_REVIEWS_FOR_SUMMARY` in the config at the top of `app.py`.

## License

Not specified.
