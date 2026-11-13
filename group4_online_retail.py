"""Group 4, Online Retail II: the whole analysis in one file.

How to run (any computer, no path to edit):

    pip install pandas==2.3.3 numpy==2.5.4 openpyxl matplotlib scipy scikit-learn==1.8.0 xgboost==3.1.3 mlxtend shap
    python group4_online_retail.py

The script looks for online_retail_II.xlsx next to this file, in the folder you run
it from, or in Downloads / Desktop / Documents. If it is not there, it downloads the
file from UCI. To point at a file yourself: python group4_online_retail.py path/to/online_retail_II.xlsx

All tables and figures go to the "outputs" folder next to this file.
Data: Chen, D. (2012). Online Retail II. UCI ML Repository. https://doi.org/10.24432/C5CG6D
"""
import sys
import time
import urllib.request
import warnings
import zipfile
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use('Agg')  # save figures to files, never open windows

warnings.filterwarnings('ignore')

HERE = Path(__file__).resolve().parent if '__file__' in globals() else Path.cwd()  # Jupyter has no __file__
OUTPUT = HERE / 'outputs'
OUTPUT.mkdir(exist_ok=True)
DATA_NAME = 'online_retail_II.xlsx'
UCI_URL = 'https://archive.ics.uci.edu/static/public/502/online+retail+ii.zip'


def find_data():
    """Return the path of online_retail_II.xlsx, downloading it from UCI if it is nowhere nearby."""
    args = [a for a in sys.argv[1:] if a.lower().endswith('.xlsx')]
    if args:
        return Path(args[0]).expanduser().resolve()
    home = Path.home()
    for folder in [HERE, Path.cwd(), home / 'Downloads', home / 'Desktop', home / 'Documents']:
        # rglob also looks inside subfolders, e.g. Downloads/online+retail+ii/online_retail_II.xlsx
        for path in folder.rglob(DATA_NAME) if folder.is_dir() else []:
            return path
    print(f'{DATA_NAME} not found, downloading it from UCI (about 45 MB)...')
    zip_path = HERE / 'online_retail_II.zip'
    urllib.request.urlretrieve(UCI_URL, zip_path)
    with zipfile.ZipFile(zip_path) as z:
        z.extract(DATA_NAME, HERE)
    return HERE / DATA_NAME


RAW_FILE = None  # set in main()

# ============================================================================
# Configuration (former config.py)
# ============================================================================

FIRST_DAY = '2009-12-01'
LAST_DAY = '2011-12-09'
FINAL_AS_OF = '2011-12-09'

# Merchandise = every StockCode except the operational fees, discounts, manual
# adjustments and test items below, and the gift-voucher prefixes.
NON_MERCHANDISE_CODES = {
    'POST', 'DOT', 'M', 'C2', 'D', 'S', 'BANK CHARGES',
    'AMAZONFEE', 'CRUK', 'B', 'ADJUST', 'ADJUST2',
    'TEST001', 'TEST002', '23444'
}
# Gift vouchers (GIFT_0001_10, GIFT_0001_20, ...) are prepaid credit, not products.
NON_MERCHANDISE_PREFIXES = ('GIFT_',)

SEGMENT_NAMES = {
    0: 'Cooling Customers',
    1: 'Loyal Customers',
    2: 'Inactive Customers',
    3: 'Growing Customers'
}
RANDOM_STATE = 42

# ============================================================================
# Shared helpers (former common.py)
# ============================================================================
import numpy as np
import pandas as pd


def load_table(filename):
    return pd.read_csv(OUTPUT / filename, parse_dates=['InvoiceDate'],
                       dtype={'Invoice': 'string', 'StockCode': 'string', 'CustomerID': 'Int64'})


def load_customers(filename):
    return pd.read_csv(OUTPUT / filename, dtype={'CustomerID': 'Int64'})


def build_customer_features(purchases, returns, as_of):
    """Create features using observed transactions up to a decision date."""
    date = pd.Timestamp(as_of)
    end = date + pd.Timedelta(days=1)
    p = purchases[(purchases['InvoiceDate'] < end) & purchases['CustomerID'].notna()].copy()
    r = returns[(returns['InvoiceDate'] < end) & returns['CustomerID'].notna()].copy()
    if p.empty:
        return pd.DataFrame()

    customer = p.groupby('CustomerID').agg(
        FirstPurchaseDate=('InvoiceDate', 'min'),
        LastPurchaseDate=('InvoiceDate', 'max'),
        FrequencyInvoices=('Invoice', 'nunique'),
        GrossMerchandiseSpend=('LineRevenue', 'sum'),
        TotalMerchandiseUnits=('Quantity', 'sum'),
        ProductDiversity=('StockCode', 'nunique')
    )
    customer['RecencyDays'] = (end - customer['LastPurchaseDate']).dt.days
    customer['CustomerTenureDays'] = (end - customer['FirstPurchaseDate']).dt.days
    customer['ActiveSpanDays'] = (customer['LastPurchaseDate'] - customer['FirstPurchaseDate']).dt.days

    invoice = p.groupby(['CustomerID', 'Invoice'], observed=True).agg(
        InvoiceDate=('InvoiceDate', 'min'),
        InvoiceRevenue=('LineRevenue', 'sum'),
        InvoiceUnits=('Quantity', 'sum')
    ).reset_index()
    inv = invoice.groupby('CustomerID').agg(
        AvgBasketValue=('InvoiceRevenue', 'mean'),
        AvgBasketUnits=('InvoiceUnits', 'mean')
    )
    customer = customer.join(inv)

    # Purchase rhythm is defined over DISTINCT CALENDAR PURCHASE DAYS, not invoices.
    # Multiple same-day invoices count as one purchase day when measuring gaps.
    purchase_days = invoice[['CustomerID', 'InvoiceDate']].copy()
    purchase_days['PurchaseDay'] = purchase_days['InvoiceDate'].dt.normalize()
    purchase_days = (purchase_days[['CustomerID', 'PurchaseDay']]
                     .drop_duplicates()
                     .sort_values(['CustomerID', 'PurchaseDay']))
    purchase_days['GapDays'] = (purchase_days.groupby('CustomerID')['PurchaseDay']
                               .diff().dt.days.astype(float))
    gaps = purchase_days.groupby('CustomerID')['GapDays'].agg(['mean', 'median', 'std'])
    customer['PurchaseDaysObserved'] = purchase_days.groupby('CustomerID').size()
    customer['AvgInterpurchaseDays'] = gaps['mean']
    customer['MedianInterpurchaseDays'] = gaps['median']
    customer['InterpurchaseStdDays'] = gaps['std']

    for days in [30, 90]:
        recent_p = p[p['InvoiceDate'] >= end - pd.Timedelta(days=days)]
        customer[f'Recent{days}GrossSpend'] = recent_p.groupby('CustomerID')['LineRevenue'].sum()
        customer[f'PurchaseInvoices{days}'] = recent_p.groupby('CustomerID')['Invoice'].nunique()

    previous30 = p[(p['InvoiceDate'] < end - pd.Timedelta(days=30)) &
                   (p['InvoiceDate'] >= end - pd.Timedelta(days=60))]
    previous90 = p[(p['InvoiceDate'] < end - pd.Timedelta(days=90)) &
                   (p['InvoiceDate'] >= end - pd.Timedelta(days=180))]
    customer['Previous30GrossSpend'] = previous30.groupby('CustomerID')['LineRevenue'].sum()
    customer['Previous90GrossSpend'] = previous90.groupby('CustomerID')['LineRevenue'].sum()

    if not r.empty:
        r = r.copy()
        r['ReturnValueLine'] = -r['LineRevenue']
        rg = r.groupby('CustomerID').agg(
            ReturnValue=('ReturnValueLine', 'sum'),
            ReturnInvoiceCount=('Invoice', 'nunique'),
            ReturnLineCount=('Invoice', 'size')
        )
        customer = customer.join(rg)
    else:
        customer['ReturnValue'] = 0.0
        customer['ReturnInvoiceCount'] = 0
        customer['ReturnLineCount'] = 0

    customer['PrimaryCountry'] = p.groupby('CustomerID')['Country'].agg(
        lambda s: s.mode().iloc[0] if not s.mode().empty else 'Unknown')
    customer['CountryCount'] = p.groupby('CustomerID')['Country'].nunique()
    customer['IsUK'] = (customer['PrimaryCountry'] == 'United Kingdom').astype(int)

    weekend = invoice.assign(IsWeekend=(invoice['InvoiceDate'].dt.dayofweek >= 5).astype(int))
    customer['WeekendInvoiceCount'] = weekend.groupby('CustomerID')['IsWeekend'].sum()

    numeric = customer.select_dtypes(include='number').columns.difference(['AvgInterpurchaseDays','MedianInterpurchaseDays','InterpurchaseStdDays'])
    customer[numeric] = customer[numeric].fillna(0)
    customer['MonetaryNet'] = customer['GrossMerchandiseSpend'] - customer['ReturnValue']
    customer['ReturnInvoiceShare'] = customer['ReturnInvoiceCount'] / customer['FrequencyInvoices'].clip(lower=1)
    customer['ReturnValueRatio'] = customer['ReturnValue'] / customer['GrossMerchandiseSpend'].clip(lower=1e-9)
    customer['HasReturn'] = (customer['ReturnInvoiceCount'] > 0).astype(int)
    customer['SinglePurchaseFlag'] = (customer['FrequencyInvoices'] == 1).astype(int)
    customer['WeekendInvoiceShare'] = customer['WeekendInvoiceCount'] / customer['FrequencyInvoices'].clip(lower=1)
    customer['SpendMomentum'] = ((customer['Recent30GrossSpend'] - customer['Previous30GrossSpend']) /
                                 (customer['Previous30GrossSpend'] + 1))
    month_angle = 2 * np.pi * date.month / 12
    customer['MonthSin'] = np.sin(month_angle)
    customer['MonthCos'] = np.cos(month_angle)
    return customer.reset_index()


def score_top20(y_true, score):
    y = np.asarray(y_true)
    s = np.asarray(score)
    count = max(1, int(np.ceil(len(y) * 0.20)))
    idx = np.argsort(-s)[:count]
    overall = float(y.mean())
    response = float(y[idx].mean())
    return {'Top20Rate': response, 'LiftAt20': response / overall if overall else np.nan,
            'CaptureAt20': float(y[idx].sum() / y.sum()) if y.sum() else np.nan,
            'Top20N': count}

# ============================================================================
# Association-rule helpers (former affinity_tools.py)
# ============================================================================
"""Association rules and rule-based recommendations (Steps 04, 06, 08).

Rule convention used in the report (Section 5): itemsets of up to three products,
one-product consequent, confidence >= 30% and lift >= 1.20. Support thresholds are
passed in by the caller (1% market; 1.5% and at least 50 baskets within a segment).
FP-Growth (mlxtend) is used when installed; the exact-count fallback returns the
same itemset counts and therefore the same rules.
"""
import math
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer


def _counting_fallback(baskets, min_count):
    """Count supported 1-, 2-, and 3-itemsets (equivalent patterns, not FP-Growth)."""
    from collections import Counter, defaultdict
    from itertools import combinations
    n = len(baskets)
    appearances = defaultdict(set)
    for row, basket in enumerate(baskets):
        for item in basket:
            appearances[item].add(row)
    singles = {k:len(v) for k,v in appearances.items() if len(v)>=min_count}
    items = sorted(singles)
    pairs = {}
    for a,b in combinations(items,2):
        count = len(appearances[a] & appearances[b])
        if count>=min_count:
            pairs[(a,b)] = count
    triples = {}
    for a,b in pairs:
        for c in items:
            if c <= b:
                continue
            if (a,c) in pairs and (b,c) in pairs:
                count = len(appearances[a] & appearances[b] & appearances[c])
                if count>=min_count:
                    triples[(a,b,c)] = count
    return singles,pairs,triples


def fp_rules(lines, support=0.01, min_count=1):
    """Return rules with one product consequent and up to two antecedent products."""
    baskets = lines.groupby('Invoice')['StockCode'].agg(lambda s:sorted(set(s.dropna()))).tolist()
    cols = ['AntecedentCodes','ConsequentCode','Support','Confidence','Lift']
    if not baskets:
        return pd.DataFrame(columns=cols)
    n = len(baskets)
    threshold = max(min_count, math.ceil(n * support))
    try:
        from mlxtend.frequent_patterns import fpgrowth
        encoder = MultiLabelBinarizer(sparse_output=True)
        matrix = encoder.fit_transform(baskets)
        df = (pd.DataFrame.sparse.from_spmatrix(matrix,columns=encoder.classes_)
              .astype(pd.SparseDtype(bool, False)))
        frequent = fpgrowth(df,min_support=threshold/n,max_len=3,use_colnames=True)
        item_counts = {tuple(sorted(items)):int(round(score*n)) for items,score in
                       zip(frequent['itemsets'],frequent['support'])}
        singles = {k[0]:v for k,v in item_counts.items() if len(k)==1}
        pairs = {k:v for k,v in item_counts.items() if len(k)==2}
        triples = {k:v for k,v in item_counts.items() if len(k)==3}
    except ImportError:
        print('mlxtend is unavailable; using exact support-count fallback (not FP-Growth).')
        singles,pairs,triples = _counting_fallback(baskets,threshold)

    records = []
    for key,count in [*pairs.items(),*triples.items()]:
        for consequent in key:
            antecedent = tuple(item for item in key if item != consequent)
            antecedent_count = singles.get(antecedent[0],0) if len(antecedent)==1 else pairs.get(antecedent,0)
            if not antecedent_count:
                continue
            confidence = count / antecedent_count
            lift = confidence / (singles[consequent]/n)
            if confidence>=.30 and lift>=1.20:
                records.append(['|'.join(antecedent),consequent,count/n,confidence,lift])
    # Rank by confidence, lift, support; rule codes break exact ties so the order never depends on sort internals.
    return (pd.DataFrame(records,columns=cols)
            .sort_values(['Confidence','Lift','Support','AntecedentCodes','ConsequentCode'],
                         ascending=[False,False,False,True,True],kind='mergesort')
            .reset_index(drop=True))


def recommend_from_rules(last_basket,rules,max_items=3):
    """Walk the ranked rules; keep consequents whose whole antecedent is in the basket."""
    codes = set(last_basket)
    chosen = []
    for row in rules.itertuples(index=False):
        ant = set(row.AntecedentCodes.split('|'))
        if ant.issubset(codes) and row.ConsequentCode not in codes:
            if row.ConsequentCode not in chosen:
                chosen.append(row.ConsequentCode)
        if len(chosen)==max_items:
            break
    return chosen


def recommend_with_fallback(last_basket, segment_rules, market_rules, max_items=3):
    """Return suggestions and source under the project's ZERO-CANDIDATE fallback.

    The market model is used only when a segment model cannot propose any
    products. It does not top up an already nonempty segment recommendation.
    This single helper is shared by backtesting (Step 06) and production (08).
    """
    local = (recommend_from_rules(last_basket, segment_rules, max_items)
             if segment_rules is not None else [])
    if local:
        return local, 'Segment rules'
    market = recommend_from_rules(last_basket, market_rules, max_items)
    return market, ('Overall fallback' if market else 'No triggered rule')


# ============================================================================
# Step 01 (former 01_data_preparation.py)
# ============================================================================
def step_01_data_preparation():
    """Step 1: read Online Retail II, keep merchandise lines, remove duplicates and cancelled orders.

Order of the cleaning rules (each count is written to 01_data_audit.csv):
1. Sheet overlap: keep the first sheet before 1 Dec 2010 and the second sheet from then on.
2. Valid merchandise purchase: positive quantity and price, invoice not starting with "C",
   StockCode not an operational code or gift voucher (config.py).
3. Duplicate lines: a line identical to an earlier line in every field is dropped.
4. Cancelled purchases: a return line ("C" invoice) that exactly reverses an earlier purchase
   line of the same customer (same StockCode, quantity and price) marks that purchase as
   cancelled; both lines are dropped. Example: invoice 581483 (80,995 units, GBP 168,469.60)
   and C581484 twelve minutes later. Each return line cancels at most one purchase line, the
   latest one on or before the return.
5. Remaining return lines of identified customers are kept to compute net value.
"""
    import datetime as dt
    from collections import defaultdict
    import pandas as pd
    from openpyxl import load_workbook

    if not RAW_FILE.exists():
        raise FileNotFoundError(f'Place online_retail_II.xlsx in {RAW_FILE.parent}')

    columns = ['Invoice','StockCode','Description','Quantity','InvoiceDate',
               'Price','CustomerID','Country','LineRevenue']
    counts = {'Raw rows':0,'After sheet-overlap rule':0,'Valid positive purchase lines':0}
    purchases, returns = [], []

    workbook = load_workbook(RAW_FILE, read_only=True, data_only=True)
    cutoff = dt.datetime(2010,12,1)
    for sheet_name in ['Year 2009-2010','Year 2010-2011']:
        worksheet = workbook[sheet_name]
        for values in worksheet.iter_rows(min_row=2,values_only=True):
            counts['Raw rows']+=1
            invoice, stock, description, quantity, date, price, cid, country = values
            if not isinstance(date,dt.datetime):
                continue
            if sheet_name == 'Year 2009-2010' and date >= cutoff:
                continue
            if sheet_name == 'Year 2010-2011' and date < cutoff:
                continue
            counts['After sheet-overlap rule']+=1
            if not isinstance(quantity,(int,float)) or not isinstance(price,(int,float)):
                continue
            stock = str(stock).strip().upper() if stock is not None else ''
            invoice = str(int(invoice)) if isinstance(invoice,(int,float)) else str(invoice).strip()
            if cid is not None and not isinstance(cid,str):
                cid = str(int(cid))
            has_id = cid is not None and str(cid).strip() != ''
            is_merchandise = stock not in NON_MERCHANDISE_CODES and not stock.startswith(NON_MERCHANDISE_PREFIXES)
            is_cancel = invoice.upper().startswith('C')
            # Figure 2: count valid positives BEFORE excluding operational codes.
            if price > 0 and quantity > 0 and not is_cancel:
                counts['Valid positive purchase lines'] += 1
            if not is_merchandise or price <= 0:
                continue
            row = [invoice,stock,description,quantity,date.strftime('%Y-%m-%d %H:%M:%S'),
                   price,cid if has_id else '',country,quantity*price]
            if quantity > 0 and not is_cancel:
                purchases.append(row)
            elif quantity < 0 and is_cancel and has_id:
                returns.append(row)
        print(sheet_name, 'processed', flush=True)

    purchases = pd.DataFrame(purchases, columns=columns)
    returns = pd.DataFrame(returns, columns=columns)
    counts['Valid merchandise purchase lines'] = len(purchases)

    # Rule 3: exact duplicate lines.
    duplicate_p = purchases.duplicated(keep='first')
    duplicate_r = returns.duplicated(keep='first')
    counts['Duplicate purchase lines removed'] = int(duplicate_p.sum())
    counts['Duplicate return lines removed'] = int(duplicate_r.sum())
    purchases = purchases[~duplicate_p].reset_index(drop=True)
    returns = returns[~duplicate_r].reset_index(drop=True)

    # Rule 4: cancelled purchases (exact reversal by the same customer).
    candidates = defaultdict(list)  # (customer, item, price, quantity) -> purchase rows, oldest first
    identified = purchases[purchases['CustomerID'] != '']
    for idx, cid, stock, price, qty, date in zip(identified.index, identified['CustomerID'], identified['StockCode'],
                                                 identified['Price'], identified['Quantity'], identified['InvoiceDate']):
        candidates[(cid, stock, price, qty)].append((date, idx))
    for key in candidates:
        candidates[key].sort()
    cancelled_p, cancelled_r = [], []
    order = returns.sort_values(['InvoiceDate','Invoice'], kind='mergesort').index
    for idx in order:
        r = returns.loc[idx]
        pool = candidates.get((r.CustomerID, r.StockCode, r.Price, -r.Quantity))
        if not pool:
            continue
        for k in range(len(pool) - 1, -1, -1):  # latest purchase on or before the return
            if pool[k][0] <= r.InvoiceDate:
                cancelled_p.append(pool.pop(k)[1]); cancelled_r.append(idx)
                break
    pairs = pd.concat([purchases.loc[cancelled_p, ['CustomerID','StockCode','Quantity','Price','LineRevenue','Invoice','InvoiceDate']]
                       .reset_index(drop=True).rename(columns={'Invoice':'PurchaseInvoice','InvoiceDate':'PurchaseDate'}),
                       returns.loc[cancelled_r, ['Invoice','InvoiceDate']].reset_index(drop=True)
                       .rename(columns={'Invoice':'ReturnInvoice','InvoiceDate':'ReturnDate'})], axis=1)
    pairs.sort_values('LineRevenue', ascending=False).to_csv(OUTPUT / '01_cancelled_purchase_pairs.csv', index=False)
    counts['Cancelled purchase lines removed'] = len(cancelled_p)
    purchases = purchases.drop(index=cancelled_p)
    returns = returns.drop(index=cancelled_r)

    counts['Purchase lines after duplicates and cancellations'] = len(purchases)
    identified = purchases[purchases['CustomerID'] != '']
    counts['Identified merchandise purchase lines'] = len(identified)
    counts['Recognized identified return lines'] = len(returns)
    counts['Customers with purchase history'] = identified['CustomerID'].nunique()
    counts['Merchandise purchase baskets'] = purchases['Invoice'].nunique()

    purchases.to_csv(OUTPUT / '01_merchandise_purchases.csv', index=False)
    identified.to_csv(OUTPUT / '01_identified_purchases.csv', index=False)
    returns.to_csv(OUTPUT / '01_recognized_returns.csv', index=False)
    summary=pd.DataFrame(list(counts.items()),columns=['Stage','Count'])
    summary.to_csv(OUTPUT/'01_data_audit.csv',index=False)
    print(summary.to_string(index=False))
    print(f"Cancelled purchases removed: {len(cancelled_p):,} lines, GBP {pairs['LineRevenue'].sum():,.2f}")


# ============================================================================
# Step 02 (former 02_descriptive_analysis.py)
# ============================================================================
def step_02_descriptive_analysis():
    """Step 2: simple descriptive statistics and figures."""
    import math
    import pandas as pd
    import matplotlib.pyplot as plt

    purchases = load_table('01_merchandise_purchases.csv')
    returns = load_table('01_recognized_returns.csv')
    features = build_customer_features(purchases, returns, '2011-12-09')
    features.to_csv(OUTPUT / '02_customer_features.csv', index=False)

    monthly = purchases.assign(Month=purchases['InvoiceDate'].dt.to_period('M').astype(str))
    monthly = monthly.groupby('Month')['LineRevenue'].sum().reset_index(name='GrossMerchandiseRevenue')
    monthly.to_csv(OUTPUT / '02_monthly_revenue.csv', index=False)

    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.plot(monthly['Month'], monthly['GrossMerchandiseRevenue'] / 1e6, marker='o')
    ax.set_title('Monthly gross merchandise revenue')
    ax.set_ylabel('Revenue (million GBP)')
    ax.tick_params(axis='x', rotation=65)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUTPUT / '02_monthly_revenue.png', dpi=170)
    plt.close(fig)

    eligible = features[features['MonetaryNet'] > 0].sort_values('MonetaryNet', ascending=False)
    shares = []
    for pct in [1, 5, 10, 20]:
        n = max(1, math.ceil(len(eligible) * pct / 100))
        shares.append((pct, eligible['MonetaryNet'].head(n).sum() / eligible['MonetaryNet'].sum()))
    concentration = pd.DataFrame(shares, columns=['TopCustomerPct', 'NetValueShare'])
    concentration.to_csv(OUTPUT / '02_customer_concentration.csv', index=False)
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.bar(concentration['TopCustomerPct'].astype(str) + '%', concentration['NetValueShare'] * 100)
    ax.set_title('Concentration of positive net merchandise value')
    ax.set_xlabel('Top customers')
    ax.set_ylabel('Net value share (%)')
    for i, v in enumerate(concentration['NetValueShare']):
        ax.text(i, v * 100 + 1, f'{v:.1%}', ha='center')
    ax.set_ylim(0, 90)
    fig.tight_layout()
    fig.savefig(OUTPUT / '02_customer_concentration.png', dpi=170)
    plt.close(fig)

    invoice_products = purchases.groupby('Invoice')['StockCode'].nunique()
    metrics = pd.DataFrame({'Metric': ['Customers', 'RepeatCustomerShare', 'MultiProductBasketShare', 'MedianDistinctProductsPerBasket'],
                            'Value': [len(features), (features['FrequencyInvoices'] >= 2).mean(),
                                      (invoice_products >= 2).mean(), invoice_products.median()]})
    metrics.to_csv(OUTPUT / '02_descriptive_summary.csv', index=False)
    print(metrics.to_string(index=False))


# ============================================================================
# Step 03 (former 03_rfm_segmentation.py)
# ============================================================================
def step_03_rfm_segmentation():
    """Step 3: RFM-based K-Means segmentation."""
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import silhouette_score, davies_bouldin_score, adjusted_rand_score

    customers = load_customers('02_customer_features.csv')
    eligible = customers[customers['MonetaryNet'] > 0].copy()
    exceptions = customers[customers['MonetaryNet'] <= 0].copy()

    rfm_columns = ['RecencyDays', 'FrequencyInvoices', 'MonetaryNet']
    X = np.log1p(eligible[rfm_columns].clip(lower=0))
    X = StandardScaler().fit_transform(X)

    scores = []
    stability_repeats = []
    for k in range(2, 9):
        labels = KMeans(n_clusters=k, n_init=30, random_state=RANDOM_STATE).fit_predict(X)
        ari_scores = []
        rng = np.random.default_rng(RANDOM_STATE)
        for repeat in range(3):
            ix = rng.choice(len(X), size=int(0.8 * len(X)), replace=False)
            model = KMeans(n_clusters=k, n_init=10, random_state=RANDOM_STATE + repeat + 1)
            sample_labels = model.fit_predict(X[ix])
            ari = adjusted_rand_score(labels[ix], sample_labels)
            ari_scores.append(ari)
            stability_repeats.append({'K':k, 'Repeat':repeat + 1, 'ARI':ari,
                                      'SubsampleShare':0.8})
        sizes = pd.Series(labels).value_counts()
        scores.append([k, silhouette_score(X, labels), davies_bouldin_score(X, labels),
                       sizes.min() / len(X), np.mean(ari_scores)])

    k_results = pd.DataFrame(scores, columns=['K', 'Silhouette', 'DaviesBouldin', 'SmallestClusterShare', 'StabilityARI'])
    k_results.to_csv(OUTPUT / '03_k_selection.csv', index=False)
    pd.DataFrame(stability_repeats).to_csv(OUTPUT / '03_stability_repeats.csv', index=False)

    eligible['ClusterId'] = KMeans(n_clusters=4, n_init=50, random_state=RANDOM_STATE).fit_predict(X)
    cluster_profile = eligible.groupby('ClusterId').agg(
        Customers=('CustomerID','size'),
        MedianRecency=('RecencyDays','median'),
        MedianFrequency=('FrequencyInvoices','median'),
        MedianNetValue=('MonetaryNet','median'),
        TotalNetValue=('MonetaryNet','sum')
    )
    # Names follow cluster behaviour, not the arbitrary integer assigned by K-Means.
    loyal_id = cluster_profile['MedianNetValue'].idxmax()
    inactive_id = cluster_profile['MedianRecency'].idxmax()
    other = [i for i in cluster_profile.index if i not in (loyal_id, inactive_id)]
    growing_id = cluster_profile.loc[other, 'MedianRecency'].idxmin()
    cooling_id = next(i for i in other if i != growing_id)
    name_map = {loyal_id:'Loyal Customers', inactive_id:'Inactive Customers',
                growing_id:'Growing Customers', cooling_id:'Cooling Customers'}
    eligible['CustomerSegment'] = eligible['ClusterId'].map(name_map)
    exceptions['ClusterId'] = -1
    exceptions['CustomerSegment'] = 'Net-Zero / Net-Negative Exception'
    all_customers = pd.concat([eligible, exceptions], ignore_index=True)
    all_customers.to_csv(OUTPUT / '03_customer_segments.csv', index=False)

    profile = all_customers.groupby('CustomerSegment').agg(
        Customers=('CustomerID','count'),
        MedianRecency=('RecencyDays','median'),
        MedianFrequency=('FrequencyInvoices','median'),
        MedianNetValue=('MonetaryNet','median'),
        TotalNetValue=('MonetaryNet','sum')
    ).reset_index()
    profile['CustomerSharePct'] = 100 * profile['Customers'] / len(all_customers)
    profile['NetValueSharePct'] = 100 * profile['TotalNetValue'] / all_customers['MonetaryNet'].sum()
    profile.to_csv(OUTPUT / '03_segment_profiles.csv', index=False)

    plot_data = profile[profile['CustomerSegment'].str.contains('Exception') == False]
    fig, ax = plt.subplots(figsize=(9, 5))
    xx = np.arange(len(plot_data))
    ax.bar(xx - 0.19, plot_data['CustomerSharePct'], width=0.38, label='Customer share')
    ax.bar(xx + 0.19, plot_data['NetValueSharePct'], width=0.38, label='Net value share')
    ax.set_xticks(xx, plot_data['CustomerSegment'], rotation=10)
    ax.set_ylabel('Share (%)')
    ax.legend()
    ax.set_title('Customer share compared with historical net value share')
    fig.tight_layout()
    fig.savefig(OUTPUT / '03_customer_share_value.png', dpi=170)
    plt.close(fig)
    print(profile.to_string(index=False))


# ============================================================================
# Step 04 (former 04_product_affinity.py)
# ============================================================================
def step_04_product_affinity():
    """Step 4: mine product associations from full-market and customer-segment baskets."""
    import pandas as pd

    purchases = load_table('01_merchandise_purchases.csv')
    segments = load_customers('03_customer_segments.csv')[['CustomerID','CustomerSegment']]
    purchases = purchases.merge(segments, on='CustomerID', how='left')

    names = (purchases.dropna(subset=['Description'])
             .groupby('StockCode')['Description'].agg(lambda s:s.value_counts().index[0]).to_dict())
    all_rules=[]

    for group, subset, minimum_support, minimum_count in [
        ('All Customers', purchases, .01, 1),
        *[(name,purchases[purchases['CustomerSegment']==name],.015,50)
          for name in ['Loyal Customers','Cooling Customers','Growing Customers','Inactive Customers']]
    ]:
        rules = fp_rules(subset,minimum_support,minimum_count)
        rules['Group']=group
        rules['BasketCount']=subset['Invoice'].nunique()
        rules['Antecedent'] = rules['AntecedentCodes'].map(
            lambda s:' + '.join(names.get(x,x) for x in s.split('|')))
        rules['Consequent'] = rules['ConsequentCode'].map(lambda c:names.get(c,c))
        rules['StrengthScore']=rules['Support']*rules['Confidence']*rules['Lift']
        all_rules.append(rules)
        print(f'{group}: {subset.Invoice.nunique():,} baskets, {len(rules):,} rules')

    pd.concat(all_rules,ignore_index=True).to_csv(OUTPUT/'04_association_rules.csv',index=False)


# ============================================================================
# Step 05 (former 05_customer_dynamics.py)
# ============================================================================
def step_05_customer_dynamics():
    """Step 5: recent spending momentum and purchase rhythm."""
    import numpy as np
    import pandas as pd

    purchases = load_table('01_identified_purchases.csv')
    segments = load_customers('03_customer_segments.csv')
    as_of = pd.Timestamp('2011-12-09')
    end = as_of + pd.Timedelta(days=1)


    def window_spend(start, stop):
        mask = (purchases['InvoiceDate'] >= start) & (purchases['InvoiceDate'] < stop)
        return purchases.loc[mask].groupby('CustomerID')['LineRevenue'].sum()


    def comparison(days):
        current = window_spend(end-pd.Timedelta(days=days), end)
        previous = window_spend(end-pd.Timedelta(days=2*days), end-pd.Timedelta(days=days))
        output = pd.DataFrame({'Current': current, 'Previous': previous}).fillna(0)
        # A fair comparison of two adjacent windows requires observation spanning
        # the complete 2*days interval. No zero-imputation for newly acquired clients.
        eligible = segments.set_index('CustomerID')['CustomerTenureDays'].ge(2 * days)
        output['ComparableHistory'] = eligible.reindex(output.index, fill_value=False).astype(bool)
        output['DeclineFlag'] = (output['ComparableHistory'] & (output['Previous'] > 0)
                                 & (output['Current'] < 0.75 * output['Previous']))
        output['GrowthFlag'] = (output['ComparableHistory'] & (output['Previous'] > 0)
                                & (output['Current'] > 1.25 * output['Previous']))
        output['ChangePct'] = np.where(output['Previous'] > 0,
                                       100 * (output['Current'] / output['Previous'] - 1), np.nan)
        return output

    momentum = comparison(90).rename(columns={
        'Current':'Latest90Spend','Previous':'Previous90Spend',
        'DeclineFlag':'Declining90','GrowthFlag':'Growing90','ChangePct':'Change90Pct',
        'ComparableHistory':'ComparableHistory90'})

    all_customers = segments.merge(momentum, on='CustomerID', how='left')
    all_customers['ShortHistory'] = all_customers['CustomerTenureDays'] < 180
    # Report Section 4.2: >=4 distinct purchase days; overdue at >=2 x MEDIAN gap.
    # A sparse history remains unassessed (rather than being called non-overdue).
    all_customers['RhythmEligible'] = (all_customers['PurchaseDaysObserved'] >= 4) & \
                                      all_customers['MedianInterpurchaseDays'].gt(0)
    all_customers['OverdueRatio'] = (all_customers['RecencyDays'] /
                                    all_customers['MedianInterpurchaseDays'].where(all_customers['RhythmEligible']))
    all_customers['OverdueFlag'] = all_customers['RhythmEligible'] & (all_customers['OverdueRatio'] >= 2.0)
    all_customers['ReviewFlag'] = (all_customers['CustomerSegment'] == 'Loyal Customers') & \
                                  all_customers['Declining90'].fillna(False) & all_customers['OverdueFlag'].fillna(False)

    # These categories partition the descriptive "Growing" cluster exactly.
    # They identify observed behavior; "Growing" itself is not a growth trajectory.
    all_customers['ObservedMomentumState'] = 'Not assessed'
    full = ~all_customers['ShortHistory']
    all_customers.loc[full & all_customers['Previous90Spend'].eq(0) &
                      all_customers['Latest90Spend'].gt(0), 'ObservedMomentumState'] = 'Reactivated'
    all_customers.loc[full & all_customers['Growing90'].fillna(False),
                      'ObservedMomentumState'] = 'Increasing'
    all_customers.loc[full & all_customers['Declining90'].fillna(False),
                      'ObservedMomentumState'] = 'Declining or stopped'
    all_customers.loc[full & all_customers['Previous90Spend'].gt(0) &
                      ~all_customers['Growing90'].fillna(False) &
                      ~all_customers['Declining90'].fillna(False),
                      'ObservedMomentumState'] = 'Stable'
    all_customers.to_csv(OUTPUT / '05_customer_dynamics.csv', index=False)

    # Report-facing, reproducible metrics (the 180-day and 4-purchase-day floors
    # must be shown explicitly in any discussion of review flags).
    loyal = all_customers[all_customers['CustomerSegment'] == 'Loyal Customers']
    growing = all_customers[all_customers['CustomerSegment'] == 'Growing Customers']
    summary = [
        ('LoyalAdjacent90GrowthPct', 100 * (loyal['Latest90Spend'].sum() /
                                             loyal['Previous90Spend'].sum() - 1)),
        ('LoyalDeclining90', int(loyal['Declining90'].fillna(False).sum())),
        ('LoyalDecliningAndOverdue', int(loyal['ReviewFlag'].sum())),
        ('LoyalEligibleMedianGapDays', float(loyal.loc[loyal['RhythmEligible'],
                                                 'MedianInterpurchaseDays'].median())),
        ('GrowingEligibleMedianGapDays', float(growing.loc[growing['RhythmEligible'],
                                                   'MedianInterpurchaseDays'].median())),
        ('GrowingEligibleGapCustomers', int(growing['RhythmEligible'].sum())),
    ]
    growing_distribution = (growing['ObservedMomentumState'].value_counts()
                            .reindex(['Not assessed', 'Reactivated', 'Increasing',
                                      'Stable', 'Declining or stopped'],fill_value=0))
    for category, count in growing_distribution.items():
        summary.append(('Growing_' + category.replace(' ', '_'), int(count)))
    pd.DataFrame(summary, columns=['Metric','Value']).to_csv(
        OUTPUT / '05_dynamics_report_metrics.csv', index=False)



    # Reproducible year-matched comparison for Figure 9.
    # Cohort definition is explicit and may DIFFER from the archived 868-customer
    # sample, whose original matching code was not supplied. We only include Loyal
    # customers whose first purchase predates the START of the prior-year window;
    # both full 90-day windows then fall in the observable transaction history.
    prior_end = end - pd.DateOffset(years=1)
    prior_start = prior_end - pd.Timedelta(days=90)
    recent_start = end - pd.Timedelta(days=90)
    seasonal_ids = loyal.loc[pd.to_datetime(loyal['FirstPurchaseDate']) < prior_start, 'CustomerID']
    recent_spend = window_spend(recent_start, end).reindex(seasonal_ids).fillna(0)
    last_year_spend = window_spend(prior_start, prior_end).reindex(seasonal_ids).fillna(0)
    latest_agg = float(recent_spend.sum())
    previous_agg = float(last_year_spend.sum())
    seasonal_pct = 100 * (latest_agg / previous_agg - 1) if previous_agg > 0 else float('nan')
    seasonal = pd.DataFrame([
        {'Comparison':'Adjacent 90-day (all Loyal)', 'Customers':len(loyal),
         'CurrentStart':str(recent_start.date()), 'CurrentEndExclusive':str(end.date()),
         'ReferenceStart':str((end-pd.Timedelta(days=180)).date()),
         'ReferenceEndExclusive':str(recent_start.date()),
         'CurrentGrossSpend':float(loyal['Latest90Spend'].sum()),
         'ReferenceGrossSpend':float(loyal['Previous90Spend'].sum()),
         'ChangePct':100 * (loyal['Latest90Spend'].sum() / loyal['Previous90Spend'].sum() - 1),
         'CohortDefinition':'All 1,117 Loyal customers; zeros included'},
        {'Comparison':'Same calendar dates, previous year (history-eligible)',
         'Customers':len(seasonal_ids),
         'CurrentStart':str(recent_start.date()), 'CurrentEndExclusive':str(end.date()),
         'ReferenceStart':str(prior_start.date()),
         'ReferenceEndExclusive':str(prior_end.date()),
         'CurrentGrossSpend':latest_agg, 'ReferenceGrossSpend':previous_agg,
         'ChangePct':seasonal_pct,
         'CohortDefinition':'Loyal customers whose first purchase predates prior-year window start; zeros included'}
    ])
    seasonal.to_csv(OUTPUT / '05_year_matched_comparison.csv', index=False)
    print('Fresh year-matched comparison:',len(seasonal_ids),'eligible Loyal customers,',
          f'{seasonal_pct:.2f}% change. Archived report: 868 / -9.6% (NOT independently reproduced).')

    sensitivity = []
    for days in [60, 90, 120]:
        flags = comparison(days)
        joined = segments[['CustomerID','CustomerSegment']].merge(flags[['DeclineFlag']], on='CustomerID', how='left')
        for segment, sub in joined.groupby('CustomerSegment'):
            sensitivity.append({'WindowDays':days,'CustomerSegment':segment,
                                'DecliningCustomers':int(sub['DeclineFlag'].fillna(False).sum())})
    pd.DataFrame(sensitivity).to_csv(OUTPUT / '05_momentum_window_sensitivity.csv', index=False)

    print(all_customers.groupby('CustomerSegment')[['Declining90','ShortHistory','OverdueFlag']].sum().to_string())


# ============================================================================
# Step 06 (former 06_recommendation_backtest.py)
# ============================================================================
def step_06_recommendation_backtest():
    """Step 6: next-basket hit rates, familiar items and new-item discovery."""
    import json
    from collections import defaultdict
    import numpy as np
    import pandas as pd
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler

    purchases = load_table('01_merchandise_purchases.csv')
    returns = load_table('01_recognized_returns.csv')
    results = []
    detail_rows = []
    interval_rows = []
    segment_type_rows = []
    overall_recall_rows = []


    def groups_as_of(history, decision_date):
        features = build_customer_features(history, returns, decision_date)
        good = features[features['MonetaryNet'] > 0].copy()
        if len(good) < 4:
            return features[['CustomerID']].assign(Group='Unclassified')
        X = np.log1p(good[['RecencyDays','FrequencyInvoices','MonetaryNet']].clip(lower=0))
        good['Group'] = KMeans(n_clusters=4, random_state=RANDOM_STATE, n_init=20).fit_predict(StandardScaler().fit_transform(X))
        profile = good.groupby('Group').agg(MedianNet=('MonetaryNet', 'median'),
                                            MedianRecency=('RecencyDays', 'median'))
        loyal = profile['MedianNet'].idxmax()
        inactive = profile['MedianRecency'].idxmax()
        remaining = [x for x in profile.index if x not in (loyal, inactive)]
        growing = profile.loc[remaining, 'MedianRecency'].idxmin()
        cooling = next(x for x in remaining if x != growing)
        names = {loyal: 'Loyal Customers', cooling: 'Cooling Customers',
                 growing: 'Growing Customers', inactive: 'Inactive Customers'}
        good['CustomerSegment'] = good['Group'].map(names)
        return good[['CustomerID', 'Group', 'CustomerSegment']]


    for cutoff_str in ['2011-06-10', '2011-09-10']:
        cutoff = pd.Timestamp(cutoff_str)
        next_day = cutoff + pd.Timedelta(days=1)
        history = purchases[purchases['InvoiceDate'] < next_day].copy()
        future = purchases[(purchases['InvoiceDate'] >= next_day) &
                           (purchases['InvoiceDate'] < next_day + pd.Timedelta(days=90)) &
                           (purchases['CustomerID'].notna())].copy()
        history_known = history[history['CustomerID'].notna()].copy()

        labels = groups_as_of(history, cutoff_str)
        history = history.merge(labels[['CustomerID','Group']], on='CustomerID', how='left')
        market_rules = fp_rules(history, support=.01)
        segment_rules = {}
        for cluster in sorted(labels['Group'].dropna().unique()):
            segment_rules[cluster] = fp_rules(history[history['Group']==cluster], support=.015, min_count=50)

        last_order = history_known.groupby(['CustomerID','Invoice']).agg(
            Date=('InvoiceDate','min'), Basket=('StockCode',lambda x:set(x))
        ).reset_index().sort_values(['CustomerID','Date','Invoice']).groupby('CustomerID').tail(1).set_index('CustomerID')
        first_future = future.groupby(['CustomerID','Invoice']).agg(
            Date=('InvoiceDate','min'), Basket=('StockCode',lambda x:set(x))
        ).reset_index().sort_values(['CustomerID','Date','Invoice']).groupby('CustomerID').head(1).set_index('CustomerID')
        personal_counts = (history_known.groupby(['CustomerID','StockCode'])['Invoice']
                           .nunique().rename('Frequency').reset_index()
                           .sort_values(['CustomerID','Frequency','StockCode'],ascending=[True,False,True]))
        personal_map = personal_counts.groupby('CustomerID')['StockCode'].apply(list).to_dict()
        seen = history_known.groupby('CustomerID')['StockCode'].agg(set).to_dict()
        popular = history.groupby('StockCode')['Invoice'].nunique().sort_values(ascending=False).index.tolist()
        label_map = labels.set_index('CustomerID')['Group'].to_dict()
        name_map = labels.set_index('CustomerID')['CustomerSegment'].to_dict()

        counts = {key:{'hits':0,'eligible':0,'covered':0,'matched_products':0,'suggestions':0}
                  for key in ['Market popularity','Overall rules','Segment rules + fallback','Personal frequency']}
        match_vectors = {name:[] for name in counts}
        family = {'Familiar':{'suggestions':0,'hits':0},'Customer-new':{'suggestions':0,'hits':0}}
        target_recalls = []
        target_type_recalls = defaultdict(list)
        segment_type = defaultdict(lambda: {'Suggestions':0, 'MatchedSuggestions':0, 'TargetRecalls':[], 'EligibleCustomers':0})

        for customer_id in set(last_order.index) & set(first_future.index):
            last_basket = last_order.loc[customer_id,'Basket']
            target_basket = first_future.loc[customer_id,'Basket']
            personal = [item for item in personal_map.get(customer_id, []) if item not in last_basket][:3]
            overall = recommend_from_rules(last_basket, market_rules)
            cluster = label_map.get(customer_id)
            # One identical fallback policy for historical tests and live scoring.
            combined, _ = recommend_with_fallback(last_basket, segment_rules.get(cluster), market_rules)
            popular_rec = [item for item in popular if item not in last_basket][:3]
            proposals = {'Market popularity':popular_rec,'Overall rules':overall,
                         'Segment rules + fallback':combined,'Personal frequency':personal}

            for method, suggestions in proposals.items():
                stat = counts[method]
                stat['eligible'] += 1
                stat['covered'] += int(bool(suggestions))
                matched = set(suggestions) & target_basket
                stat['hits'] += int(bool(matched))
                stat['matched_products'] += len(matched)
                stat['suggestions'] += len(suggestions)
                match_vectors[method].append(int(bool(matched)))

            segment_name = name_map.get(customer_id, 'Unclassified')
            historic_codes = seen.get(customer_id, set())
            for item in combined:
                category = 'Familiar' if item in historic_codes else 'Customer-new'
                family[category]['suggestions'] += 1
                family[category]['hits'] += int(item in target_basket)
                segment_type[(segment_name, category)]['Suggestions'] += 1
                segment_type[(segment_name, category)]['MatchedSuggestions'] += int(item in target_basket)
            for category, target_items in [('Familiar', target_basket & historic_codes),
                                           ('Customer-new', target_basket - historic_codes)]:
                if target_items:
                    type_predictions = {v for v in combined
                                        if (v in historic_codes) == (category == 'Familiar')}
                    recall = len(type_predictions & target_items) / len(target_items)
                    target_type_recalls[category].append(recall)
                    segment_type[(segment_name, category)]['TargetRecalls'].append(recall)
            target_recalls.append(len(set(combined) & target_basket) / len(target_basket))

        for method, stat in counts.items():
            n = stat['eligible']
            results.append({'Cutoff':cutoff_str,'Method':method, 'EligibleReturningCustomers':n,
                            'Coverage':stat['covered']/n if n else np.nan,
                            'HitRate':stat['hits']/n if n else np.nan,
                            'ItemPrecision':stat['matched_products']/stat['suggestions'] if stat['suggestions'] else np.nan,
                            'NumberSuggestions':stat['suggestions']})
        for category, stat in family.items():
            detail_rows.append({'Cutoff':cutoff_str,'Category':category,'Suggestions':stat['suggestions'],
                                'MatchedSuggestions':stat['hits'],
                                'ItemPrecision':stat['hits']/stat['suggestions'] if stat['suggestions'] else np.nan,
                                'MeanTargetRecall':np.mean(target_type_recalls[category])
                                                    if target_type_recalls[category] else np.nan,
                                'CustomersWithTargetType':len(target_type_recalls[category]),
                                'MeanOverallTargetRecall':np.mean(target_recalls)})
        overall_recall_rows.append({'Cutoff':cutoff_str,
                                    'MeanOverallTargetRecall':np.mean(target_recalls),
                                    'EligibleReturningCustomers':len(target_recalls)})
        for (name, category), values in sorted(segment_type.items()):
            segment_type_rows.append({'Cutoff':cutoff_str, 'CustomerSegment':name,
                                      'Category':category,
                                      'Suggestions':values['Suggestions'],
                                      'MatchedSuggestions':values['MatchedSuggestions'],
                                      'ItemPrecision':values['MatchedSuggestions']/values['Suggestions']
                                                        if values['Suggestions'] else np.nan,
                                      'CustomersWithTargetType':len(values['TargetRecalls']),
                                      'MeanTargetRecall':np.mean(values['TargetRecalls'])
                                                         if values['TargetRecalls'] else np.nan})

        diff = np.array(match_vectors['Segment rules + fallback']) - np.array(match_vectors['Overall rules'])
        rng = np.random.default_rng(RANDOM_STATE)
        bootstrap = [rng.choice(diff,size=len(diff),replace=True).mean() for _ in range(500)]
        interval_rows.append({'Cutoff':cutoff_str,'MeanDifferenceSegmentMinusOverall':diff.mean(),
                              'CI95_Lower':np.quantile(bootstrap,.025),'CI95_Upper':np.quantile(bootstrap,.975)})
        print(f'{cutoff_str}: evaluated {counts["Overall rules"]["eligible"]:,} next baskets')

    pd.DataFrame(results).to_csv(OUTPUT/'06_next_basket_backtest.csv',index=False)
    pd.DataFrame(detail_rows).to_csv(OUTPUT/'06_familiar_new_items.csv',index=False)
    pd.DataFrame(interval_rows).to_csv(OUTPUT/'06_paired_bootstrap.csv',index=False)
    pd.DataFrame(segment_type_rows).to_csv(OUTPUT/'06_familiar_new_by_segment.csv',index=False)
    pd.DataFrame(overall_recall_rows).to_csv(OUTPUT/'06_target_recall_summary.csv',index=False)
    with open(OUTPUT/'06_backtest_methodology.json','w',encoding='utf-8') as handle:
        json.dump({'cutoffs':['2011-06-10','2011-09-10'], 'forecast_horizon_days':90,
                   'next_basket_definition':'First merchandise invoice for an observed returning customer',
                   'max_suggestions_per_customer':3,
                   'segment_fallback':'Overall rules used only when NO segment rule triggers',
                   'no_suggestion_counts_as_miss':True,
                   'paired_bootstrap_draws':500,
                   'label':'Any recommended item present in first future basket',
                   'historic_metrics_are_fresh_recomputations':True},handle,indent=2)

    print(pd.DataFrame(results).to_string(index=False))


# ============================================================================
# Step 07 (former 07_repeat_purchase_prediction.py)
# ============================================================================
def step_07_repeat_purchase_prediction():
    """Step 7: compare classifiers using past-to-future 90-day purchase targets."""
    import json
    import sklearn
    import xgboost
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_fscore_support, brier_score_loss
    from xgboost import XGBClassifier

    purchases = load_table('01_identified_purchases.csv')
    returns = load_table('01_recognized_returns.csv')

    FEATURES = [
        'RecencyDays','FrequencyInvoices','MonetaryNet','GrossMerchandiseSpend',
        'AvgBasketValue','AvgBasketUnits','TotalMerchandiseUnits','ProductDiversity',
        'CustomerTenureDays','ActiveSpanDays','Recent30GrossSpend','Previous30GrossSpend',
        'Recent90GrossSpend','Previous90GrossSpend','SpendMomentum',
        'PurchaseInvoices30','PurchaseInvoices90','ReturnInvoiceCount','ReturnValue',
        'ReturnValueRatio','ReturnInvoiceShare','IsUK','MonthSin','MonthCos'
    ]


    def make_snapshot(as_of, labelled=True):
        as_of_date = pd.Timestamp(as_of)
        current = build_customer_features(purchases, returns, as_of_date)
        current['AsOfDate'] = as_of_date
        if labelled:
            next_day = as_of_date + pd.Timedelta(days=1)
            future = purchases[(purchases['InvoiceDate'] >= next_day) &
                               (purchases['InvoiceDate'] < next_day + pd.Timedelta(days=90))]
            repeating = set(future['CustomerID'].dropna())
            current['PurchaseNext90D'] = current['CustomerID'].isin(repeating).astype(int)
        return current


    def prepared_features(table):
        X = table[FEATURES].fillna(0).replace([np.inf, -np.inf], 0).astype(float).copy()
        for name in FEATURES:
            if name not in ['MonthSin','MonthCos','IsUK','ReturnInvoiceShare','ReturnValueRatio']:
                x = X[name].to_numpy()
                X[name] = np.sign(x) * np.log1p(np.abs(x))
        return X


    def evaluate(model, table, name):
        y = table['PurchaseNext90D'].to_numpy()
        score = model.predict_proba(prepared_features(table))[:,1]
        threshold = 0.50
        pred = (score >= threshold).astype(int)
        precision, recall, f1, _ = precision_recall_fscore_support(y, pred, average='binary',zero_division=0)
        metrics = {'Model':name,'Customers':len(y),'RepeatRate':float(y.mean()),
                   'ROC_AUC':roc_auc_score(y, score),'PR_AUC':average_precision_score(y, score),
                   'PrecisionAt05':precision,'RecallAt05':recall,'F1At05':f1,
                   'Brier':brier_score_loss(y, score)}
        metrics.update(score_top20(y,score))
        return metrics, score

    snapshots = []
    for year in [2010,2011]:
        for month in range(1,13):
            day = pd.Timestamp(year=year,month=month,day=10)
            if pd.Timestamp('2010-06-10') <= day <= pd.Timestamp('2011-09-10'):
                snapshots.append(make_snapshot(day))
                print('Snapshot:',day.date(),len(snapshots[-1]))

    snapshots_by_date = {pd.Timestamp(frame['AsOfDate'].iloc[0]).strftime('%Y-%m-%d'):frame
                         for frame in snapshots}
    train_dates = [d for d in snapshots_by_date if d <= '2011-03-10']
    train = pd.concat([snapshots_by_date[d] for d in train_dates], ignore_index=True)
    validation = snapshots_by_date['2011-06-10']
    test = snapshots_by_date['2011-09-10']

    models = {
        'Logistic Regression':make_pipeline(StandardScaler(),LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)),
        'Random Forest':RandomForestClassifier(n_estimators=160,max_depth=10,min_samples_leaf=10,
                                               random_state=RANDOM_STATE,n_jobs=-1),
        'XGBoost':XGBClassifier(n_estimators=220,max_depth=3,learning_rate=.05,
                                subsample=1,colsample_bytree=1,objective='binary:logistic',
                                eval_metric='logloss',random_state=RANDOM_STATE,n_jobs=4)
    }
    validation_rows = []
    for name, model in models.items():
        model.fit(prepared_features(train), train['PurchaseNext90D'])
        scores, _ = evaluate(model, validation, name)
        validation_rows.append(scores)
    validation_table = pd.DataFrame(validation_rows).sort_values('PR_AUC',ascending=False)
    validation_table.to_csv(OUTPUT / '07_validation_comparison.csv',index=False)

    # Keep the June-selected specification fixed for the untouched September test.
    selected_name = validation_table.iloc[0]['Model']
    selected_model = models[selected_name]
    test_result, test_scores = evaluate(selected_model, test, selected_name)
    pd.DataFrame([test_result]).to_csv(OUTPUT / '07_out_of_time_metrics.csv', index=False)

    ranked = test[['CustomerID','AsOfDate','PurchaseNext90D']].copy()
    ranked['Score'] = test_scores
    ranked = ranked.sort_values('Score',ascending=False).reset_index(drop=True)
    ranked['Decile'] = np.clip(np.ceil((ranked.index + 1) / len(ranked) * 10).astype(int),1,10)
    # Keep row-level scores so published AUC/lift can be independently audited.
    ranked.to_csv(OUTPUT / '07_test_customer_predictions.csv', index=False)
    decile = ranked.groupby('Decile').agg(Customers=('CustomerID','size'),
                                          ActualRepeatRate=('PurchaseNext90D','mean'))
    decile.to_csv(OUTPUT / '07_test_decile_rates.csv')
    fig, ax = plt.subplots(figsize=(8,4.6))
    ax.bar(decile.index.astype(str),decile['ActualRepeatRate']*100)
    ax.set_title('Out-of-time repeat purchase rate by predicted decile')
    ax.set_xlabel('Predicted propensity decile (1 = highest)')
    ax.set_ylabel('Actual repeat rate (%)')
    fig.tight_layout()
    fig.savefig(OUTPUT / '07_out_of_time_deciles.png',dpi=170)
    plt.close(fig)

    # Refit on all labelled snapshots only after the out-of-time evaluation.
    all_labelled = pd.concat(snapshots,ignore_index=True)
    all_labelled.to_csv(OUTPUT/'07_labelled_snapshots.csv',index=False)
    selected_model.fit(prepared_features(all_labelled),all_labelled['PurchaseNext90D'])
    future = make_snapshot('2011-12-09',labelled=False)
    future['PredictedRepeatProbability90D'] = selected_model.predict_proba(prepared_features(future))[:,1]
    future = future.sort_values('PredictedRepeatProbability90D',ascending=False).reset_index(drop=True)
    future['ProbabilityRank'] = future.index + 1
    future['PriorityBand'] = pd.cut(future.index / len(future),
                                   bins=[-0.001,.10,.20,.50,1.01],
                                   labels=['Top 10%','Top 20%','Top 50%','Lower 50%'])
    future.to_csv(OUTPUT / '07_future_scores.csv',index=False)

    # ARCHIVED report scores cannot be inferred from source code alone. Preserve
    # model specification and temporal boundaries, and report fresh test metrics.
    model_metadata = {
        'model_selection_metric': 'June 2011 validation PR_AUC',
        'selected_model': selected_name,
        'training_snapshots': sorted(train_dates),
        'validation_snapshot': '2011-06-10',
        'out_of_time_snapshot': '2011-09-10',
        'production_as_of': '2011-12-09',
        'forecast_horizon_days': 90,
        'positive_label': 'At least one valid identified merchandise purchase in next 90 days',
        'classification_threshold_for_F1': 0.5,
        'feature_names': FEATURES,
        'feature_transform': 'Signed log1p for continuous features; original binary/cyclical/share features',
        'seed': RANDOM_STATE,
        'model_parameters': selected_model.get_params(),
        'scikit_learn_version': sklearn.__version__,
        'xgboost_version': xgboost.__version__,
        'holdout_test_used_for_selection': False,
        'production_model_refitted_after_test_on_observed_labelled_snapshots': True,
        'numerical_calibration_validated': False,
        'archived_report_metrics_not_guaranteed_to_match': True
    }
    with open(OUTPUT / '07_model_metadata.json', 'w', encoding='utf-8') as handle:
        json.dump(model_metadata, handle, indent=2, default=str)

    try:
        import shap
        sample = prepared_features(future).sample(n=min(300,len(future)),random_state=RANDOM_STATE)
        explainer = shap.TreeExplainer(selected_model) if selected_name == 'XGBoost' else None
        if explainer is not None:
            contributions = explainer.shap_values(sample)
            shap_table = pd.DataFrame({'Feature':sample.columns,'MeanAbsoluteSHAP':np.abs(contributions).mean(axis=0)})
            shap_table.sort_values('MeanAbsoluteSHAP',ascending=False).to_csv(OUTPUT/'07_shap_importance.csv',index=False)
    except (ImportError, ValueError) as error:
        print('SHAP is optional:',error)

    print('\nValidation model comparison:')
    print(validation_table[['Model','ROC_AUC','PR_AUC','LiftAt20']].to_string(index=False))
    print('\nOut-of-time test:')
    print(pd.DataFrame([test_result]).to_string(index=False))
    print('\nProduction scores:',len(future),'customers. Their future outcomes are NOT observed.')


# ============================================================================
# Step 08 (former 08_customer_actions.py)
# ============================================================================
def step_08_customer_actions():
    """Step 8: integrate report-consistent recommendations and customer decisions."""
    import pandas as pd

    customers = load_customers('05_customer_dynamics.csv')
    scores = load_customers('07_future_scores.csv')[
        ['CustomerID', 'PredictedRepeatProbability90D', 'ProbabilityRank', 'PriorityBand']]
    rules = pd.read_csv(OUTPUT / '04_association_rules.csv')
    purchases = load_table('01_identified_purchases.csv')

    # Cache rule sets and segment labels once; searching the full dataframes for
    # each customer would be prohibitively slow and can obscure the fallback logic.
    segment_by_customer = customers.set_index('CustomerID')['CustomerSegment'].to_dict()
    rule_sets = {
        group: subset.sort_values(['Confidence', 'Lift', 'Support', 'AntecedentCodes', 'ConsequentCode'],
                                  ascending=[False, False, False, True, True], kind='mergesort')
        for group, subset in rules.groupby('Group', sort=False)
    }
    market_rules = rule_sets['All Customers']

    invoice_baskets = purchases.groupby(['CustomerID', 'Invoice'], observed=True).agg(
        InvoiceDate=('InvoiceDate', 'min'),
        Codes=('StockCode', lambda codes: set(codes.dropna()))
    ).reset_index()
    last_baskets = (invoice_baskets
                    .sort_values(['CustomerID', 'InvoiceDate', 'Invoice'])
                    .groupby('CustomerID').tail(1))

    recommendations = []
    for row in last_baskets.itertuples(index=False):
        segment = segment_by_customer.get(row.CustomerID)
        if segment is None:
            continue
        own_rules = rule_sets.get(segment)
        # Shared with Step 06: never pad a partially filled segment list.
        final, source = recommend_with_fallback(row.Codes, own_rules, market_rules)
        recommendations.append({
            'CustomerID': row.CustomerID, 'LastInvoice': row.Invoice,
            'RecommendedProductCodes': '|'.join(final),
            'RecommendationSource': source
        })

    output = (customers.merge(scores, on='CustomerID', how='left')
              .merge(pd.DataFrame(recommendations), on='CustomerID', how='left'))


    def assign_action(row):
        segment = row['CustomerSegment']
        band = str(row['PriorityBand'])
        if 'Exception' in segment:
            return 'Manual review of zero/negative net value'
        if segment == 'Loyal Customers':
            return ('Review declining activity before discounts'
                    if pd.notna(row.get('Declining90')) and bool(row.get('Declining90'))
                    else 'Protect relationship and monitor activity')
        if segment == 'Growing Customers':
            return 'Nurture recent relationship with relevant follow-up'
        if segment == 'Cooling Customers':
            return ('Test selective win-back contact' if band != 'Lower 50%'
                    else 'Use selective low-cost re-engagement')
        return 'Lower-cost outreach / lower priority for paid contact'


    output['SuggestedAction'] = output.apply(assign_action, axis=1)
    output.to_csv(OUTPUT / '08_customer_intelligence.csv', index=False)

    # Composition uses each customer's complete observed purchase history, not
    # just the latest trigger basket. This does NOT imply future purchase impact.
    previous = purchases.groupby('CustomerID')['StockCode'].agg(set).to_dict()
    familiar = customer_new = 0
    for row in output[['CustomerID', 'RecommendedProductCodes']].itertuples(index=False):
        codes = row.RecommendedProductCodes
        if pd.isna(codes) or not codes:
            continue
        seen = previous.get(row.CustomerID, set())
        for code in codes.split('|'):
            if code in seen:
                familiar += 1
            else:
                customer_new += 1

    summary = pd.DataFrame([
        {'Metric':'IdentifiedCustomers', 'Value':len(output)},
        {'Metric':'CoveredCustomers', 'Value':int(output['RecommendedProductCodes'].fillna('').ne('').sum())},
        {'Metric':'TotalSuggestions', 'Value':familiar + customer_new},
        {'Metric':'FamiliarSuggestions', 'Value':familiar},
        {'Metric':'CustomerNewSuggestions', 'Value':customer_new},
        {'Metric':'MeanDecemberRepeatScorePct', 'Value':100*output['PredictedRepeatProbability90D'].mean()},
    ])
    summary.to_csv(OUTPUT / '08_recommendation_report_metrics.csv', index=False)
    print(summary.to_string(index=False))


# ============================================================================
# Step 09 (former 09_report_checks.py)
# ============================================================================
def step_09_report_checks():
    """Step 9: check every number printed in the report text and tables against the code outputs.

Benchmarks are the values in the revised report (Group-4 Online Retail report,
November 2026 revision). Figures are not listed here because Step 12 draws all of
them from the same outputs.

PASS: recomputed value matches the report to its published precision.
FAIL: recomputed value differs from the report.
UNVERIFIED: the output file needed for the check is missing (an earlier step did not run).

Run after Steps 01-08 and 11. Use --strict to exit with an error on any FAIL.
"""
    import json
    import math
    from pathlib import Path
    import pandas as pd

    rows = []


    def read_csv(name):
        path = OUTPUT / name
        return pd.read_csv(path) if path.exists() else None


    def check(section, metric, benchmark, actual=None, tolerance=0.0,
              unit='', source='', comment=''):
        """Record one independently interpretable check, preserving missingness."""
        if actual is None or pd.isna(actual):
            status = 'UNVERIFIED'
            difference = None
        else:
            actual = float(actual)
            difference = actual - benchmark
            status = 'PASS' if math.isclose(actual, benchmark, abs_tol=tolerance,
                                             rel_tol=0.0) else 'FAIL'
        rows.append({'Section':section, 'Metric':metric, 'ReportBenchmark':benchmark,
                     'Recomputed':actual, 'Difference':difference, 'Tolerance':tolerance,
                     'Unit':unit, 'Status':status, 'OutputSource':source,
                     'Notes':comment})


    def get(table, key_col, key, value_col):
        if table is None or any(column not in table for column in (key_col, value_col)):
            return None
        values = table.loc[table[key_col].astype(str).eq(str(key)), value_col]
        return float(values.iloc[0]) if len(values) else None


    def row_value(table, column):
        return (float(table.iloc[0][column]) if table is not None and not table.empty
                and column in table else None)


    def group_metric(table, name, column):
        return get(table, 'CustomerSegment', name, column)


    # Section 2 and Figure 2: line counts after each cleaning stage (merchandise excludes gift vouchers).
    audit = read_csv('01_data_audit.csv')
    for key, reported in [('Raw rows', 1067371), ('After sheet-overlap rule', 1044848),
                          ('Valid positive purchase lines', 1019654),
                          ('Valid merchandise purchase lines', 1015009),
                          ('Duplicate purchase lines removed', 11731),
                          ('Cancelled purchase lines removed', 6022),
                          ('Purchase lines after duplicates and cancellations', 997256),
                          ('Identified merchandise purchase lines', 770495),
                          ('Customers with purchase history', 5838),
                          ('Merchandise purchase baskets', 39258)]:
        check('2. Data preparation', key, reported, get(audit,'Stage',key,'Count'),
              source='01_data_audit.csv')

    # Section 2: the largest cancelled order quoted in the text.
    pairs = read_csv('01_cancelled_purchase_pairs.csv')
    top = pairs.sort_values('LineRevenue', ascending=False).iloc[0] if pairs is not None and len(pairs) else None
    check('2. Data preparation','Largest cancelled order value (GBP)',168469.60,
          top.LineRevenue if top is not None else None,.005,source='01_cancelled_purchase_pairs.csv')
    check('2. Data preparation','Largest cancelled order units',80995,
          top.Quantity if top is not None else None,source='01_cancelled_purchase_pairs.csv')

    features = read_csv('02_customer_features.csv')
    if features is not None and 'MonetaryNet' in features:
        check('2. Data preparation','Positive net customers',5831,
              int(features.MonetaryNet.gt(0).sum()),source='02_customer_features.csv')
        check('2. Data preparation','Nonpositive net exceptions',7,
              int(features.MonetaryNet.le(0).sum()),source='02_customer_features.csv')
    else:
        for metric,n in [('Positive net customers',5831),('Nonpositive net exceptions',7)]:
            check('2. Data preparation',metric,n,source='02_customer_features.csv')

    monthly = read_csv('02_monthly_revenue.csv')
    for month, amount in [('2010-11',1410000),('2011-11',1440000)]:
        check('3. Descriptive', f'Gross revenue {month}', amount,
              get(monthly,'Month',month,'GrossMerchandiseRevenue'), tolerance=10000,
              unit='GBP',source='02_monthly_revenue.csv',
              comment='Approximate report text (~£1.41m/~£1.44m)')
    concentration = read_csv('02_customer_concentration.csv')
    for pct, amount in [(1,31.2),(20,76.7)]:
        val = get(concentration,'TopCustomerPct',pct,'NetValueShare')
        check('3. Descriptive', f'Top {pct}% net value share',amount,
              val * 100 if val is not None else None, tolerance=.05,unit='percentage points',
              source='02_customer_concentration.csv')

    k_table = read_csv('03_k_selection.csv')
    for k,col,ref in [(2,'Silhouette',.439),(4,'Silhouette',.364),
                       (4,'StabilityARI',.995)]:
        subset = k_table[k_table.K.eq(k)] if k_table is not None else None
        value = row_value(subset,col)
        check('4. Segmentation',f'K={k} {col}',ref,value,tolerance=.0005,
              source='03_k_selection.csv',comment='Different K-means fit/subsample settings can change indices')
    profiles = read_csv('03_segment_profiles.csv')
    for segment, count, days, freq, monetary, share in [
        ('Loyal Customers',1157,16,13,4922,73.1),
        ('Cooling Customers',1467,177,5,1451,16.9),
        ('Growing Customers',1234,23,3,712,6.3),
        ('Inactive Customers',1973,400,1,268,3.8)]:
        prefix = segment.split()[0]
        for column, benchmark, tolerance in [('Customers',count,0),
                                             ('MedianRecency',days,.5),
                                             ('MedianFrequency',freq,0),
                                             ('MedianNetValue',monetary,1),
                                             ('NetValueSharePct',share,.05)]:
            check('4. Segmentation',f'{prefix} {column}',benchmark,
                  group_metric(profiles,segment,column),tolerance,
                  source='03_segment_profiles.csv')
    check('4. Segmentation','Exception customers',7,
          group_metric(profiles,'Net-Zero / Net-Negative Exception','Customers'),
          source='03_segment_profiles.csv')

    dynamics = read_csv('05_dynamics_report_metrics.csv')
    sensitivity = read_csv('05_momentum_window_sensitivity.csv')
    for days, benchmark in [(60,287),(90,209),(120,202)]:
        subset = (sensitivity[sensitivity.WindowDays.eq(days)] if sensitivity is not None else None)
        check('8. Sensitivity',f'Loyal decline count {days}-day window',benchmark,
              get(subset,'CustomerSegment','Loyal Customers','DecliningCustomers'),
              source='05_momentum_window_sensitivity.csv')
    for name,ref,tol in [('LoyalAdjacent90GrowthPct',66.0,.05),
                          ('LoyalDeclining90',209,0),
                          ('LoyalDecliningAndOverdue',47,0),
                          ('LoyalEligibleMedianGapDays',38,0),
                          ('GrowingEligibleMedianGapDays',72,0),
                          ('GrowingEligibleGapCustomers',391,0)]:
        check('4.2-4.3 Dynamics',name,ref,get(dynamics,'Metric',name,'Value'),tol,
              source='05_dynamics_report_metrics.csv')
    for metric, benchmark, key in [
        ('Growing short-history customers',568,'Growing_Not_assessed'),
        ('Growing reactivated customers',491,'Growing_Reactivated'),
        ('Growing materially increasing customers',82,'Growing_Increasing'),
        ('Growing stable customers',53,'Growing_Stable'),
        ('Growing declining or stopped customers',40,'Growing_Declining_or_stopped')]:
        check('4.3 Dynamics',metric,benchmark,get(dynamics,'Metric',key,'Value'),
              source='05_dynamics_report_metrics.csv')
    year_matched = read_csv('05_year_matched_comparison.csv')
    ym = (year_matched[year_matched.Comparison.str.startswith('Same calendar')]
          if year_matched is not None else None)
    check('4.3 Dynamics','Loyal history-eligible year-matched customers',967,
          row_value(ym,'Customers'),source='05_year_matched_comparison.csv')
    check('4.3 Dynamics','Loyal year-matched change (%)',-6.4,row_value(ym,'ChangePct'),.05,
          source='05_year_matched_comparison.csv')

    rules = read_csv('04_association_rules.csv')
    for label, count, basket in [('All Customers',857,39258),('Loyal Customers',282,22376),
                                 ('Cooling Customers',94,7465),('Growing Customers',71,3758),
                                 ('Inactive Customers',3,2729)]:
        subset = rules[rules.Group.eq(label)] if rules is not None else None
        actual_rules = len(subset) if subset is not None else None
        actual_baskets = (float(subset.BasketCount.iloc[0]) if subset is not None
                          and len(subset) else None)
        check('5. Product affinity', f'{label} rule count', count,actual_rules,
              source='04_association_rules.csv')
        check('5. Product affinity', f'{label} basket count',basket,actual_baskets,
              source='04_association_rules.csv')

    recommendations = read_csv('08_recommendation_report_metrics.csv')
    for metric,benchmark in [('CoveredCustomers',4547),('TotalSuggestions',10058),
                             ('FamiliarSuggestions',2342),('CustomerNewSuggestions',7716)]:
        check('5.1 December recommendations',metric,benchmark,
              get(recommendations,'Metric',metric,'Value'),
              source='08_recommendation_report_metrics.csv')

    backtests = read_csv('06_next_basket_backtest.csv')
    backtest_benchmarks = {
        '2011-06-10': [('Market popularity',17.8), ('Overall rules',24.6),
                       ('Segment rules + fallback',21.9),('Personal frequency',38.1)],
        '2011-09-10':[('Market popularity',15.8), ('Overall rules',21.2),
                      ('Segment rules + fallback',19.6),('Personal frequency',34.2)]
    }
    for cutoff, entries in backtest_benchmarks.items():
        subset = (backtests[backtests.Cutoff.eq(cutoff)] if backtests is not None else None)
        for method,ref in entries:
            raw = get(subset,'Method',method,'HitRate')
            check('5.2 Next basket',f'{cutoff} {method} hit rate',ref,
                  100*raw if raw is not None else None,tolerance=.05,
                  unit='percentage points', source='06_next_basket_backtest.csv',
                  comment='Reported to one decimal place; denominator includes returners without recommendations')
    bootstrap = read_csv('06_paired_bootstrap.csv')
    for date, ref in [('2011-06-10',-2.65),('2011-09-10',-1.62)]:
        raw = get(bootstrap,'Cutoff',date,'MeanDifferenceSegmentMinusOverall')
        check('5.2 Next basket',f'{date} segment-minus-overall (pp)',ref,
              100*raw if raw is not None else None,tolerance=.005,
              unit='percentage points',source='06_paired_bootstrap.csv')

    items = read_csv('06_familiar_new_items.csv')
    sept = (items[items.Cutoff.eq('2011-09-10')] if items is not None else None)
    for kind,count,matched,precision in [('Familiar',1539,365,23.7),
                                          ('Customer-new',2714,204,7.5)]:
        raw = get(sept,'Category',kind,'ItemPrecision')
        for key,benchmark,actual,tolerance in [
            ('Suggestions',count,get(sept,'Category',kind,'Suggestions'),0),
            ('MatchedSuggestions',matched,get(sept,'Category',kind,'MatchedSuggestions'),0),
            ('ItemPrecisionPct',precision,100*raw if raw is not None else None,.05)]:
            check('5.3 Familiar vs new',f'September {kind} {key}',benchmark,actual,tolerance,
                  source='06_familiar_new_items.csv')
    recall = read_csv('06_target_recall_summary.csv')
    raw = get(recall,'Cutoff','2011-09-10','MeanOverallTargetRecall')
    check('5.3 Familiar vs new','September overall mean target recall (%)',1.24,
          100*raw if raw is not None else None,tolerance=.05,
          unit='percentage points',source='06_target_recall_summary.csv',
          comment='Report uses approximately 1.24%')

    validation = read_csv('07_validation_comparison.csv')
    for model, ref_auc, ref_pr, ref_lift in [
        ('Logistic Regression',.811,.710,None),
        ('Random Forest',.817,.723,2.40),
        ('XGBoost',.821,.725,2.38)]:
        for col,ref,tol in [('ROC_AUC',ref_auc,.0005),('PR_AUC',ref_pr,.0005),
                            ('LiftAt20',ref_lift,.005)]:
            if ref is not None:
                check('6. Model validation',f'{model} {col}',ref,
                      get(validation,'Model',model,col),tol,source='07_validation_comparison.csv',
                      comment='June 2011 validation snapshot')

    test = read_csv('07_out_of_time_metrics.csv')
    for column,benchmark,tolerance,scale in [
        ('Customers',5247,0,1),('RepeatRate',43.55,.005,100),
        ('ROC_AUC',.7881,.00005,1),('PR_AUC',.7673,.00005,1),
        ('F1At05',.6778,.00005,1),('Top20Rate',84.57,.005,100),
        ('LiftAt20',1.94,.01,1),('CaptureAt20',38.86,.005,100)]:
        raw=row_value(test,column)
        check('6.1 OOT prediction',column,benchmark,raw*scale if raw is not None else None,
              tolerance,source='07_out_of_time_metrics.csv',
              comment='September 2011 out-of-time snapshot')
    mean=get(recommendations,'Metric','MeanDecemberRepeatScorePct','Value')
    check('6.1 Production forecasts','December mean model score (%)',30.2,mean,.05,
          unit='percentage points',source='08_recommendation_report_metrics.csv',
          comment='Future observed outcomes unavailable; score is not a calibrated causal effect')

    integrated=read_csv('08_customer_intelligence.csv')
    check('7. Customer decision table','Final customer records',5838,
          len(integrated) if integrated is not None else None,
          source='08_customer_intelligence.csv')
    examples = (integrated.dropna(subset=['PredictedRepeatProbability90D']).copy()
                if integrated is not None else None)
    for label, segment, reported in [('A','Loyal Customers',.774),('B','Cooling Customers',.229),
                                     ('C','Growing Customers',.248),('D','Inactive Customers',.071)]:
        # Section 7.1 rule: customer closest to the within-segment median December score.
        value = None
        if examples is not None:
            sub = examples[examples.CustomerSegment.eq(segment)]
            median = sub.PredictedRepeatProbability90D.median()
            pick = sub.assign(d=(sub.PredictedRepeatProbability90D-median).abs()).sort_values(['d','CustomerID']).iloc[0]
            value = pick.PredictedRepeatProbability90D
        check('7. Customer decision table',f'Illustrative customer {label} score',reported,value,.0005,
              source='08_customer_intelligence.csv')


    # ---- Checks added for the November 2026 revision: every remaining number in the text ----
    extra = None
    extra_path = OUTPUT / '11_extra_report_metrics.json'
    if extra_path.exists():
        extra = json.loads(extra_path.read_text(encoding='utf-8'))

    # Section 4.1 table: customer shares and the exception row.
    for segment, share in [('Loyal Customers',19.8),('Cooling Customers',25.1),('Growing Customers',21.1),
                           ('Inactive Customers',33.8),('Net-Zero / Net-Negative Exception',0.1)]:
        check('4.1 Segment table',f'{segment.split()[0]} customer share (%)',share,
              group_metric(profiles,segment,'CustomerSharePct'),.05,source='03_segment_profiles.csv')
    for column, benchmark in [('MedianRecency',374),('MedianFrequency',1),('MedianNetValue',0)]:
        check('4.1 Segment table',f'Exceptions {column}',benchmark,
              group_metric(profiles,'Net-Zero / Net-Negative Exception',column),.5,source='03_segment_profiles.csv')

    # Section 5.1 table: coverage and Loyal familiar mix.
    covered = get(recommendations,'Metric','CoveredCustomers','Value')
    n_customers = len(integrated) if integrated is not None else None
    check('5.1 December recommendations','Customer coverage (%)',77.89,
          100*covered/n_customers if covered is not None and n_customers else None,.005,source='08_recommendation_report_metrics.csv')
    check('5.1 December recommendations','Loyal familiar mix (%)',56.1,
          extra['fig12_mix']['Loyal Customers']['familiar_pct'] if extra else None,.05,
          source='11_extra_report_metrics.json')

    # Section 5.2: "the reported paired bootstrap intervals exclude zero" (1 = upper bound below zero).
    for date in ['2011-06-10','2011-09-10']:
        upper = get(bootstrap,'Cutoff',date,'CI95_Upper')
        check('5.2 Next basket',f'{date} bootstrap interval excludes zero',1,
              float(upper < 0) if upper is not None else None,source='06_paired_bootstrap.csv')

    # Section 5.3: segment precisions and the size of the first future basket.
    by_segment = read_csv('06_familiar_new_by_segment.csv')
    sept_seg = by_segment[by_segment.Cutoff.eq('2011-09-10')] if by_segment is not None else None
    for segment, category, benchmark in [('Loyal Customers','Familiar',24.1),('Loyal Customers','Customer-new',4.9),
                                         ('Growing Customers','Familiar',22.9),('Growing Customers','Customer-new',9.1)]:
        value = None
        if sept_seg is not None:
            hit = sept_seg[sept_seg.CustomerSegment.eq(segment) & sept_seg.Category.eq(category)]
            value = 100*float(hit.ItemPrecision.iloc[0]) if len(hit) else None
        check('5.3 Familiar vs new',f'September {segment.split()[0]} {category} precision (%)',benchmark,value,.05,
              source='06_familiar_new_by_segment.csv')
    check('5.3 Familiar vs new','September mean distinct products in first future basket',26.3,
          extra['first_future_basket']['2011-09-10']['mean_distinct'] if extra else None,.05,
          source='11_extra_report_metrics.json')

    # Section 6: selected model settings.
    meta_path = OUTPUT / '07_model_metadata.json'
    params = json.loads(meta_path.read_text(encoding='utf-8')).get('model_parameters', {}) if meta_path.exists() else {}
    for name, benchmark in [('n_estimators',220),('max_depth',3),('learning_rate',.05)]:
        check('6. Model validation',f'XGBoost {name}',benchmark,params.get(name),1e-9,source='07_model_metadata.json')

    # Section 9 table (RQ3): rounded restatements of the out-of-time results.
    for column, benchmark, tolerance, scale in [('ROC_AUC',.788,.0005,1),('Top20Rate',84.6,.05,100),
                                                ('RepeatRate',43.5,.05,100)]:
        raw = row_value(test,column)
        check('9. Conclusions',f'RQ3 {column}',benchmark,raw*scale if raw is not None else None,tolerance,
              source='07_out_of_time_metrics.csv')

    # Section 7.1 table: recency, frequency and rule source of the illustrative customers.
    source_names = {'Segment rules':'Segment','Overall fallback':'Overall fallback','No triggered rule':'No triggered rule'}
    for label, segment, recency, frequency, rule_source in [
            ('A','Loyal Customers',3,11,'Segment'),('B','Cooling Customers',92,3,'Segment'),
            ('C','Growing Customers',58,4,'Segment'),('D','Inactive Customers',642,1,'Segment')]:
        pick = None
        if examples is not None:
            sub = examples[examples.CustomerSegment.eq(segment)]
            median = sub.PredictedRepeatProbability90D.median()
            pick = sub.assign(d=(sub.PredictedRepeatProbability90D-median).abs()).sort_values(['d','CustomerID']).iloc[0]
        check('7. Customer decision table',f'Illustrative customer {label} recency',recency,
              pick.RecencyDays if pick is not None else None,source='08_customer_intelligence.csv')
        check('7. Customer decision table',f'Illustrative customer {label} frequency',frequency,
              pick.FrequencyInvoices if pick is not None else None,source='08_customer_intelligence.csv')
        check('7. Customer decision table',f'Illustrative customer {label} rule source is "{rule_source}"',1,
              float(source_names.get(pick.RecommendationSource) == rule_source) if pick is not None else None,
              source='08_customer_intelligence.csv')

    result=pd.DataFrame(rows)
    result.to_csv(OUTPUT / '09_report_reconciliation.csv',index=False)
    summary={status:int((result.Status==status).sum()) for status in ['PASS','FAIL','UNVERIFIED']}
    with open(OUTPUT / '09_report_summary.json','w',encoding='utf-8') as handle:
        json.dump({'counts':summary,'criteria':'Absolute tolerance equal to the published precision',
                   'benchmarks':'Numbers printed in the revised report text and tables (November 2026)'},
                  handle,ensure_ascii=False,indent=2)

    report_lines=['# Report-to-code reconciliation (recomputed run)', '',
                  f"PASS: {summary['PASS']} | FAIL: {summary['FAIL']} | UNVERIFIED: {summary['UNVERIFIED']}", '',
                  'Benchmarks are the values printed in the revised report (November 2026 revision). '
                  'A FAIL indicates that report text or a figure does not match the executable code.', '',
                  '| Section | Metric | Report | Recomputed | Difference | Status |',
                  '|---|---|---:|---:|---:|---|']
    for record in result.itertuples(index=False):
        if record.Status == 'PASS':
            continue
        fmt=lambda x: '—' if x is None or pd.isna(x) else f'{x:.6g}'
        report_lines.append('| '+ ' | '.join([
            record.Section,record.Metric.replace('|','/'),fmt(record.ReportBenchmark),
            fmt(record.Recomputed),fmt(record.Difference),record.Status])+' |')
    report_lines.extend(['','Report and slides were revised with tracked changes to the values recomputed by this package.'])
    (OUTPUT / '09_report_reconciliation.md').write_text('\n'.join(report_lines),encoding='utf-8')

    print(result[['Section','Metric','ReportBenchmark','Recomputed','Difference','Status']].to_string(index=False))
    print('\nAUDIT SUMMARY:',summary)
    print(f'Detailed checks: {OUTPUT / "09_report_reconciliation.csv"}')


# ============================================================================
# Step 11 (former 11_extra_report_metrics.py)
# ============================================================================
def step_11_extra_report_metrics():
    """Supplementary report metrics computed from the pipeline outputs (no refitting).
Covers figures/statements whose numbers are not written by steps 01-09:
Figure 8 (gap distribution), Figure 12 (suggestion mix), Figure 20 (window sensitivity),
Section 5.3 mean distinct products in first future basket, Figure 19 (mean score by segment)."""
    import json, numpy as np, pandas as pd
    out={}
    dyn=load_customers('05_customer_dynamics.csv')
    seg_names=['Loyal Customers','Cooling Customers','Growing Customers','Inactive Customers']
    # Figure 8
    g={}
    for s in seg_names:
        d=dyn[(dyn.CustomerSegment==s)&dyn.RhythmEligible]['MedianInterpurchaseDays']
        g[s]={'n':int(len(d)),'median':float(d.median()),'q1':float(d.quantile(.25)),'q3':float(d.quantile(.75)),
              'p10':float(d.quantile(.10)),'p90':float(d.quantile(.90)),'p5':float(d.quantile(.05)),'p95':float(d.quantile(.95))}
    out['fig8_gap']=g
    out['purchase_days_ge4']={s:int(((dyn.CustomerSegment==s)&(dyn.PurchaseDaysObserved>=4)).sum()) for s in seg_names}
    # Figure 12
    ci=load_customers('08_customer_intelligence.csv')
    purch=load_table('01_identified_purchases.csv')
    seen=purch.groupby('CustomerID')['StockCode'].agg(set).to_dict()
    mix={}
    for s in seg_names:
        f=n=0
        for r in ci[ci.CustomerSegment==s][['CustomerID','RecommendedProductCodes']].itertuples(index=False):
            if pd.isna(r.RecommendedProductCodes) or not r.RecommendedProductCodes: continue
            for c in r.RecommendedProductCodes.split('|'):
                n+=1; f+= c in seen.get(r.CustomerID,set())
        mix[s]={'suggestions':n,'familiar':f,'familiar_pct':100*f/n if n else None}
    out['fig12_mix']=mix
    out['fig12_source']=ci.groupby('CustomerSegment')['RecommendationSource'].value_counts().unstack(fill_value=0).to_dict('index')
    out['fig19']=ci.groupby('CustomerSegment')['PredictedRepeatProbability90D'].mean().mul(100).to_dict()
    out['priority_bands']=ci['PriorityBand'].value_counts().to_dict()
    # Figure 20
    end=pd.Timestamp('2011-12-10')
    def spend(a,b):
        m=(purch.InvoiceDate>=a)&(purch.InvoiceDate<b); return purch[m].groupby('CustomerID')['LineRevenue'].sum()
    sens=[]
    for days in [30,45,60,90,120,180]:
        cur=spend(end-pd.Timedelta(days=days),end); prev=spend(end-pd.Timedelta(days=2*days),end-pd.Timedelta(days=days))
        t=dyn[['CustomerID','CustomerSegment','CustomerTenureDays']].copy()
        t['cur']=t.CustomerID.map(cur).fillna(0); t['prev']=t.CustomerID.map(prev).fillna(0)
        t['comp']=(t.CustomerTenureDays>=2*days)&(t.prev>0)
        t['dec']=t.comp&(t.cur<.75*t.prev); t['inc']=t.comp&(t.cur>1.25*t.prev)
        L=t[t.CustomerSegment=='Loyal Customers']; G=t[t.CustomerSegment=='Growing Customers']
        sens.append({'days':days,'loyal_decline':int(L.dec.sum()),'loyal_decline_pct':100*L.dec.mean(),
                     'growing_comparable':int(G.comp.sum()),'growing_increasing':int(G.inc.sum()),
                     'growing_increasing_pct':100*G.inc.sum()/G.comp.sum()})
    out['fig20']=sens
    # threshold sensitivity at 90 days (10/25/50%)
    cur=spend(end-pd.Timedelta(days=90),end); prev=spend(end-pd.Timedelta(days=180),end-pd.Timedelta(days=90))
    L=dyn[dyn.CustomerSegment=='Loyal Customers'][['CustomerID','CustomerTenureDays']].copy()
    L['cur']=L.CustomerID.map(cur).fillna(0); L['prev']=L.CustomerID.map(prev).fillna(0)
    out['loyal_decline_threshold']={str(th):int(((L.CustomerTenureDays>=180)&(L.prev>0)&(L.cur<(1-th)*L.prev)).sum()) for th in [.10,.25,.50]}
    # Section 5.3: distinct products in first future basket
    allp=load_table('01_merchandise_purchases.csv')
    fb={}
    for c in ['2011-06-10','2011-09-10']:
        nd=pd.Timestamp(c)+pd.Timedelta(days=1)
        hist=allp[(allp.InvoiceDate<nd)&allp.CustomerID.notna()]
        fut=allp[(allp.InvoiceDate>=nd)&(allp.InvoiceDate<nd+pd.Timedelta(days=90))&allp.CustomerID.notna()]
        ff=fut.groupby(['CustomerID','Invoice']).agg(Date=('InvoiceDate','min'),n=('StockCode','nunique')).reset_index().sort_values(['CustomerID','Date','Invoice']).groupby('CustomerID').head(1)
        ff=ff[ff.CustomerID.isin(set(hist.CustomerID))]
        fb[c]={'returners':int(len(ff)),'mean_distinct':float(ff.n.mean()),'median_distinct':float(ff.n.median())}
    out['first_future_basket']=fb
    json.dump(out,open(OUTPUT/'11_extra_report_metrics.json','w'),indent=1,default=float)
    print(json.dumps(out,indent=1,default=float))


# ============================================================================
# Step 12 (former 12_final_report_figures.py)
# ============================================================================
def step_12_final_report_figures():
    """Step 12: every data figure in the report and the slide charts, drawn ONLY from pipeline outputs.

Report figures 2-14 and 16-20 (1 and 15 are method diagrams without data) and the slide
charts on slides 6, 9, 13, 14, in the visual style of the submitted documents.
Every plotted value is read from outputs/*.csv; nothing is typed in by hand.
Run after steps 01-09 and 11: python 12_final_report_figures.py
"""
    import numpy as np, pandas as pd, matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter, FuncFormatter

    FIG = OUTPUT / 'final_figures'; FIG.mkdir(exist_ok=True)
    NAVY, TEAL, GRAY, RED, GOLD, INK = '#15395b', '#137f86', '#82909d', '#bc4444', '#b87d25', '#172a38'
    GRID, SPINE = '#e2e7eb', '#aab5be'
    SEG = ['Loyal', 'Cooling', 'Growing', 'Inactive']
    read = lambda n: pd.read_csv(OUTPUT / n)


    def report_style():
        plt.rcdefaults()
        plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 7.2, 'text.color': INK,
            'axes.labelcolor': INK, 'xtick.color': INK, 'ytick.color': INK, 'axes.edgecolor': SPINE,
            'axes.spines.top': False, 'axes.spines.right': False, 'axes.titleweight': 'bold',
            'axes.titlesize': 7.6, 'axes.labelsize': 7.2, 'axes.linewidth': .8})


    def new(px, py, dpi=300, **kw):
        return plt.subplots(figsize=(px / dpi, py / dpi), dpi=dpi, **kw)


    def save(fig, name, dpi=300, transparent=False):
        fig.savefig(FIG / name, dpi=dpi, transparent=transparent); plt.close(fig); print('wrote', name)


    def grid(ax, axis='y'):
        ax.grid(True, axis=axis, color=GRID, lw=.6); ax.set_axisbelow(True)


    # ---------------- Figure 2 ----------------
    def fig02():
        report_style()
        a = read('01_data_audit.csv').set_index('Stage')['Count']
        keys = ['Raw rows', 'After sheet-overlap rule', 'Valid positive purchase lines',
                'Valid merchandise purchase lines', 'Purchase lines after duplicates and cancellations',
                'Identified merchandise purchase lines']
        v = a.loc[keys].to_numpy(int)
        fig, ax = new(1444, 560)
        ax.bar(range(6), v / 1000, width=.58, color=[GRAY, NAVY, NAVY, NAVY, TEAL, TEAL], zorder=3)
        for i, x in enumerate(v):
            ax.text(i, x / 1000 + 8, f'{x:,}', ha='center', va='bottom', fontweight='bold', fontsize=5.8)
        for i in range(1, 6):
            ax.text(i - .5, 1205, f'−{v[i-1]-v[i]:,}', ha='center', va='center', color=RED, fontsize=6.4)
        ax.set_xticks(range(6), ['Raw rows', 'After sheet\noverlap', 'Valid positive\npurchases',
                                 'Merchandise\npurchases', 'No duplicates\nor cancelled', 'With\ncustomer ID'], fontsize=6.2)
        ax.set_ylabel('Lines (thousands)'); ax.set_ylim(0, 1240); grid(ax)
        fig.subplots_adjust(left=.1, right=.985, top=.97, bottom=.25)
        save(fig, 'figure_02.png')


    # ---------------- Figure 5 (report) and slide 9 ----------------
    def fig05(name='figure_05.png', px=1646, py=680):
        report_style()
        k = read('03_k_selection.csv').sort_values('K'); r = read('03_stability_repeats.csv')
        b = r.groupby('K')['ARI'].agg(['min', 'max']).reindex(k['K'])
        fig, ax = new(px, py, ncols=2)
        ax[0].plot(k.K, k.Silhouette, color=NAVY, marker='o', ms=4, lw=1.6)
        ax[0].set_ylabel('Silhouette score'); ax[0].set_title('Separation')
        ax[0].set_ylim(.28, .47); ax[0].set_yticks([.30, .35, .40, .45])
        ax[1].fill_between(k.K, b['min'], b['max'], color=TEAL, alpha=.15, lw=0)
        ax[1].plot(k.K, k.StabilityARI, color=TEAL, marker='o', ms=4, lw=1.6)
        ax[1].set_ylabel('Adjusted Rand index'); words = {2: 'two', 3: 'three', 4: 'four', 5: 'five'}
        ax[1].set_title(f'Stability across {words.get(r.Repeat.nunique(), r.Repeat.nunique())} subsamples')
        ax[1].set_ylim(.65, 1.005)
        for a in ax:
            a.axvline(4, color=RED, ls='--', lw=.9); a.set_xticks(k.K); a.set_xlabel('Number of clusters'); grid(a)
        fig.subplots_adjust(left=.1, right=.985, top=.9, bottom=.2, wspace=.33)
        save(fig, name)


    # ---------------- Figure 9 ----------------
    def fig09():
        report_style()
        s = read('05_year_matched_comparison.csv')
        fig, ax = new(1646, 553, ncols=2)
        spec = [(s.iloc[0], f'All {int(s.iloc[0].Customers):,} Loyal customers', TEAL),
                (s.iloc[1], f'{int(s.iloc[1].Customers):,} history-eligible Loyal customers', RED)]
        for a, (row, title, col) in zip(ax, spec):
            end = 100 + row.ChangePct
            a.plot([0, 1], [100, end], color=col, marker='o', ms=4.5, lw=1.8)
            a.text(.9, end + 6, f'{row.ChangePct:+.1f}%', color=col, fontweight='bold', ha='center', va='bottom')
            a.set_xticks([0, 1], ['Comparison\nperiod', 'Latest\n90 days']); a.set_xlim(-.2, 1.2)
            a.set_ylim(70, 190); a.set_yticks(range(80, 181, 20)); a.set_ylabel('Comparison period = 100')
            a.set_title(title); grid(a)
        fig.subplots_adjust(left=.09, right=.985, top=.87, bottom=.24, wspace=.3)
        save(fig, 'figure_09.png')


    # ---------------- Figure 13 ----------------
    METHODS = ['Market popularity', 'Overall rules', 'Segment rules + fallback', 'Personal frequency']
    def fig13():
        report_style()
        b = read('06_next_basket_backtest.csv'); ci = read('06_paired_bootstrap.csv')
        fig, ax = new(1647, 727, ncols=2, gridspec_kw={'width_ratios': [1.45, 1]})
        a = ax[0]
        for i, m in enumerate(METHODS):
            j = 100 * b[(b.Cutoff == '2011-06-10') & (b.Method == m)].HitRate.iloc[0]
            s = 100 * b[(b.Cutoff == '2011-09-10') & (b.Method == m)].HitRate.iloc[0]
            y = 3 - i
            a.plot(j, y, 'o', color=GRAY, ms=5.5, zorder=3); a.plot(s, y, 's', color=NAVY, ms=5.5, zorder=3)
            a.text(j, y + .2, f'{j:.1f}%', ha='center', va='bottom', fontsize=6.2)
            a.text(s, y - .2, f'{s:.1f}%', ha='center', va='top', fontsize=6.2)
        a.set_yticks([3, 2, 1, 0], ['Market popularity', 'Overall rules', 'Segment rules + fallback',
                                    'Personal purchase frequency'])
        a.set_ylim(-.5, 3.7); a.set_xlim(10, 44); a.set_xticks([10, 20, 30, 40]); a.xaxis.set_major_formatter(PercentFormatter(decimals=1))
        a.set_xlabel('Hit rate among returning customers'); grid(a, 'x')
        a.plot([], [], 'o', color=GRAY, ms=5.5, label='June backtest'); a.plot([], [], 's', color=NAVY, ms=5.5, label='September backtest')
        a.legend(frameon=False, ncol=2, loc='lower center', bbox_to_anchor=(.5, 1.0), fontsize=6.2)
        a = ax[1]
        for y, (c, lab) in zip([1, 0], [('2011-06-10', 'June'), ('2011-09-10', 'September')]):
            r = ci[ci.Cutoff == c].iloc[0]; d, lo, hi = 100 * r.MeanDifferenceSegmentMinusOverall, 100 * r.CI95_Lower, 100 * r.CI95_Upper
            a.errorbar(d, y, xerr=[[d - lo], [hi - d]], fmt='o', color=RED, ms=4.5, capsize=3, lw=1.2)
            a.text(d, y - .1, f'{d:.2f} pp', color=RED, ha='center', va='top', fontsize=6.4)
        a.axvline(0, color=GRAY, ls='--', lw=1.1); a.set_yticks([1, 0], ['June', 'September']); a.set_ylim(-.6, 1.5)
        a.set_xlim(-4.5, .5); a.set_xticks([-4, -2, 0]); a.xaxis.set_major_formatter(PercentFormatter(decimals=1))
        a.set_xlabel('Hit rate difference\nsegment minus overall'); grid(a, 'x')
        fig.subplots_adjust(left=.3, right=.98, top=.9, bottom=.23, wspace=.5)
        save(fig, 'figure_13.png')


    # ---------------- Figure 14 ----------------
    def fig14():
        report_style()
        d = read('06_familiar_new_by_segment.csv'); d = d[d.Cutoff == '2011-09-10'].set_index(['CustomerSegment', 'Category'])
        fig, ax = new(1666, 670, ncols=2)
        y = np.arange(4)[::-1]
        fam = [100 * d.loc[(s + ' Customers', 'Familiar'), 'ItemPrecision'] for s in SEG]
        nw = [100 * d.loc[(s + ' Customers', 'Customer-new'), 'ItemPrecision'] for s in SEG]
        a = ax[0]
        a.barh(y + .16, fam, height=.3, color=NAVY, label='Familiar', zorder=3)
        a.barh(y - .16, nw, height=.3, color=TEAL, label='New to customer', zorder=3)
        for yy, f, n in zip(y, fam, nw):
            a.text(f + .5, yy + .16, f'{f:.1f}%', va='center', fontsize=6); a.text(n + .5, yy - .16, f'{n:.1f}%', va='center', fontsize=6)
        a.set_yticks(y, SEG); a.set_xlim(0, 34); a.set_xticks([0, 10, 20, 30]); a.xaxis.set_major_formatter(PercentFormatter(decimals=1))
        a.set_xlabel('Products hit / products recommended'); grid(a, 'x')
        a.legend(frameon=False, ncol=2, loc='lower left', bbox_to_anchor=(.05, 1.0), fontsize=6.2)
        m = np.array([[100 * d.loc[(s + ' Customers', c), 'MeanTargetRecall'] for c in ['Familiar', 'Customer-new']] for s in SEG])
        a = ax[1]
        a.imshow(m, cmap='Blues', vmin=0, vmax=m.max() * 1.25, aspect='auto')
        for i in range(4):
            for j in range(2):
                a.text(j, i, f'{m[i, j]:.2f}%', ha='center', va='center', color='white' if m[i, j] > .6 * m.max() * 1.25 else INK)
        a.set_xticks([0, 1], ['Familiar target\nrecall', 'New target\nrecall']); a.set_yticks(range(4), SEG)
        a.set_title('Mean share of each target type recovered', fontsize=6.6); a.tick_params(length=0)
        fig.subplots_adjust(left=.1, right=.985, top=.88, bottom=.23, wspace=.38)
        save(fig, 'figure_14.png')


    # ---------------- Figure 16 ----------------
    def fig16():
        report_style()
        v = read('07_validation_comparison.csv').set_index('Model')
        order = ['Logistic Regression', 'Random Forest', 'XGBoost']; lab = ['Logistic regression', 'Random forest', 'XGBoost']
        fig, ax = new(1646, 611, ncols=2)
        for a, col, xl in zip(ax, ['ROC_AUC', 'PR_AUC'], ['Validation ROC AUC', 'Validation PR AUC']):
            for i, m in enumerate(order):
                x = v.loc[m, col]; y = 2 - i
                a.plot(x, y, 'o', ms=6, color=NAVY if m == 'XGBoost' else GRAY, zorder=3)
                a.text(x + .012, y, f'{x:.3f}', va='center')
            a.set_yticks([2, 1, 0], lab); a.set_xlim(.65, .9); a.set_xticks(np.arange(.65, .86, .1))
            a.set_ylim(-.3, 2.3); a.set_xlabel(xl); grid(a, 'x')
        fig.subplots_adjust(left=.205, right=.985, top=.95, bottom=.23, wspace=.95)
        save(fig, 'figure_16.png')


    # ---------------- Figure 17 ----------------
    def fig17():
        report_style()
        d = read('07_test_decile_rates.csv'); t = read('07_out_of_time_metrics.csv').iloc[0]
        fig, ax = new(1453, 599)
        r = 100 * d.ActualRepeatRate
        ax.bar(d.Decile, r, width=.68, color=[TEAL, TEAL] + [NAVY] * 8, zorder=3)
        base = 100 * t.RepeatRate
        for x, y in zip(d.Decile, r):
            near = abs(y - base) < 6 and y < base
            ax.text(x, y - 1.5 if near else y + 1.2, f'{y:.1f}%', ha='center', va='top' if near else 'bottom',
                    color='white' if near else INK, fontsize=5.6, zorder=5)
        ax.axhline(base, color=RED, ls='--', lw=1, label=f'Overall repeat rate {base:.2f}%', zorder=4)
        ax.legend(frameon=False, loc='upper right'); ax.set_xticks(d.Decile); ax.set_ylim(0, 108)
        ax.yaxis.set_major_formatter(PercentFormatter(decimals=0)); ax.set_ylabel('Actual repeat rate')
        ax.set_xlabel('Predicted propensity decile with 1 highest'); grid(ax)
        fig.subplots_adjust(left=.13, right=.985, top=.97, bottom=.21)
        save(fig, 'figure_17.png')


    # ---------------- Figure 18 ----------------
    def fig18():
        plt.rcdefaults(); plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 7})
        s = read('07_shap_importance.csv').head(8).iloc[::-1]
        fig, ax = plt.subplots(figsize=(1781 / 220, 1031 / 220), dpi=220)
        ax.barh(s.Feature, s.MeanAbsoluteSHAP, color='#1f4e78', zorder=3)
        ax.set_title('Production XGBoost: Global Feature Importance', fontsize=10)
        ax.set_xlabel('Mean absolute SHAP value', fontsize=7.5); ax.grid(True, axis='x', color='#e6e6e6', lw=.6); ax.set_axisbelow(True)
        fig.subplots_adjust(left=.19, right=.965, top=.93, bottom=.11)
        save(fig, 'figure_18.png', dpi=220)


    # ---------------- Figure 19 ----------------
    def fig19():
        report_style()
        c = read('08_customer_intelligence.csv'); p = read('03_segment_profiles.csv').set_index('CustomerSegment')
        fig, ax = new(1434, 626)
        cols = {'Loyal': NAVY, 'Cooling': GOLD, 'Growing': TEAL, 'Inactive': GRAY}
        off = {'Loyal': (9, 8), 'Cooling': (9, 9), 'Growing': (9, -13), 'Inactive': (10, 9)}
        for s in SEG:
            full = s + ' Customers'
            x = 100 * c.loc[c.CustomerSegment == full, 'PredictedRepeatProbability90D'].mean()
            y = p.loc[full, 'NetValueSharePct']; n = p.loc[full, 'Customers']
            ax.scatter(x, y, s=n * .42, color=cols[s], alpha=.8, edgecolor='white', lw=1, zorder=3)
            ax.annotate(s, (x, y), xytext=off[s], textcoords='offset points', fontweight='bold', fontsize=8)
        ax.set_xlim(0, 93); ax.set_ylim(-6, 87); ax.set_yticks(range(0, 81, 20))
        ax.xaxis.set_major_formatter(PercentFormatter(decimals=0)); ax.yaxis.set_major_formatter(PercentFormatter(decimals=0))
        ax.set_xlabel('Mean predicted 90-day repeat probability, December 2011 scores')
        ax.set_ylabel('Historical net value share'); grid(ax)
        fig.subplots_adjust(left=.12, right=.985, top=.97, bottom=.21)
        save(fig, 'figure_19.png')


    # =============== Slide charts (deck style: Arial-like, Okabe-Ito colours) ===============
    BLUE, ORANGE, SKY, DARK = '#0072b2', '#d55e00', '#56b4e9', '#051c2c'
    def slide_style(size=11):
        plt.rcdefaults()
        plt.rcParams.update({'font.family': ['Liberation Sans', 'Arial', 'DejaVu Sans'], 'font.size': size, 'axes.spines.top': False,
            'axes.spines.right': False, 'axes.linewidth': 1.1, 'xtick.major.width': 1.1, 'ytick.major.width': 1.1,
            'text.color': '#111111', 'axes.edgecolor': '#111111'})


    def slide06():
        slide_style(13)
        a = read('01_data_audit.csv').set_index('Stage')['Count']
        keys = ['Raw rows', 'After sheet-overlap rule', 'Valid positive purchase lines',
                'Valid merchandise purchase lines', 'Purchase lines after duplicates and cancellations',
                'Identified merchandise purchase lines']
        v = a.loc[keys].to_numpy(int)
        m = read('02_monthly_revenue.csv'); m['Y'] = m.Month.str[:4].astype(int); m['M'] = m.Month.str[5:].astype(int)
        fig = plt.figure(figsize=(1276 / 115, 1010 / 115), dpi=115)
        ax = fig.add_axes([.24, .625, .73, .305])
        y = np.arange(6)[::-1]
        ax.barh(y, v, height=.62, color=[BLUE] * 5 + [ORANGE])
        for yy, x, bold in zip(y, v, [0, 0, 0, 0, 0, 1]):
            ax.text(x + 13000, yy, f'{x:,}', va='center', fontweight='bold' if bold else 'normal')
        ax.set_yticks(y, ['Raw rows', 'After sheet overlap', 'Valid positive purchases', 'Merchandise purchases',
                          'No duplicates or cancelled', 'With customer ID'])
        ax.set_xlim(0, 1.25e6); ax.set_xticks(np.arange(0, 1.21e6, 2e5))
        ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: '0' if x == 0 else f'{x/1e6:.1f}m'))
        ax.set_xlabel('Transaction lines (count)', labelpad=8)
        ax.set_title('Online Retail II | Rows after each cleaning step (lines)', loc='left', fontweight='bold', fontsize=14, pad=14)
        fig.text(.01, .965, 'a', fontweight='bold', fontsize=16); fig.text(.01, .472, 'b', fontweight='bold', fontsize=16)
        ax = fig.add_axes([.24, .08, .73, .34])
        mon = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        y10 = m[m.Y == 2010].sort_values('M').GrossMerchandiseRevenue.to_numpy() / 1000
        y11 = m[m.Y == 2011].sort_values('M').GrossMerchandiseRevenue.to_numpy() / 1000
        ax.plot(range(12), y10, color=BLUE, marker='o', lw=2.6, ms=6)
        ax.plot(range(11), y11[:11], color=ORANGE, marker='o', lw=2.6, ms=6)
        ax.plot([10, 11], y11[10:], color=ORANGE, lw=2.6, ls=(0, (2, 2)))
        ax.plot(11, y11[11], 'o', mfc='white', mec=ORANGE, mew=2.4, ms=10)
        ax.text(9.9, y11[10] + 250, f'Nov 2011: {y11[10]:,.0f}', color=ORANGE, fontweight='bold', ha='center', fontsize=13)
        ax.text(9.9, y10[10] + 120, f'Nov 2010: {y10[10]:,.0f}', color=BLUE, fontweight='bold', ha='center', fontsize=13)
        ax.text(6.6, y11[7] + 75, '2011', color=ORANGE, fontweight='bold', fontsize=14)
        ax.text(7.1, y10[7] - 120, '2010', color=BLUE, fontweight='bold', fontsize=14)
        ax.annotate('Dec 2011: 1-9 Dec only', (11, y11[11]), xytext=(8.2, 240), arrowprops=dict(arrowstyle='-', lw=1), fontsize=12.5)
        ax.set_xticks(range(12), mon); ax.set_ylim(0, 1800); ax.set_yticks([0, 500, 1000, 1500]); ax.set_xlim(-.4, 11.5)
        ax.set_ylabel('Gross revenue (GBP 000)')
        fig.text(.24, .468, 'Merchandise | Gross revenue by month (GBP 000)', fontweight='bold', fontsize=14)
        fig.text(.24, .436, 'Gross revenue of merchandise lines, not net value', style='italic', fontsize=11.5)
        save(fig, 'slide06_cleaning_revenue.png', dpi=115)


    def slide13():
        slide_style(15)
        b = read('06_next_basket_backtest.csv')
        fig, ax = plt.subplots(figsize=(1243 / 130, 733 / 130), dpi=130)
        vals, labs = [], []
        short = {'Market popularity': 'Market popularity', 'Overall rules': 'Overall rules',
                 'Segment rules + fallback': 'Segment rules', 'Personal frequency': 'Personal frequency'}
        for m in METHODS:
            for c, t in [('2011-06-10', 'Jun'), ('2011-09-10', 'Sep')]:
                vals.append(100 * b[(b.Cutoff == c) & (b.Method == m)].HitRate.iloc[0]); labs.append(t)
        x = np.arange(8)
        ax.bar(x, vals, width=.7, color='#004aad')
        for xx, v in zip(x, vals):
            ax.text(xx, v - .8, f'{v:.1f}', ha='center', va='top', color='white', fontsize=15)
        ax.set_xticks(x, labs); ax.set_ylim(0, 45); ax.set_yticks(range(0, 41, 10))
        for i, m in enumerate(METHODS):
            ax.text(2 * i + .5, max(vals[2 * i:2 * i + 2]) + 1.2, short[m], ha='center', va='bottom', fontsize=11.5, fontweight='bold')
        for s in ax.spines.values(): s.set_visible(False)
        ax.tick_params(length=0)
        fig.subplots_adjust(left=.12, right=.92, top=.855, bottom=.195)
        save(fig, 'slide13_hit_rates.png', dpi=130, transparent=True)


    def slide14():
        slide_style(13)
        v = read('07_validation_comparison.csv').set_index('Model'); d = read('07_test_decile_rates.csv'); t = read('07_out_of_time_metrics.csv').iloc[0]
        fig = plt.figure(figsize=(2161 / 115, 826 / 115), dpi=115)
        ax = fig.add_axes([.055, .19, .29, .67])
        order = [('Logistic Regression', 'LR', BLUE), ('Random Forest', 'RF', SKY), ('XGBoost', 'XGBoost', ORANGE)]
        for i, (m, lab, col) in enumerate(order):
            vals = [v.loc[m, 'ROC_AUC'], v.loc[m, 'PR_AUC']]
            xs = np.array([0, 1]) + (i - 1) * .26
            ax.bar(xs, vals, width=.24, color=col, label=lab)
            for xx, val in zip(xs, vals):
                ax.text(xx, val + .015, f'{val:.3f}', rotation=90, ha='center', va='bottom', fontsize=11.5,
                        fontweight='bold' if m == 'XGBoost' else 'normal')
        ax.set_xticks([0, 1], ['ROC-AUC', 'PR-AUC']); ax.set_ylim(0, 1.12); ax.set_yticks(np.arange(0, 1.01, .2))
        ax.tick_params(labelsize=15); ax.set_ylabel('Area under curve (0 to 1)', fontsize=15); ax.set_xlabel('Validation metric, June', fontsize=15)
        ax.legend(frameon=False, ncol=3, loc='upper center', fontsize=12.5, handlelength=1, columnspacing=1.2)
        fig.text(.02, .905, 'a  June validation | ROC-AUC and PR-AUC by model', fontweight='bold', fontsize=15)
        ax = fig.add_axes([.44, .19, .55, .67])
        r = 100 * d.ActualRepeatRate; base = 100 * t.RepeatRate
        ax.bar(d.Decile, r, width=.7, color=[ORANGE, ORANGE] + [DARK] * 8)
        for x, y in zip(d.Decile, r):
            near = abs(y - base) < 7 and y < base
            ax.text(x, y - 2 if near else y + 1.8, f'{y:.1f}', ha='center', va='top' if near else 'bottom',
                    color='white' if near else '#111111', fontweight='bold' if x <= 2 else 'normal', zorder=5)
        ax.axhline(base, color='#111111', ls='--', lw=1.6, zorder=4)
        ax.text(10.45, base + 2, f'All customers {base:.2f}%', ha='right', va='bottom', fontsize=12.5)
        ax.text(10.5, 97, f'Top 20%: {100*t.Top20Rate:.2f}%, lift {t.LiftAt20:.2f}x; ROC-AUC {t.ROC_AUC:.4f}; n = {int(t.Customers):,}',
                ha='right', va='center', fontweight='bold', fontsize=12.5, bbox=dict(fc='white', ec='#111111', lw=1.3, pad=5))
        ax.set_xticks(d.Decile); ax.set_ylim(0, 105); ax.set_xlim(.4, 10.6); ax.tick_params(labelsize=15)
        ax.set_ylabel('Actual repeat rate (%)', fontsize=15); ax.set_xlabel('Predicted score decile (1 = highest score)', fontsize=15)
        fig.text(.4, .905, 'b  September out-of-time | Actual repeat rate by score decile (%)', fontweight='bold', fontsize=15)
        fig.text(.005, .025, 'Source: DM_Report ver 2.0, Fig. 16, Fig. 17, OOT metrics table (sec. 6.1); recomputed from pipeline outputs. Source does not report uncertainty.',
                 style='italic', fontsize=11.5)
        save(fig, 'slide14_model.png', dpi=115)


    # ================= Figures 3, 4, 6, 7, 8, 10, 11, 12, 20 =================
    import json
    EXTRA = lambda: json.load(open(OUTPUT / '11_extra_report_metrics.json'))
    MON = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']


    def fig03():
        report_style()
        m = read('02_monthly_revenue.csv'); m['Y'] = m.Month.str[:4].astype(int); m['M'] = m.Month.str[5:].astype(int)
        y10 = m[m.Y == 2010].sort_values('M').GrossMerchandiseRevenue.to_numpy() / 1000
        y11 = m[m.Y == 2011].sort_values('M').GrossMerchandiseRevenue.to_numpy() / 1000
        fig, ax = new(1646, 1013, nrows=2, gridspec_kw={'height_ratios': [1.25, 1]})
        a = ax[0]
        a.plot(range(12), y10 / 1000, color=GRAY, marker='o', ms=3, lw=1.4, label='2010')
        a.plot(range(11), y11[:11] / 1000, color=NAVY, marker='o', ms=3, lw=1.4, label='2011')
        a.plot([10, 11], y11[10:] / 1000, color=NAVY, lw=1.2, ls=':')
        a.plot(11, y11[11] / 1000, 'o', mfc='white', mec=NAVY, ms=5)
        a.annotate('Dec 2011\npartial month', (11, y11[11] / 1000), xytext=(9.3, .36), fontsize=6.4,
                   arrowprops=dict(arrowstyle='-', color=GRAY, lw=.7))
        a.set_xticks(range(12), MON); a.set_ylabel('Gross merchandise revenue (£m)'); a.set_ylim(.3, 1.65)
        a.legend(frameon=False, ncol=2, loc='upper left'); grid(a)
        a = ax[1]
        mat = np.vstack([y10, y11])
        a.imshow(mat, cmap='Blues', aspect='auto', vmin=mat.min() * .8, vmax=mat.max())
        for i in range(2):
            for j in range(12):
                a.text(j, i, f'{mat[i, j]:,.0f}', ha='center', va='center', fontsize=6,
                       color='white' if mat[i, j] > 1000 else INK)
        a.set_yticks([0, 1], ['2010', '2011']); a.set_xticks(range(12), [x[0] for x in MON[:11]] + ['D*'])
        a.set_xlabel('Monthly revenue (£000)   *December 2011 ends on 9 December'); a.tick_params(length=0)
        for sp in a.spines.values(): sp.set_visible(False)
        fig.subplots_adjust(left=.085, right=.985, top=.98, bottom=.09, hspace=.22)
        save(fig, 'figure_03.png')


    def fig04():
        report_style()
        f = read('02_customer_features.csv'); v = np.sort(f.loc[f.MonetaryNet > 0, 'MonetaryNet'].to_numpy())
        c = read('02_customer_concentration.csv')
        fig, ax = new(1697, 737, ncols=2)
        x = np.r_[0, np.arange(1, len(v) + 1) / len(v)]; y = np.r_[0, np.cumsum(v) / v.sum()]
        a = ax[0]
        a.plot([0, 1], [0, 1], color=GRAY, ls='--', lw=.7); a.plot(x, y, color=NAVY, lw=1.6)
        a.fill_between(x, x, y, color='#e7ecf1')
        a.xaxis.set_major_formatter(PercentFormatter(1, decimals=0)); a.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
        a.set_xlabel('Cumulative customers from lowest value'); a.set_ylabel('Cumulative positive net value')
        a.set_title('Lorenz curve', fontsize=8.5)
        a = ax[1]
        sh = 100 * c.NetValueShare.to_numpy()
        a.bar(range(len(c)), sh, width=.55, color=[TEAL] + [NAVY] * (len(c) - 1), zorder=3)
        for i, val in enumerate(sh):
            a.text(i, val + 1, f'{val:.1f}%', ha='center', va='bottom')
        a.set_xticks(range(len(c)), [f'Top {int(p)}%' for p in c.TopCustomerPct])
        a.yaxis.set_major_formatter(PercentFormatter(decimals=0)); a.set_ylim(0, 90)
        a.set_title('Value held by the largest customers', fontsize=8.5); grid(a)
        fig.subplots_adjust(left=.075, right=.985, top=.92, bottom=.15, wspace=.2)
        save(fig, 'figure_04.png')


    def fig06():
        report_style()
        p = read('03_segment_profiles.csv').set_index('CustomerSegment').loc[[s + ' Customers' for s in SEG]]
        cols = ['MedianRecency', 'MedianFrequency', 'MedianNetValue']
        z = np.log1p(p[cols].to_numpy(float)); z = (z - z.mean(0)) / z.std(0)
        fig, ax = new(1379, 571)
        im = ax.imshow(z, cmap='RdBu', vmin=-1.7, vmax=1.7, aspect='auto')
        for i in range(4):
            labels = [f'{p.MedianRecency.iloc[i]:.0f}', f'{p.MedianFrequency.iloc[i]:.0f}', f'{p.MedianNetValue.iloc[i]:,.0f}']
            for j in range(3):
                ax.text(j, i, labels[j], ha='center', va='center', fontweight='bold',
                        color='white' if abs(z[i, j]) > .9 else INK)
        ax.set_xticks(range(3), ['Median recency\ndays', 'Median purchase\ninvoices', 'Median net spend\n£'])
        ax.set_yticks(range(4), SEG); ax.tick_params(length=0)
        cb = fig.colorbar(im, ax=ax, fraction=.03, pad=.03); cb.set_label('Relative log profile'); cb.set_ticks([-1, 0, 1], labels=['−1', '0', '1'])
        fig.subplots_adjust(left=.1, right=.96, top=.97, bottom=.17)
        save(fig, 'figure_06.png')


    def fig07():
        report_style()
        p = read('03_segment_profiles.csv').set_index('CustomerSegment')
        fig, ax = new(1484, 607)
        for i, s in enumerate(SEG):
            y = 3 - i; cs = p.loc[s + ' Customers', 'CustomerSharePct']; vs = p.loc[s + ' Customers', 'NetValueSharePct']
            ax.plot([cs, vs], [y, y], color='#b8c2cb', lw=2.4, zorder=2)
            ax.plot(cs, y, 'o', color=GRAY, ms=6, zorder=3); ax.plot(vs, y, 'o', color=NAVY, ms=7, zorder=3)
            ax.text(cs, y + .2, f'{cs:.1f}%', ha='center', color=GRAY); ax.text(vs, y - .2, f'{vs:.1f}%', ha='center', va='top', color=NAVY)
        ax.set_yticks([3, 2, 1, 0], SEG); ax.set_xlim(0, 80); ax.set_ylim(-.6, 3.6)
        ax.xaxis.set_major_formatter(PercentFormatter(decimals=0))
        ax.set_xlabel('Share of all identified customers or net merchandise value'); grid(ax, 'x')
        ax.plot([], [], 'o', color=GRAY, label='Customer share'); ax.plot([], [], 'o', color=NAVY, label='Net value share')
        ax.legend(frameon=False, ncol=2, loc='lower center', bbox_to_anchor=(.5, 1.0))
        fig.subplots_adjust(left=.1, right=.98, top=.9, bottom=.17)
        save(fig, 'figure_07.png')


    def fig08():
        report_style()
        d = read('05_customer_dynamics.csv')
        order = [('Loyal', NAVY), ('Growing', TEAL), ('Cooling', GOLD)]  # Inactive has too few eligible customers
        fig, ax = new(1456, 644)
        for k, (s, col) in enumerate(order):
            g = d[(d.CustomerSegment == s + ' Customers') & d.RhythmEligible.astype(bool)]['MedianInterpurchaseDays']
            y = 2 - k; q1, med, q3, lo, hi = g.quantile([.25, .5, .75, .10, .90])
            ax.add_patch(plt.Rectangle((q1, y - .25), q3 - q1, .5, fc=col, ec=INK, lw=.8, alpha=.9))
            ax.plot([med, med], [y - .25, y + .25], color='white', lw=1.8)
            ax.plot([lo, q1], [y, y], color=INK, lw=.8); ax.plot([q3, hi], [y, y], color=INK, lw=.8)
            ax.plot([lo, lo], [y - .1, y + .1], color=INK, lw=.8); ax.plot([hi, hi], [y - .1, y + .1], color=INK, lw=.8)
            ax.text(med, y + .31, f'Median {med:g}', ha='center', fontsize=6.6)
        ax.set_yticks([2, 1, 0], [f'{s}\nn={((d.CustomerSegment == s + " Customers") & d.RhythmEligible.astype(bool)).sum():,}' for s, _ in order])
        ax.set_ylim(-.6, 2.7); ax.set_xlim(0, 220)
        ax.set_xlabel('Customer median gap between distinct purchase days\nBoxes: quartiles; whiskers: 10th to 90th percentile')
        grid(ax, 'x')
        fig.subplots_adjust(left=.12, right=.98, top=.98, bottom=.25)
        save(fig, 'figure_08.png')


    def _band(ax, x0, x1, y0a, y0b, y1a, y1b, color, alpha=.45):
        t = np.linspace(0, 1, 60); s = 3 * t ** 2 - 2 * t ** 3
        xs = x0 + (x1 - x0) * t
        ax.fill_between(xs, y0a + (y1a - y0a) * s, y0b + (y1b - y0b) * s, color=color, alpha=alpha, lw=0)


    def fig10():
        report_style()
        m = read('05_dynamics_report_metrics.csv').set_index('Metric')['Value']
        total = int(read('03_segment_profiles.csv').set_index('CustomerSegment').loc['Growing Customers', 'Customers'])
        short = int(m['Growing_Not_assessed'])
        ends = [('Reactivated', int(m['Growing_Reactivated']), GOLD), ('Increasing', int(m['Growing_Increasing']), TEAL),
                ('Stable', int(m['Growing_Stable']), NAVY), ('Declining or stopped', int(m['Growing_Declining_or_stopped']), RED)]
        full = total - short
        assert full == sum(n for _, n, _ in ends), 'Growing decomposition does not add up'
        fig, ax = new(1302, 657); ax.axis('off')
        H = 1.0; sc = H / total; gap = .04
        # column 1: all Growing
        ax.add_patch(plt.Rectangle((0, 0), .04, H, color=TEAL))
        # column 2: short vs full history
        y_short = (H + gap - short * sc, H + gap); y_full = (0, full * sc)
        ax.add_patch(plt.Rectangle((.36, y_short[0]), .03, short * sc, color=GRAY))
        ax.add_patch(plt.Rectangle((.36, y_full[0]), .03, full * sc, color=NAVY))
        _band(ax, .04, .36, H - short * sc, H, y_short[0], y_short[1], '#c5ccd3')
        _band(ax, .04, .36, 0, full * sc, y_full[0], y_full[1], '#9eabb8')
        ax.text(.2, (y_short[0] + y_short[1]) / 2, 'Short history', ha='center', va='center')
        ax.text(.2, full * sc / 2, 'Full 180 day\nhistory', ha='center', va='center')
        # column 3: outcomes
        ax.add_patch(plt.Rectangle((.72, y_short[0] + .05), .03, short * sc, color=GRAY))
        _band(ax, .39, .72, y_short[0], y_short[1], y_short[0] + .05, y_short[1] + .05, '#c5ccd3')
        ax.text(.77, y_short[0] + .05 + short * sc / 2, f'Not assessed\n{short} ({100 * short / total:.1f}%)', va='center')
        top_src = full * sc; top_dst = y_short[0] - .02; label_y = top_dst
        for k, (name, n, col) in enumerate(ends):
            h = n * sc; src0 = top_src - h; dst0 = top_dst - h
            _band(ax, .39, .72, src0, top_src, dst0, top_dst, col, .35)
            ax.add_patch(plt.Rectangle((.72, dst0), .03, h, color=col))
            label_y = min(label_y, dst0 + h / 2)  # small groups get evenly spaced labels
            ax.text(.77, label_y, f'{name}\n{n} ({100 * n / total:.1f}%)', va='center', fontsize=6.6)
            label_y -= .24
            top_src = src0; top_dst = dst0 - .05
        ax.text(0, H + gap + .05, f'Growing\n{total:,} customers', fontweight='bold', fontsize=8.5, va='bottom')
        ax.set_xlim(0, 1.02); ax.set_ylim(min(top_dst, label_y) - .02, H + .36)
        fig.subplots_adjust(left=.01, right=.99, top=.99, bottom=.01)
        save(fig, 'figure_10.png')


    def fig11(name='figure_11.png', px=1646, py=795):
        report_style()
        r = read('04_association_rules.csv'); a = r[r.Group == 'All Customers']
        fig, ax = new(px, py, ncols=2, gridspec_kw={'width_ratios': [1.25, 1]})
        net = ax[0]; net.axis('off')
        families = [(['22745', '22748', '22746'], ['POPPY\nBEDROOM', 'POPPY\nKITCHEN', 'POPPY\nLIVING'], TEAL, 1.0),
                    (['22698', '22697', '22699'], ['PINK\nREGENCY', 'GREEN\nREGENCY', 'ROSE\nREGENCY'], NAVY, 0.0)]
        for codes, labels, col, y in families:
            pos = {codes[0]: (0, y), codes[1]: (1, y + .2), codes[2]: (2, y)}
            single = a[a.AntecedentCodes.isin(codes) & a.ConsequentCode.isin(codes)]
            for row in single.itertuples():
                (x0, y0), (x1, y1) = pos[str(row.AntecedentCodes)], pos[str(row.ConsequentCode)]
                net.annotate('', (x1, y1), (x0, y0), arrowprops=dict(arrowstyle='-|>', color=col, alpha=.6, lw=1.1,
                             shrinkA=17, shrinkB=17, connectionstyle='arc3,rad=.12'))
            for c, lab in zip(codes, labels):
                x, yy = pos[c]
                net.scatter(x, yy, s=900, color='#e8f0f2', edgecolor=col, lw=1.1, zorder=3)
                net.text(x, yy, lab, ha='center', va='center', fontsize=5.3, zorder=4)
        net.set_xlim(-.35, 2.35); net.set_ylim(-.35, 1.5)
        net.set_title('Selected coordinated product families', fontsize=7.6)
        sc = ax[1]
        sc.scatter(100 * a.Confidence, a.Lift, s=a.Support * 2000, color=NAVY, alpha=.45, lw=0)
        sc.set_yscale('log'); sc.set_yticks([2, 5, 10, 20, 50], ['2', '5', '10', '20', '50']); sc.minorticks_off(); sc.xaxis.set_major_formatter(PercentFormatter(decimals=0))
        sc.set_xlabel('Rule confidence'); sc.set_ylabel('Lift on a logarithmic scale')
        sc.set_title(f'All {len(a):,} market rules', fontsize=7.6); grid(sc)
        fig.subplots_adjust(left=.02, right=.985, top=.93, bottom=.15, wspace=.25)
        save(fig, name)


    def fig12():
        report_style()
        mix = EXTRA()['fig12_mix']
        fig, ax = new(1451, 708)
        for i, s in enumerate(SEG):
            f = mix[s + ' Customers']['familiar_pct']; y = 3 - i
            ax.barh(y, f, color=NAVY, height=.75); ax.barh(y, 100 - f, left=f, color=TEAL, height=.75)
            ax.text(f / 2 if f > 8 else f + 1, y, f'{f:.1f}%', ha='center' if f > 8 else 'left', va='center',
                    color='white' if f > 8 else INK)
            ax.text(f + (100 - f) / 2, y, f'{100 - f:.1f}%', ha='center', va='center', color='white')
        ax.set_yticks([3, 2, 1, 0], SEG); ax.set_xlim(0, 100); ax.xaxis.set_major_formatter(PercentFormatter(decimals=0))
        ax.set_xlabel('Share of generated product recommendations')
        from matplotlib.patches import Patch
        ax.legend(handles=[Patch(color=NAVY, label='Previously purchased item'), Patch(color=TEAL, label='Item new to the customer')],
                  frameon=False, ncol=2, loc='lower center', bbox_to_anchor=(.5, 1.0))
        fig.subplots_adjust(left=.11, right=.97, top=.9, bottom=.16)
        save(fig, 'figure_12.png')


    def fig20():
        report_style()
        s = pd.DataFrame(EXTRA()['fig20'])
        fig, ax = new(1645, 680, ncols=2)
        a = ax[0]
        a.plot(s.days, s.loyal_decline_pct, color=RED, marker='o', ms=4, lw=1.6)
        for d, n, pct in zip(s.days, s.loyal_decline, s.loyal_decline_pct):
            if d in (60, 90, 120):
                a.text(d + 3, pct + .6, f'{n}', ha='left', fontsize=6.4)
        loyal_n = int(read('03_segment_profiles.csv').set_index('CustomerSegment').loc['Loyal Customers', 'Customers'])
        a.set_ylabel(f'Share of all {loyal_n:,} Loyal customers'); a.set_title('Loyal decline signals', fontsize=8.5)
        a.yaxis.set_major_formatter(PercentFormatter(decimals=1))
        a = ax[1]
        a.plot(s.days, s.growing_increasing_pct, color=TEAL, marker='o', ms=4, lw=1.6)
        for d, n, pct in zip(s.days, s.growing_comparable, s.growing_increasing_pct):
            below = d < 60  # first points sit close together: label them under the line
            a.text(d + (3 if below else 0), pct - 5 if below else pct + 2.5, f'n={n}', ha='left' if below else 'center', fontsize=5.6)
        a.set_ylabel('Increasing / comparable customers'); a.set_title('Growing measured increases', fontsize=8.5)
        a.yaxis.set_major_formatter(PercentFormatter(decimals=0)); a.set_ylim(0, max(s.growing_increasing_pct) + 12)
        for a in ax:
            a.set_xticks([30, 60, 90, 120, 180]); a.set_xlabel('Days in each of two comparison windows'); grid(a)
        fig.subplots_adjust(left=.085, right=.985, top=.9, bottom=.17, wspace=.3)
        save(fig, 'figure_20.png')


    # ================= Slide charts 10, 11, 12 (deck style) =================
    SEGCOL = {'Loyal': BLUE, 'Cooling': '#e69f00', 'Growing': '#009e73', 'Inactive': '#cc79a7'}


    def slide10():
        slide_style(9.5)
        c = read('02_customer_concentration.csv'); p = read('03_segment_profiles.csv').set_index('CustomerSegment')
        f = read('02_customer_features.csv'); n_pos = int((f.MonetaryNet > 0).sum()); n_all = len(f)
        exc = p.loc['Net-Zero / Net-Negative Exception']
        fig = plt.figure(figsize=(1945 / 190, 797 / 190), dpi=190)
        a = fig.add_axes([.05, .22, .14, .56])
        sh = 100 * c.NetValueShare.to_numpy()
        a.bar(range(len(c)), sh, width=.65, color=[BLUE] * (len(c) - 1) + [ORANGE])
        for i, v in enumerate(sh):
            a.text(i, v + 1.5, f'{v:.1f}', ha='center', fontweight='bold' if i == len(c) - 1 else 'normal')
        a.set_xticks(range(len(c)), [f'{int(x)}%' for x in c.TopCustomerPct]); a.set_ylim(0, 100)
        a.set_xlabel('Top x% of customers'); a.set_ylabel('Share of positive net value (%)')
        a.text(len(c) - .6, 93, f'n = {n_pos:,}', ha='right')
        fig.text(.005, .97, 'a', fontweight='bold', fontsize=13)
        fig.text(.02, .95, 'Customers with positive net value |\nShare of net value held by top x% (%)', fontweight='bold', va='top')
        b = fig.add_axes([.31, .22, .15, .56])
        for i, sname in enumerate(SEG):
            y = 3 - i; row = p.loc[sname + ' Customers']; col = SEGCOL[sname]
            b.plot([row.CustomerSharePct, row.NetValueSharePct], [y, y], color=col, lw=2.2)
            b.plot(row.NetValueSharePct, y, 'o', color=col, ms=8)
            b.plot(row.CustomerSharePct, y, 'o', mfc='white', mec=col, mew=2, ms=8)
            b.text(row.CustomerSharePct, y + .28, f'{row.CustomerSharePct:.1f}', ha='center', fontsize=9)
            b.text(row.NetValueSharePct, y - .3, f'{row.NetValueSharePct:.1f}', ha='center', va='top', fontsize=9,
                   fontweight='bold' if sname == 'Loyal' else 'normal')
            b.text(-6, y, sname, ha='right', va='center', color=col, fontweight='bold')
        b.set_yticks([]); b.set_xlim(0, 85); b.set_ylim(-.6, 3.7); b.set_xlabel('Share (%)'); b.spines['left'].set_visible(True)
        b.grid(True, axis='x', color='#dddddd', lw=.8)
        fig.text(.245, .97, 'b', fontweight='bold', fontsize=13)
        fig.text(.26, .95, 'Segments | Customer share vs\nnet value share (%)', fontweight='bold', va='top')
        fig.text(.26, .03, f'○ customer share   ● net value share (n = {n_all:,})', fontsize=9)
        t = fig.add_axes([.52, .24, .47, .6]); t.axis('off')
        fig.text(.505, .97, 'c', fontweight='bold', fontsize=13)
        fig.text(.52, .95, f'Segments | Median RFM profile (n = {n_all:,})', fontweight='bold', va='top')
        heads = ['Segment', f'Customers\n(of {n_all:,})', 'Median\nrecency (days)', 'Median\ninvoices', 'Median net\nspend (GBP)']
        xs = [0, .36, .6, .78, 1.0]; ha = ['left', 'right', 'right', 'right', 'right']
        for x, h, al in zip(xs, heads, ha):
            t.text(x, .97, h, ha=al, va='top', fontweight='bold', fontsize=9.5)
        t.set_xlim(0, 1); t.set_ylim(0, 1)
        t.plot([0, 1], [.8, .8], color='#111111', lw=1)
        rows = [(sname, p.loc[sname + ' Customers'], SEGCOL[sname]) for sname in SEG] + [('Exceptions', exc, '#111111')]
        for k, (sname, row, col) in enumerate(rows):
            y = .7 - k * .14
            vals = [sname, f'{int(row.Customers):,} ({row.CustomerSharePct:.1f}%)', f'{row.MedianRecency:.0f}',
                    f'{row.MedianFrequency:.0f}', f'{row.MedianNetValue:,.0f}']
            for j, (x, v, al) in enumerate(zip(xs, vals, ha)):
                t.text(x, y, v, ha=al, va='center', color=col if j == 0 else '#111111', fontweight='bold' if j == 0 or sname == 'Loyal' else 'normal', fontsize=9.5)
        t.plot([0, 1], [.0, .0], color='#111111', lw=1)
        fig.text(.52, .16, f'Exceptions: {int(exc.Customers)} customers with zero or negative net value,\nkept in the report but excluded from K-means ({n_all:,} - {int(exc.Customers)} = {n_pos:,}).', fontsize=9, va='top')
        save(fig, 'slide10_value_segments.png', dpi=190)


    def slide11():
        slide_style(9.5)
        m = read('05_dynamics_report_metrics.csv').set_index('Metric')['Value']
        p = read('03_segment_profiles.csv').set_index('CustomerSegment'); d = read('05_customer_dynamics.csv')
        loyal_n = int(p.loc['Loyal Customers', 'Customers']); grow_n = int(p.loc['Growing Customers', 'Customers'])
        fig = plt.figure(figsize=(2161 / 190, 826 / 190), dpi=190)
        a = fig.add_axes([.05, .3, .2, .5])
        v = [loyal_n, int(m['LoyalDeclining90']), int(m['LoyalDecliningAndOverdue'])]
        a.bar(range(3), v, width=.6, color=[BLUE, BLUE, ORANGE])
        for i, x in enumerate(v):
            a.text(i, x + 15, f'{x:,}', ha='center', fontweight='bold', color=ORANGE if i == 2 else '#111111')
        a.set_xticks(range(3), ['All Loyal', 'Spent >25% less\n(latest vs prior\n90 days)', 'Also unusually\nlong gap'], fontsize=7.8)
        a.set_yticks([]); a.spines['left'].set_visible(False)
        fig.text(.005, .95, 'a', fontweight='bold', fontsize=13); fig.text(.02, .95, 'Loyal customers | Momentum flags\n(customers)', fontweight='bold', va='top')
        b = fig.add_axes([.4, .25, .2, .55])
        items = [('Not assessed\n(<180 days history)', m['Growing_Not_assessed']), ('Reactivated', m['Growing_Reactivated']),
                 ('Increasing >25%', m['Growing_Increasing']), ('Stable', m['Growing_Stable']),
                 ('Declining\nor stopped', m['Growing_Declining_or_stopped'])]
        y = np.arange(5)[::-1]
        b.barh(y, [x for _, x in items], height=.62, color=['#009e73', '#009e73', ORANGE, '#009e73', '#009e73'])
        for yy, (lab, x) in zip(y, items):
            b.text(x + 8, yy, f'{int(x)} ({100 * x / grow_n:.1f}%)', va='center', color=ORANGE if lab.startswith('Incr') else '#111111',
                   fontweight='bold' if lab.startswith('Incr') else 'normal')
        b.set_yticks(y, [lab for lab, _ in items]); b.set_xticks([]); b.spines['bottom'].set_visible(False); b.set_xlim(0, max(x for _, x in items) * 1.35)
        fig.text(.265, .95, 'b', fontweight='bold', fontsize=13)
        fig.text(.28, .95, f'Growing customers | Decomposition of {grow_n:,}\n(customers, % of segment)', fontweight='bold', va='top')
        c = fig.add_axes([.74, .3, .22, .5])
        segs = [('Loyal', BLUE), ('Growing', '#009e73'), ('Cooling', '#e69f00')]
        for i, (sname, col) in enumerate(segs):
            g = d[(d.CustomerSegment == sname + ' Customers') & d.RhythmEligible.astype(bool)]['MedianInterpurchaseDays']
            c.bar(i, g.median(), width=.62, color=col); c.text(i, g.median() + 1.5, f'{g.median():g}', ha='center', fontweight='bold')
            c.text(i, -7, f'{sname}\nn={len(g):,}', ha='center', va='top')
        c.set_xticks([]); c.set_yticks([]); c.spines['left'].set_visible(False); c.set_ylim(0, 85)
        fig.text(.705, .95, 'c', fontweight='bold', fontsize=13); fig.text(.72, .95, 'Segments | Median days between\npurchase dates (days)', fontweight='bold', va='top')
        fig.text(.005, .02, 'Source: DM_Report ver 2.0, sec 4.2-4.3, Fig. 8, Fig. 10; recomputed from the project pipeline. c: customers with ≥4 purchase days; spread (box) in Fig. 8. Source reports no uncertainty for a, b.',
                 style='italic', fontsize=8)
        save(fig, 'slide11_dynamics.png', dpi=190)


    def slide12_bars():
        slide_style(10)
        r = read('04_association_rules.csv')
        groups = ['All Customers'] + [s + ' Customers' for s in SEG]
        v = [int(r.loc[r.Group == g, 'BasketCount'].iloc[0]) if (r.Group == g).any() else 0 for g in groups]
        fig = plt.figure(figsize=(297 / 100, 371 / 100), dpi=100)
        ax = fig.add_axes([55 / 297, 55 / 371, 185 / 297, 261 / 371]); ax.axis('off')  # same transparent margin as the deck's original picture
        y = np.arange(5)[::-1] * 1.0
        ax.barh(y, v, height=.8, color='#004aad')
        for yy, x in zip(y, v):
            inside = x > .45 * max(v)
            ax.text(x - .02 * max(v) if inside else x + .03 * max(v), yy, f'{x:,}', va='center', ha='right' if inside else 'left',
                    color='white' if inside else '#111111', fontsize=10)
        ax.set_xlim(0, max(v) * 1.02); ax.set_ylim(-.5, 4.5)
        save(fig, 'slide12_basket_bars.png', dpi=100, transparent=True)


    if True:
        fig02(); fig03(); fig04(); fig05(); fig06(); fig07(); fig08(); fig09(); fig10(); fig11(); fig12()
        fig13(); fig14(); fig16(); fig17(); fig18(); fig19(); fig20()
        fig05('slide09_k_selection.png', 1482, 612); slide06(); slide13(); slide14()
        slide10(); slide11(); fig11('slide12_rules.png'); slide12_bars()


# ============================================================================
# Run everything
# ============================================================================
STEPS = [step_01_data_preparation, step_02_descriptive_analysis, step_03_rfm_segmentation,
         step_04_product_affinity, step_05_customer_dynamics, step_06_recommendation_backtest,
         step_07_repeat_purchase_prediction, step_08_customer_actions,
         step_11_extra_report_metrics,  # before step 09: the checks read its metrics
         step_09_report_checks, step_12_final_report_figures]


def main():
    global RAW_FILE
    RAW_FILE = find_data()
    print('Data  :', RAW_FILE)
    print('Output:', OUTPUT)
    for step in STEPS:
        print(f'\n=== {step.__name__} ===', flush=True)
        start = time.time()
        step()
        print(f'--- done in {time.time() - start:.0f} s', flush=True)

    # Self-check: the counts every reader of the report will look at first.
    audit = pd.read_csv(OUTPUT / '01_data_audit.csv').set_index('Stage')['Count']
    assert audit['Raw rows'] == 1067371, 'Raw row count differs: is this the original UCI file?'
    assert audit['Identified merchandise purchase lines'] == 770495
    assert len(pd.read_csv(OUTPUT / '08_customer_intelligence.csv')) == 5838
    print('\nFinished. All tables and figures are in', OUTPUT)


if __name__ == '__main__':
    main()
