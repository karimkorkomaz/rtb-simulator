# Dataset Evaluation: iPinYou, Criteo, and Avazu for RTB Thesis

**Verification date for all live-URL checks:** 2026-08-23 (all `curl`/fetch tests below were run on this date; treat any status as a snapshot, not a permanent guarantee).

**Purpose:** Determine which public advertising dataset(s) can support a thesis requiring bid-level auction simulation (win prices, bid prices, second-price outcomes), as opposed to CTR-prediction-only impression/click records.

---

## 1. Summary comparison table

| Dataset | Live download today? | Registration required? | License | Compressed / uncompressed size | Bid-level win prices? | Best source to use |
|---|---|---|---|---|---|---|
| **iPinYou RTB** | Partial — official UCL mirror is dead; Baidu Pan link loads but is high-friction/unverifiable; Hugging Face and GitHub-pipeline routes work | No (HF mirror); yes for Baidu Pan (Baidu account) | Stated as "non-commercial use" on official site; no formal SPDX license; GitHub processing pipeline is Apache-2.0 (code only, not data) | Raw ZIP unconfirmed size; HF mirror ~252 MB / ~19.5M rows | **Yes** — `biddingprice` (own bid) and `payingprice` (market/winning price) columns, plus floor price | GitHub pipeline (`wnzhang/make-ipinyou-data`) + Hugging Face `reczoo/iPinYou_x1` mirror |
| **Criteo Display Advertising Challenge (Kaggle DAC)** | Yes — direct Criteo-hosted link confirmed live (302 → Azure Blob, 200 OK) | No, if using Criteo's direct link; Yes, if using Kaggle | CC BY-NC-SA 4.0 (non-commercial, share-alike, attribution) | 4,576,820,670 bytes (~4.26 GiB) compressed; ~11 GB uncompressed (commonly cited, not independently re-verified); ~45.8M train rows | **No** — anonymized/hashed integer + categorical features only, no price field | Direct Criteo link: `https://go.criteo.net/criteo-research-kaggle-display-advertising-challenge-dataset.tar.gz` |
| **Criteo 1TB Click Logs** | Yes — page reachable; download now via Hugging Face `criteo/CriteoClickLogs` per Criteo's own page | No (HF, public dataset card) | CC BY-NC-SA 4.0 (non-commercial) | ~1 TB uncompressed across 24 daily files; exact compressed size unconfirmed | **No** — same 13 int + 26 categorical schema as DAC, click label only | Exceeds "laptop-comfortable" by a wide margin regardless of license |
| **Criteo Attribution Modeling for Bidding (CAMB)** | Yes — direct link confirmed live (302 → Azure Blob, 200 OK, Content-Length verified) | No | CC BY-NC-SA 4.0 (non-commercial) | 653,128,946 bytes (~623 MB) compressed / ~2.4 GB uncompressed (per Criteo's own page); ~16.5M rows | **Partial** — has `cost`, explicitly stated by Criteo to be a **transformed (obfuscated) version** of the real second-price paid, not the raw win price; no bid request/response log | `https://go.criteo.net/criteo-research-attribution-dataset.zip` |
| **Avazu CTR Prediction (Kaggle)** | Yes — Kaggle competition data page returns HTTP 200 and is reachable; requires Kaggle login to actually download | Yes — Kaggle account + rules acceptance | Kaggle Competition Rules: academic research/education use permitted; redistribution outside competition participants prohibited (no open license, no CC grant) | train.gz reported ~6+ GB compressed, ~40,428,967 rows, 10 days; test.gz 1 day (exact byte sizes unconfirmed — Kaggle page not machine-fetchable, see note below) | **No** — click/no-click label + 22 categorical/anonymized context fields, no price field of any kind | Kaggle only; Hugging Face `reczoo/Avazu_x4` mirror exists as a secondary route |

---

## 2. iPinYou Global RTB Bidding Algorithm Competition Dataset

### 2.1 Downloadability today

- **Official site:** `https://contest.ipinyou.com/` — confirmed **live**, HTTP 200 (checked 2026-08-23). It is a static archive page, not an active service.
- **Links found on the official page:**
  - `http://data.computational-advertising.org` — the UCL mirror historically cited by nearly every paper using this dataset. **Confirmed DEAD**: DNS resolution fails outright (`curl: (6) Could not resolve host`). Do not rely on this URL; every paper/blog post that lists it as the download source is now stale.
  - `http://pan.baidu.com/s/1kTwX2mF` — a Baidu Netdisk share link from the original release. Returns HTTP 200 for the general Baidu domain, but Baidu Pan share links are known to silently expire or throttle without warning, and full-speed download typically requires a Baidu account (and historically a China-based phone number / VIP membership for reasonable speed). **Access status: unconfirmed / high friction** — the 200 response only proves Baidu's site is up, not that this specific decade-old share link still serves the file.
  - `http://arxiv.org/pdf/1407.7073v2.pdf` — the benchmarking paper (not the data).
- **GitHub processing pipeline:** `https://github.com/wnzhang/make-ipinyou-data` is live and actively documents the expected raw file (`ipinyou.contest.dataset.zip`), but the repo itself does not host the data — it expects you to already have the raw ZIP from the UCL mirror (dead) or another source, then symlink it in and run `make all` to produce ~14 GB of processed output.
- **Working mirror found:** `https://huggingface.co/datasets/reczoo/iPinYou_x1` — reachable, appears to require no login for download, ~252 MB in CSV format (auto-convertible to Parquet on HF), ~19.5M rows, states it covers "all three seasons training datasets and leaderboard testing datasets" (final held-out test data remains withheld by iPinYou, as in the original competition).
- **Other mirrors seen but not independently verified:** `https://www.kaggle.com/datasets/lastsummer/ipinyou` (page loads, but its content — size, completeness, upload provenance — could not be extracted through available tooling; treat as **unconfirmed**) and an OpenDataLab listing at `https://opendatalab.com/OpenDataLab/iPinYou/download` (page loads, HTTP 200, but registration requirements and completeness could not be confirmed through available tooling).
- **Bottom line:** the canonical academic mirror is dead. The most trustworthy currently-working route is the Hugging Face `reczoo/iPinYou_x1` mirror combined with the `wnzhang/make-ipinyou-data` repo for schema/processing logic, cross-checked against the original benchmarking paper for field semantics.

### 2.2 Licensing for academic use

- The official site states only: **"The data is made available for non-commercial use."** No formal license identifier (no CC, no SPDX tag) is attached to the dataset itself.
- The `wnzhang/make-ipinyou-data` GitHub repository is Apache-2.0 licensed, but that license covers the **processing code**, not the underlying data.
- **Verification status: partially unconfirmed.** No detailed terms-of-use document (redistribution rights, attribution requirements, commercial-use boundary specifics) was found beyond the single sentence above. For a thesis, "non-commercial use" almost certainly covers academic research, but there is no explicit redistribution or publication clause to quote to a committee — flag this gap in the thesis itself.
- **Canonical citation** (confirmed from the paper itself):
  > Weinan Zhang, Shuai Yuan, Jun Wang, Xuehua Shen. "Real-Time Bidding Benchmarking with iPinYou Dataset." UCL Technical Report, 23 July 2014. arXiv:1407.7073.
  
  A companion dataset-description paper also exists:
  > Hairen Liao, Lingxiao Peng, Zhenchuan Liu, Xuehua Shen. "iPinYou Global RTB Bidding Algorithm Competition Dataset." Proceedings of the Eighth International Workshop on Data Mining for Online Advertising (ADKDD 2014). ACM. DOI: 10.1145/2648584.2648590.

### 2.3 Size

- Raw archive (`ipinyou.contest.dataset.zip`): size not independently confirmed (dead mirror, could not download to measure).
- Processed pipeline output (per `make-ipinyou-data` README): ~14 GB total across all campaigns after `make all`.
- Hugging Face `reczoo/iPinYou_x1` mirror: ~252 MB, ~19.5M rows (CSV; convertible to Parquet). This is comfortably within a laptop's working set.
- Per the benchmarking paper (Table 3, confirmed from primary source): ~64.7M total bid records, ~15.4M impressions, 11,557 clicks, 935 conversions in the training portion; ~4.1M impressions / 3,230 clicks / 318 conversions in test; 9 advertiser campaigns; ~1.24M RMB total spend. Season 2 (June 2013) and Season 3 (October–November 2013) share the same 24-field schema and are the seasons normally used in research (Season 1 has a different, incompatible schema per multiple secondary sources — **unconfirmed against a primary source**, flagged for follow-up).

### 2.4 Schema — CRITICAL: this is the only dataset with genuine auction economics

Confirmed from the primary paper (arXiv:1407.7073, Table 1) via ar5iv rendering:

| Field (paper's terminology) | Meaning |
|---|---|
| Bid ID | Unique identifier for a bid record |
| Timestamp | Bid request time |
| Log type | Impression / click / conversion marker |
| iPinYou ID | User identifier (hashed) |
| User-Agent, IP, Region, City | Request context (IP partially masked) |
| Ad exchange | Which exchange the auction ran on |
| Domain, URL | Publisher page context (hashed/anonymized) |
| Ad slot ID, width, height, visibility, format | Ad placement details |
| **Ad slot floor price** | Reserve price — "no bid lower than the floor price could win auctions" |
| Creative ID | Creative served |
| **Bidding price** | **iPinYou's own submitted bid for this auction** |
| **Paying price** | **The actual settlement price — described in the paper as "the highest bid from competitors, also called market price and auction winning price."** This is the win price needed for second-price auction simulation. |
| Key page URL | Landing page reference |
| Advertiser ID | Campaign/advertiser identifier |
| User tags | Audience segment tags |

Several fields are hashed or otherwise anonymized before release (noted in the paper with asterisks), but the **bidding price / paying price / floor price triad is intact and unredacted**, which is what makes this dataset unique among the three: it is genuinely usable to reconstruct second-price auction outcomes (won/lost, margin between own bid and market price) rather than only click labels.

### 2.5 Supports

Bid-level auction simulation: **yes**, uniquely among the three datasets evaluated here. Also supports CTR and conversion-rate modeling as a byproduct, since click/conversion logs share the same key structure.

### 2.6 Prior academic use

Confirmed as the standard RTB benchmark in the computational advertising literature:
- Zhang, Yuan, Wang, Shen (2014) — the introducing benchmark paper itself, arXiv:1407.7073.
- Liao, Peng, Liu, Shen (2014) — ADKDD dataset paper, ACM DOI 10.1145/2648584.2648590.
- Widely used in subsequent bid-optimization and RL-for-RTB papers, e.g. "Real-Time Bidding by Reinforcement Learning in Display Advertising" (arXiv:1701.02490) and "Managing Risk of Bidding in Display Advertising" (arXiv:1701.02433), both of which explicitly benchmark on iPinYou. (Confirmed these papers exist and reference iPinYou via search results; full-text auction-simulation methodology was not re-verified line-by-line — flagged as **secondary confirmation only**.)

---

## 3. Criteo datasets (four variants evaluated — do not conflate them)

Criteo AI Lab hosts multiple, structurally different datasets under one umbrella site (`https://ailab.criteo.com/ressources/`, confirmed live). These are frequently confused in blog posts. Distinguishing them is critical for this thesis.

### 3.1 Criteo Display Advertising Challenge (Kaggle DAC) — CTR-only, no bidding data

- **Downloadability:** Two independent routes both confirmed:
  1. Criteo's own direct link (no Kaggle account needed): `https://go.criteo.net/criteo-research-kaggle-display-advertising-challenge-dataset.tar.gz` — confirmed live via `curl -I` following the redirect: HTTP 302 → `https://criteostorage.blob.core.windows.net/criteo-research-datasets/kaggle-display-advertising-challenge-dataset.tar.gz`, which returned HTTP 200 with `Content-Length: 4576820670` (~4.26 GiB) on 2026-08-23.
  2. The original Kaggle competition page `https://www.kaggle.com/c/criteo-display-ad-challenge` (requires Kaggle account + rules acceptance).
  - The legacy blog post URL `http://labs.criteo.com/2014/02/kaggle-display-advertising-challenge-dataset/` returns **HTTP 403 Forbidden** as of 2026-08-23 — this specific old URL, cited in many papers' footnotes, is dead. Use the `ailab.criteo.com` / `go.criteo.net` route instead.
- **Licensing:** Confirmed from `https://ailab.criteo.com/kaggle-contest-dataset-now-available-academic-use/`: **CC BY-NC-SA 4.0** — "reproduce and Share the Licensed Material, in whole or in part, for NonCommercial purposes only," attribution to Criteo required, derivative works must carry a BY-NC-SA-compatible license. This is a clean, explicit academic-use grant, more permissive in spirit for thesis use than routing through Kaggle's competition-specific rules (which restrict redistribution to competition participants only). **Recommendation: cite and use the Criteo direct link, not the Kaggle competition page, as the licensing basis.**
- **Size:** Compressed 4,576,820,670 bytes (~4.26 GiB), confirmed via HTTP header. Uncompressed size (~11 GB) and row count (~45,840,617 train rows) are widely and consistently cited across independent sources (Criteo's own historical blog description, multiple GitHub reproductions) but were **not independently re-measured** by extracting the archive — marked **unconfirmed at the byte level, high-confidence from convergent secondary sources**.
- **Schema (confirmed, consistent across primary and secondary sources):** first column = binary click label; 13 integer count-features (`I1`–`I13`); 26 categorical features (`C1`–`C26`), each hashed to 32 bits for anonymization. **No price field of any kind, no bid ID, no auction identifier.** This is purely a per-impression click-label dataset.
- **Supports:** CTR prediction only. Cannot support auction/bid simulation.
- **Prior academic use:** Extremely well established as the standard CTR benchmark — used in Field-aware Factorization Machines (Juan, Zhuang, Chin, Lin, RecSys 2016), DeepFM, Wide & Deep, Deep & Cross Network (Wang et al., arXiv:1708.05123), and dozens of subsequent CTR architecture papers. This is the single most-cited dataset of the three for pure CTR modeling.

### 3.2 Criteo 1TB Click Logs (a.k.a. Criteo Terabyte dataset) — CTR-only, larger version of DAC

- **Downloadability:** Page confirmed live (`https://ailab.criteo.com/download-criteo-1tb-click-logs-dataset/`, HTTP 200). Per Criteo's own page, the dataset is now distributed via Hugging Face: `https://huggingface.co/datasets/criteo/CriteoClickLogs`. No registration required for a public HF dataset card, though downloading ~1 TB requires substantial bandwidth and storage.
- **Licensing:** CC BY-NC-SA 4.0, confirmed on the page (same license family as DAC and CAMB below).
- **Size:** 24 daily files covering 24 days of Criteo traffic; Criteo's own materials describe it as "the world's largest truly public ML dataset" at ~1 TB / ~4 billion events (this framing is quoted from Criteo's own resources page, not independently re-measured). Exact compressed size and per-file breakdown were **not confirmed** — the fetched pages did not surface a byte-level table.
- **Schema:** Same structure as DAC — 1 click label, 13 integer + 26 categorical (hashed) features. **No price/bid fields.**
- **Fit for a laptop-scale solo thesis:** Explicitly **not recommended** — at ~1 TB this exceeds "comfortable working set" by roughly two orders of magnitude versus DAC. Only worth considering if the thesis specifically needs terabyte-scale throughput/systems benchmarking (e.g., DLRM-style training-infrastructure work), which is out of scope for an auction-simulation-focused RTB thesis.
- **Prior academic use:** Used in large-scale systems papers (e.g., DLRM/MLPerf-adjacent benchmarking work); less common in CTR-modeling-accuracy papers, which typically use the smaller DAC instead.

### 3.3 Criteo Attribution Modeling for Bidding (CAMB) — the "bidding" dataset that is NOT full bid-level data

This is the dataset most likely to be confused with a true RTB bid-log dataset because of its name. It is not one.

- **Downloadability:** Confirmed live. Direct link `https://go.criteo.net/criteo-research-attribution-dataset.zip` returns HTTP 302 → `https://criteostorage.blob.core.windows.net/criteo-research-datasets/criteo_attribution_dataset.zip`, confirmed HTTP 200 with `Content-Length: 653128946` (~623 MB) on 2026-08-23. No registration required.
- **Licensing:** CC BY-NC-SA 4.0 ("Non-commercial purposes only"), confirmed on `https://ailab.criteo.com/criteo-attribution-modeling-bidding-dataset/`.
- **Size:** 653,128,946 bytes (~623 MB) compressed (confirmed via HTTP header); ~2.4 GB uncompressed, ~16.5 million impression rows over 30 days (per Criteo's own page — uncompressed size and row count not independently re-measured, but consistent with the page's own stated figures).
- **Schema (confirmed from Criteo's own page):** `timestamp`, `uid`, `campaign`, `conversion`, `conversion_timestamp`, `conversion_id`, `attribution`, `click`, `click_pos`, `click_nb`, **`cost`**, **`cpo`**, `time_since_last_click`, `cat1`–`cat9` (9 anonymized contextual categorical features).
- **The critical caveat, stated explicitly by Criteo itself:** `cost` is described on Criteo's own page as corresponding to "the second highest bid submitted to each auction" (i.e., conceptually the win price) but is **"a transformed version"** of the real price — Criteo obfuscates the actual monetary value before release, and `cpo` (cost-per-order) is likewise transformed. There is **no bid request log, no bid response log, no explicit "your bid" vs. "market price" pair, and no confirmation that the transformation preserves auction dynamics (e.g., relative ordering, currency scale) in a way suitable for realistic auction simulation.**
- **Supports:** Attribution modeling and conversion modeling with a proxy cost signal. **Does not support realistic second-price auction simulation** the way iPinYou's `biddingprice`/`payingprice` pair does — you get one obfuscated cost number per impression, not the two-sided bid/market-price structure needed to simulate win/loss outcomes under alternative bidding strategies.
- **Canonical citation (confirmed):**
  > Eustache Diemert, Julien Meynet, Pierre Galland, Damien Lefortier. "Attribution Modeling Increases Efficiency of Bidding in Display Advertising." AdKDD & TargetAd Workshop, KDD 2017, Halifax, NS, Canada.
- **Prior academic use:** Used in attribution-modeling and some RTB-adjacent optimization papers (e.g., "Real-time Bidding campaigns optimization using attribute selection," arXiv:1910.13292, and "Joint Online Learning and Decision-making via Dual Mirror Descent," arXiv:2104.09750, both reference this dataset per search results — **not independently verified line-by-line, secondary confirmation only**).

### 3.4 Other Criteo datasets checked (for completeness, not recommended for this thesis)

- **Criteo Sponsored Search Conversion Log** (`https://ailab.criteo.com/criteo-sponsored-search-conversion-log-dataset/`, confirmed live): 90 days of Criteo Predictive Search click-to-conversion logs, 6.4 GB uncompressed / 310 MB compressed, CC BY-NC-SA 4.0, download `http://go.criteo.net/criteo-research-search-conversion.tar.gz` (link format consistent with the two confirmed-live links above but not independently HEAD-checked). **Conversion logs only — no auction/bid data.**
- **Criteo Uplift Modeling Dataset:** Listed on the resources page (`https://ailab.criteo.com/criteo-uplift-prediction-dataset/`) but not fetched in detail — out of scope (uplift/causal inference, not CTR or RTB auction modeling). **Not evaluated in depth; flagged for the reader if uplift modeling becomes relevant.**
- **CriteoPrivateAd** (`https://huggingface.co/datasets/criteo/CriteoPrivateAd`, arXiv:2502.12103): A newer (2025) dataset explicitly built for "learning bidding models," CC-BY-SA 4.0, 10M–100M rows, 80+ fields (`id`, `user_id`, `display_order`, `publisher_id`, `campaign_id`, click/sale delay arrays, `is_visit`, `nb_sales`, `is_clicked`, `is_click_landed`, plus constrained/unconstrained feature groups for differential-privacy research). At time of checking, the Hugging Face dataset viewer reported a `FileSystemError`, and **no win-price or bid-price field was found in the schema** — its "bidding" framing appears to be about privacy-constrained bidding-signal design (e.g. aggregated reports, delayed reporting) rather than exposing raw auction win prices. **Verification status: incomplete** — the dataset viewer error prevented full schema confirmation; worth a manual follow-up check if the thesis pivots toward privacy-preserving bidding, but not recommended as a primary source given the access error and apparent absence of price fields.

---

## 4. Avazu Click-Through Rate Prediction (Kaggle)

### 4.1 Downloadability today

- **Kaggle competition page** `https://www.kaggle.com/c/avazu-ctr-prediction/data` returns HTTP 200 (confirmed 2026-08-23) — the competition page itself is reachable, but the tooling available for this evaluation could not render Kaggle's JavaScript-driven page content (multiple fetch attempts, including via a reader proxy, returned only the page title). **Actual download requires a Kaggle account and acceptance of competition rules**; this was not exercised (no credentials in this environment), so the presence/absence of an active "Download" button (Kaggle sometimes disables downloads for very old closed competitions) is **unconfirmed** — flagged explicitly rather than assumed.
- **Secondary mirror:** `https://huggingface.co/datasets/reczoo/Avazu_x4` — confirmed reachable (HTTP 200). This is a processed/reformatted version used by the RecZoo/BARS benchmarking framework, not a raw untouched copy; treat as a convenience mirror, not the authoritative source for a thesis appendix.
- **No independent, license-clean, non-Kaggle direct download was found.** Unlike the Criteo datasets, Avazu Inc. does not appear to host its own copy outside Kaggle.

### 4.2 Licensing for academic use

- Kaggle's general Competition Rules (confirmed from Kaggle's own documentation/community answers, not Avazu-specific but standard boilerplate applied to competitions of this era) state: **"you may access and use the Competition Data for the purposes of the Competition, participation on Kaggle Website forums, academic research and education, and other non-commercial purposes"**, but **"you agree not to transmit, duplicate, publish, redistribute or otherwise provide or make available the Data to any party not participating in the Competition."**
- This means: academic research use is explicitly permitted, but **redistributing the raw files (e.g., committing them to a public thesis-code GitHub repo, or shipping them in a reproducibility archive) is explicitly prohibited** under these terms. A thesis can describe results and even publish derived/aggregated statistics, but should not bundle the raw Avazu files.
- **Verification status:** The exact competition-specific rules text for `avazu-ctr-prediction` itself was **not directly fetched** (Kaggle's rules page did not render through available tooling); the language quoted above is Kaggle's standard competition-data clause, confirmed from a Kaggle community Q&A page as generally applicable, and should be treated as **high-confidence but not a verbatim quote from this specific competition's rules page**. Recommend the pipeline/thesis-writing stage re-verify by logging into Kaggle directly before finalizing the thesis's data-availability statement.
- No formal academic citation/DOI exists for the Avazu dataset itself (it is a company-provided Kaggle competition dataset, not a published dataset paper) — cite it as "Avazu Inc., Click-Through Rate Prediction, Kaggle, 2014" plus the competition URL.

### 4.3 Size

- Per multiple independent, mutually consistent secondary sources (GitHub reproductions, blog writeups): `train.gz` contains **40,428,967 rows**, described as "over 6GB" compressed; covers **10 days** of click-through data (an 11th day is held out as the unlabeled test set). 22–24 fields depending on how `hour`'s decomposed components are counted.
- **Verification status: unconfirmed at the byte-exact level.** The authoritative Kaggle data page (which lists exact `train.gz`/`test.gz` byte sizes) could not be rendered by available tooling. The 40,428,967-row figure and "10 days" framing recur consistently across independent third-party sources, giving reasonable confidence, but this is **secondary-source corroboration, not primary confirmation** — flag this in the thesis methodology section and re-confirm once you have Kaggle access.
- At ~6+ GB compressed / commonly cited ~7 GB uncompressed, this fits comfortably on a personal laptop.

### 4.4 Schema

Reconstructed from multiple independent, mutually consistent secondary sources (GitHub reproductions of the Kaggle data dictionary; the primary Kaggle page could not be directly fetched by available tooling — **verification status: corroborated but not independently confirmed from the primary source**):

| Field | Meaning |
|---|---|
| `id` | Ad/impression identifier |
| `click` | Binary label: 0 = no click, 1 = click |
| `hour` | Timestamp in `YYMMDDHH` format |
| `C1` | Anonymized categorical variable |
| `banner_pos` | Banner position on page |
| `site_id`, `site_domain`, `site_category` | Publisher site identifiers (anonymized/hashed) |
| `app_id`, `app_domain`, `app_category` | App identifiers, for in-app inventory |
| `device_id`, `device_ip`, `device_model`, `device_type`, `device_conn_type` | Device/user context (anonymized) |
| `C14`–`C21` | Anonymized categorical variables, high cardinality, undocumented semantics |

**No price field of any kind — no bid price, no win price, no floor price, no auction identifier.** This is purely a labeled click/no-click dataset with anonymized contextual features, structurally similar in spirit to Criteo's DAC (categorical-heavy, label + context) but for mobile in-app/web inventory rather than Criteo's display network.

### 4.5 Supports

CTR prediction only. **Cannot support any auction or bid-price modeling** — there is nothing in the schema to simulate a second-price auction against.

### 4.6 Prior academic use

Extremely well established alongside Criteo DAC as one of the two standard CTR benchmarks: used in Field-aware Factorization Machines (Juan et al., RecSys 2016 — FFM won the Avazu Kaggle competition itself), and cited as a standard baseline dataset in DeepFM, Deep & Cross Network, AutoInt, and most subsequent CTR-architecture papers.

---

## 5. Recommendation

### For the RTB thesis's core requirement — bid-level win prices for auction simulation:

**Use iPinYou as the primary dataset.** It is the only one of the three (and, to my knowledge from this research, one of very few public datasets anywhere) that exposes a genuine two-sided bid/market-price pair (`biddingprice` and `payingprice`, plus `slotprice` floor) at the individual-auction level, confirmed directly from the paper that introduced it. This is what makes second-price auction simulation, win-rate curves, and bid-shading experiments possible at all. Criteo's CAMB dataset uses the word "bidding" in its name but explicitly ships an obfuscated, transformed cost proxy rather than real bid/win prices — it is not a substitute.

**Caveat to build into the thesis timeline immediately:** the canonical download mirror (`data.computational-advertising.org`, cited by nearly every paper that uses this dataset) is confirmed dead as of 2026-08-23. Do not spend week 1 trying that URL. Go directly to:
1. `https://huggingface.co/datasets/reczoo/iPinYou_x1` (working, no-login mirror, ~252 MB) for the actual data, and
2. `https://github.com/wnzhang/make-ipinyou-data` for the field-mapping/processing logic and campaign-ID breakdown, and
3. arXiv:1407.7073 as the schema and citation anchor.

Treat the Baidu Pan link, the Kaggle `lastsummer/ipinyou` mirror, and the OpenDataLab listing as unverified fallbacks only — test them personally before committing to them, since none could be fully verified through available tooling in this evaluation.

Also flag explicitly in the thesis: the license basis for iPinYou is a single sentence ("non-commercial use") with no formal redistribution/attribution clause — weaker footing than Criteo's explicit CC BY-NC-SA grant. This is a real risk (not a blocker, but worth a paragraph in the thesis's data-availability section, and worth emailing `dsp-competition@ipinyou.com` early in week 1 to get anything in writing, since that contact address is still listed on the live site).

### Fallback — unconditionally downloadable today, no dependency risk:

**Criteo Display Advertising Challenge (DAC), via Criteo's own direct link:**
`https://go.criteo.net/criteo-research-kaggle-display-advertising-challenge-dataset.tar.gz`

This was independently confirmed live today (HTTP 302 → Azure Blob Storage, HTTP 200, `Content-Length: 4576820670` bytes), requires no registration, carries an explicit, unambiguous CC BY-NC-SA 4.0 academic-use grant quoted directly from Criteo's own site, and is comfortably laptop-sized (~4.3 GB compressed). It cannot support auction simulation — no price fields exist — so if used as the primary dataset, the thesis scope must narrow to CTR prediction rather than full RTB auction simulation, or DAC must be paired with a synthetic/simulated auction layer (e.g., generating floor/competing-bid prices analytically) built on top of it, which is an engineering decision for the pipeline stage, not a data-acquisition one.

**Second-line fallback (if DAC access were ever to break):** Avazu, via Kaggle (`https://www.kaggle.com/c/avazu-ctr-prediction/data`), reachable today but gated behind a Kaggle account and non-redistribution terms — same CTR-only limitation as DAC, no bid data, slightly more access friction due to the account/terms requirement.

### Bottom line

- If the thesis's central contribution depends on realistic auction/bid simulation: **commit to iPinYou now**, verify the Hugging Face mirror downloads cleanly in week 1, and get the licensing sentence confirmed in writing from iPinYou's contact email as insurance.
- If the thesis can be reframed as CTR prediction only (no auction economics): **Criteo DAC via the direct Criteo link is the lowest-risk choice** — fully verified live today, clearest license of any dataset evaluated, and laptop-friendly.
- Do **not** rely on the Criteo 1TB Click Logs dataset (too large for a laptop) or the Criteo Attribution Modeling for Bidding dataset (its "bidding" framing is misleading for this thesis's needs — no true bid/win price pair).

---

## 6. Open verification gaps (explicit, for the record)

These items could not be confirmed from a primary source with the tooling available in this session and should be re-checked before the thesis's data-availability chapter is finalized:

1. iPinYou raw archive exact byte size (mirror was dead; could not download to measure).
2. Whether iPinYou Season 1 truly uses an incompatible schema versus Seasons 2–3 (repeated in secondary sources, not confirmed against the primary paper's full text).
3. Full text of iPinYou's licensing terms beyond the single "non-commercial use" sentence — no formal terms-of-use document was located.
4. Avazu's exact `train.gz`/`test.gz` byte sizes and the exact wording of the `avazu-ctr-prediction`-specific Kaggle rules page (JS-rendered page could not be fetched by available tooling).
5. Whether Kaggle's download button is still active for the (long-closed) Avazu competition, versus data-only/no-download-for-non-participants status.
6. Criteo 1TB Click Logs' exact compressed size and per-file breakdown.
7. CriteoPrivateAd's full schema (Hugging Face dataset viewer returned a `FileSystemError` at time of checking) and whether it contains any price signal at all.
8. Baidu Pan iPinYou share link's actual current validity (domain-level 200 response does not confirm the specific decade-old share link still resolves to a file).

---

## Sources

- iPinYou official site: https://contest.ipinyou.com/
- iPinYou dataset PDF: https://contest.ipinyou.com/ipinyou-dataset.pdf
- iPinYou processing pipeline: https://github.com/wnzhang/make-ipinyou-data
- iPinYou benchmarking paper: https://arxiv.org/abs/1407.7073 (https://arxiv.org/pdf/1407.7073)
- iPinYou dataset paper (ACM): https://dl.acm.org/doi/10.1145/2648584.2648590
- iPinYou Hugging Face mirror: https://huggingface.co/datasets/reczoo/iPinYou_x1
- iPinYou Kaggle mirror (unverified): https://www.kaggle.com/datasets/lastsummer/ipinyou
- iPinYou OpenDataLab listing (unverified): https://opendatalab.com/OpenDataLab/iPinYou/download
- RL for RTB using iPinYou: https://arxiv.org/pdf/1701.02490
- Risk-managed bidding using iPinYou: https://arxiv.org/pdf/1701.02433
- Criteo AI Lab resources index: https://ailab.criteo.com/ressources/
- Criteo DAC academic-use license page: https://ailab.criteo.com/kaggle-contest-dataset-now-available-academic-use/
- Criteo DAC description page: https://ailab.criteo.com/display-advertising-challenge-criteo/
- Criteo DAC direct download: https://go.criteo.net/criteo-research-kaggle-display-advertising-challenge-dataset.tar.gz
- Criteo Kaggle competition (alternate route): https://www.kaggle.com/c/criteo-display-ad-challenge/data
- Criteo 1TB Click Logs page: https://ailab.criteo.com/download-criteo-1tb-click-logs-dataset/
- Criteo 1TB Click Logs Hugging Face: https://huggingface.co/datasets/criteo/CriteoClickLogs
- Criteo Attribution Modeling for Bidding (CAMB) page: https://ailab.criteo.com/criteo-attribution-modeling-bidding-dataset/
- Criteo CAMB direct download: https://go.criteo.net/criteo-research-attribution-dataset.zip
- Criteo CAMB Kaggle mirror: https://www.kaggle.com/datasets/sharatsachin/criteo-attribution-modeling
- Criteo Sponsored Search Conversion Log page: https://ailab.criteo.com/criteo-sponsored-search-conversion-log-dataset/
- CriteoPrivateAd paper: https://arxiv.org/html/2502.12103v1 (PDF: https://arxiv.org/pdf/2502.12103)
- CriteoPrivateAd Hugging Face: https://huggingface.co/datasets/criteo/CriteoPrivateAd
- Avazu Kaggle competition data page: https://www.kaggle.com/c/avazu-ctr-prediction/data
- Avazu Kaggle competition rules: https://www.kaggle.com/competitions/avazu-ctr-prediction/rules
- Avazu Hugging Face mirror: https://huggingface.co/datasets/reczoo/Avazu_x4
- Kaggle general competition-data rules discussion: https://www.kaggle.com/questions-and-answers/129958 and https://www.kaggle.com/general/9893
- RecSysDatasets Avazu conversion notes: https://github.com/RUCAIBox/RecSysDatasets/blob/master/conversion_tools/usage/Avazu.md
- Field-aware Factorization Machines paper (Criteo + Avazu benchmark): https://www.csie.ntu.edu.tw/~cjlin/papers/ffm.pdf
- Deep & Cross Network paper (Criteo benchmark): https://arxiv.org/pdf/1708.05123
