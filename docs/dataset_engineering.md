## Dataset Engineering & Data Quality

Building the StackExchange portion of the DPO dataset turned out to be a data engineering problem rather than a modeling problem.

### Domain-specific ingestion

My initial approach loaded the entire `HuggingFaceH4/stack-exchange-preferences` dataset and applied keyword-based filtering to extract AI and technical content. This introduced significant noise and required downloading and processing hundreds of irrelevant StackExchange domains.

While exploring the dataset, I discovered that individual StackExchange sites could be loaded directly using `data_dir`. I redesigned the ingestion pipeline to load only relevant domains.

```python
ALLOWED_DOMAINS = [
    "ai",
    "biology",
    "chemistry",
    "cs",
    "datascience",
    "economics",
    "softwareengineering",
    "stats",
    "physics",
]
```

```python
load_dataset(
    "HuggingFaceH4/stack-exchange-preferences",
    data_dir=f"data/{domain}.stackexchange.com",
)
```

This significantly reduced noise and eliminated several hours of unnecessary downloading and preprocessing.

---

### Data quality debugging

During dataset construction, the pipeline initially generated only **112 preference pairs**, which suggested either poor source quality or overly restrictive filtering.

Instead of modifying the model or collecting additional data, I instrumented the data pipeline with rejection statistics.

```text
Reasons skipped:

few_answers: 0
score_gap: 198844
answer_length: 1
```

This revealed that a single hyperparameter, `MIN_SCORE_GAP = 10`, was filtering out nearly **99.8%** of the candidate examples.

After analyzing the score distribution, I recalibrated the threshold to `MIN_SCORE_GAP = 4`, increasing the dataset size from **112 → 600** preference pairs without changing the source data.

---

### Final quality gates

Each StackExchange pair must satisfy:

* At least two candidate answers
* `MIN_SCORE_GAP = 4`
* `MIN_ANSWER_LENGTH = 80`
* `chosen != rejected`
* `best_score > worst_score`

---

### Key takeaway

> Most machine learning problems are data engineering and data quality problems before they become model problems.

Building instrumentation and auditing data pipelines often produces larger improvements than immediately iterating on model architectures.
