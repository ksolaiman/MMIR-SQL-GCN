# MMIR-SQL-GCN

From the following repo - **/homes/ksolaima/scratch1/SimGNN/EARS_WLIR+MARS**


db_config.json // remote, local_mac_1 (2016), local_mac_2 (2021)
# remote not active anymore


Exactly: when I did P over relation all instead of P over R for the normalization of AP, actually calc_MAP_25 and calc_MAP_mac_1 gave the same result on the same data. So that's one thing—you have to be careful about how you normalize the AP.

Also, you have to be careful about K. Both of them gave the same result for K=0. At this point, none of them matches the paper result, but they're very close.

Second thing: you have to run the data after the database read. You need to read the dataset and then calculate the distance matrix after reading the dataset, not from reading from file. Reading from file causes a KeyError on the text IDs. So you have to be careful about that.

At this point, this was all I find.


-----

✅ Excellent—let’s **rewrite** your *README.md* **based on**:

⭐️ What you wrote in your note (above)
⭐️ The *set\_train\_test\_data\_and\_distance\_matrix\_from\_mmir\_ground.py* design (as we've fixed it)
⭐️ The *calc\_MAP\_25.py* style and typical usage

---

## ✅ 📌 *Updated, clear, helpful* `README.md` section:

You can copy-paste this directly.

---

# MMIR-SQL-GCN

## Overview

This repo generates **train/test splits** and **distance matrices** from the `mmir_ground` and `mmir_predicted` tables in PostgreSQL.

It also computes **Mean Average Precision (MAP)** with different normalization options.

---

## 📌 Data Source Configuration

* Database connection settings are in **`db_config.json`**.
* Supports multiple local setups:

  * `local_mac_1` (2016)
  * `local_mac_2` (2021)
  * `remote` (not active anymore — **do not use**)

✅ Always select your target DB with:

```json
{
  "database_source": "local_mac_1"
}
```

---

## 📌 Splitting and Saving Data

**Key script:**
`set_train_test_data_and_distance_matrix_from_mmir_ground.py`

✅ This prepares:

* Overall **train pool** (80%)
* **Test set** (20%)
* Separate **image/video/text train pools** and **test sets**

✅ It saves everything in:

```
dataset/
  trainpool.pkl
  testset.pkl
  image/
  video/
  text/
```

✅ *Important*:

* **Train/validation splitting** is done *later*, dynamically, on the saved **train pool**.
* **Test set is fixed** forever after initial creation.

---

## 📌 Distance Matrix Generation

✅ After defining splits, you must calculate **distance matrices** from the **DB**, not from pickled ID files:

```
set_train_test_data_and_distance_matrix_from_mmir_ground.py
```

* Calculates:

  * `ground_distance_matrix`
  * `predicted_distance_matrix`
  * `item2idx`, `idx2item` mappings
* Saves in:

```
dataset/dist_matrices/train/
dataset/dist_matrices/test/
```

✅ *Note*:

> **Always run distance matrix calculation *after* reading IDs from DB.**
> Reading old pickled ID files alone can cause **KeyError** for text IDs if data changes.

---

## 📌 MAP Calculation

**Key script:**
`calc_MAP_25.py`

✅ Calculates Mean Average Precision (MAP) given:

* ground distances
* predicted distances
* relevance threshold
* normalization method

### ⚠️ **Normalization Option**

Be careful:

* `"p_over_r"` and `"p_over_relation_all"` can give different results.
* But on *same* data, they *may* produce same MAP if chosen correctly.

✅ Typical config:

```json
{
  "normalize_ap": "p_over_r",
  "relevance_threshold": 3
}
```

✅ Watch out:

* Changing `normalize_ap` changes MAP meaning.
* Make sure to match your paper setting.

---

## 📌 About K

* `K` parameter controls cutoff for ranking.
* Be cautious: both `calc_MAP_25` and `calc_MAP_mac_1` match when `K=0`, but may differ at other values.
* Neither exactly replicates paper result without careful tuning.

---

## 📌 Recommended Workflow

✅ 1️⃣ Configure DB source in `db_config.json`
✅ 2️⃣ Run data preparation:

```
python set_train_test_data_and_distance_matrix_from_mmir_ground.py
```

* Generates splits and saves IDs
* Calculates and saves distance matrices

✅ 3️⃣ Run MAP calculation:

```
python calc_MAP_25.py
```

* Loads distance matrices from `dataset/dist_matrices/`
* Calculates MAP with chosen threshold and normalization

---

## 📌 Notes

* Always rerun **dataset read + distance matrix** after any DB update.
* Don't rely on old pickled IDs if underlying DB has changed.
* Ensure consistency of threshold and normalization for comparability.


---

Below is the **updated, ready-to-use `README.md`** in pure Markdown:

---

````markdown
# MMIR Retrieval Evaluation Toolkit

This repository provides tools to:

✅ Connect to a PostgreSQL MMIR database  
✅ Prepare train/validation/test splits with distance matrices  
✅ Evaluate retrieval performance using Mean Average Precision (MAP)  
✅ Optionally generate and save interpolated Precision-Recall (PR) curves

---

## 📌 Main Scripts Overview

### 🗂️ `set_train_test_data_and_distance_matrix.py`
- Connects to the specified database using `config.json`
- Reads raw MMIR data
- Splits data into:
  - Train set
  - Validation set
  - Test set
  - Modality-specific splits (Image / Video / Text)
- Computes:
  - Ground-truth distance matrix
  - Predicted distance matrix
- Saves:
  - Dataset splits (IDs, mappings)
  - Distance matrices

---

### 🗂️ `calc_MAP.py`
- Loads saved distance matrices
- Computes ranking-based retrieval MAP
- Supports:
  - Adjustable relevance threshold
  - Custom AP normalization methods
  - Single or multiple k-values for MAP@k
  - Optional PR-curve generation and saving
- All settings are controlled via `config.json`

---

### 🗂️ `utils.py`
- Helper functions for:
  - Writing dataset splits
  - Saving distance matrices
  - General file I/O

---

### 🗂️ `connection_to_database.py`
- Centralized PostgreSQL connection logic
- Reads DB connection parameters from `config.json`

---

### 📌 Explanation of Fields

* `db_host`: Which DB profile to use (must match one of the entries below)
* `remote`, `local_mac_1`, `local_mac_2`: Connection settings for different environments
* `database_source`: Selects which profile to use for DB connection
* `relevance_threshold`: Distance threshold for defining "relevant" items in evaluation
* `normalize_ap`: AP normalization mode

  * `"p_over_r"`: divides by retrieved relevant hits
  * `"p_over_all"`: divides by all ground-truth relevant items
* `save_pr`: If true, saves interpolated PR-curves
* `k_values`: **Supports both single int or list** for k cutoffs

  * Example single int:

    * `0` → MAP over all retrieved items
    * `50` → MAP\@50 only
    * `-1` → special case that evaluates `[50, all]`
  * Example list:

    * `[50, 0]` → MAP\@50 and MAP\@all
* `dataset_dir`: Directory to save/load dataset splits and distance matrices

---

## 📌 Recommended Workflow

### 1️⃣ **Configure `config.json`**

* Choose the correct `database_source` profile
* Set `relevance_threshold`, `normalize_ap`, and `k_values` as needed

✅ *Example single k:*

```json
"k_values": 0
```

✅ *Example multiple k values:*

```json
"k_values": [50, 0]
```

---

### 2️⃣ **Prepare Dataset Splits and Distance Matrices**

Run:

```bash
python set_train_test_data_and_distance_matrix.py
```

✅ Reads directly from the database
✅ Generates and saves:

* Train / validation / test splits
* Modality-specific splits (image, video, text)
* Ground-truth and predicted distance matrices

✅ Output structure:

```
dataset/
  071420250000/
    testset.pkl
    trainset.pkl
    validset.pkl
    test_distance_matrix.pkl
    train_distance_matrix.pkl
    valid_distance_matrix.pkl
```

*Note*: Always rerun this if the DB data changes!

---

### 3️⃣ **Evaluate MAP**

Run:

```bash
python calc_MAP.py
```

✅ Loads saved matrices
✅ Computes MAP for all specified k-values in `k_values`
✅ Supports:

* Single k (int)
* Multiple k values (list)
* Special case:

  * `-1` → evaluates both MAP\@50 and MAP\@all

✅ Saves MAP summary results to:

```
prcurve/EARS/<timestamp>/map_results.txt
```

✅ Also saves PR-curve .pkl files if `save_pr` is true:

```
prcurve/
  EARS/
    <timestamp>/
      Text_Text_precision.pkl
      Text_Text_recall.pkl
      ...
```

---

### 4️⃣ **Analyze or Plot PR Curves**

Use:

```
pr_curve_generation.ipynb
```

✅ Loads saved .pkl files
✅ Plots mean interpolated Precision-Recall curves
✅ Supports both 11-point and 1000-point interpolation curves

---

## 📌 Notes on Precision-Recall Interpolation

* Interpolation granularity is customizable in code:

  * 11-point (standard IR-style)
  * 1000-point (fine-grained)

* Definition:

  > At each recall level r, precision is the max precision observed at any recall ≥ r.

✅ Standard Information Retrieval practice.

---

## 📌 Normalization Options for AP

* `"p_over_r"`:

  * Sum of precisions at hits / number of true positives retrieved
  * Reflects retrieval *efficiency*
* `"p_over_all"`:

  * Sum of precisions at hits / total relevant items in ground truth
  * Reflects retrieval *coverage*

⚠️ These can yield **different MAP scores**.
✅ Choose carefully to match your experimental setup.

---

## 📌 Archival Note

Older scripts like `calc_MAP_mac_1.py` and `calc_MAP_mac_1_8pm.py` are **deprecated**.
✅ Use **`calc_MAP.py`** as the final, maintained evaluation script.

---

## 📌 Final Note

✅ Always ensure consistency between:

* Database source
* Threshold
* Normalization mode
* k\_values
* Dataset splits

for reproducible, comparable MAP evaluation results.

---

## 📌 Graph Generation Output Structure (simplified)

Under your `dataset/` folder:

- **hargs/**: Raw serialized *Hierarchical Attributed Relational Graphs (HARGs)*
- **harg_turned_simgnn_graphs/**: Flattened, SimGNN-ready JSON graph pairs

✅ **gold** = built from gold-annotated properties (targets for training)  
✅ **noisy** = built from extracted (noisy) properties (inputs for training)

✅ Both contain separate `label_vocab.json` and `edge_type_vocab.json` for consistent mapping.

---

## ✅ Generation Details

- Controlled via `config.json`:
  - `"noisy": true/false` → select predicted vs gold property extraction
  - `"save_hargs": true/false` → toggle saving full HARG objects
  - `"save_simgnn_graphs": true/false` → toggle saving flattened SimGNN graph pairs
  - `"harg_dir"`, `"simgnn_graph_dir"` → specify subdirectory names

✅ Example usage:

```bash
python create_harg.py --noisy False   # generate gold graphs
python create_harg.py --noisy True    # generate noisy graphs

---

## 📌 CED Computation

The `ced.py` module implements:

- Hierarchical Attributed Relational Graph (HARG) construction
- Cost model and edit distance calculation (Content Edit Distance, CED)
- Alignment and edge cost strategies (e.g. Munkres assignment)

✅ Used to generate **pairwise CED** between graph representations.


> For MUQNOL, i do not need to use ced.py, but for general other datasets we may use this to calculate dist_matrix. 
> currently my SQL query calculates the dist_matrix.
---

```markdown
# FemmIR Graph Pair Dataset Construction

This repository supports the construction of graph datasets and distance supervision pipelines for multimodal entity matching using the FemmIR framework. It is designed to:

- Construct **HARG graphs** for each item using structured attributes.
- Compute **Content Edit Distance (CED)** between multimodal items.
- Generate **SimGNN-compatible graph pair samples** using these distances.
- Ensure **balanced sampling** across modalities for robust training.

---

## 📁 Directory Structure

```

dataset/
├── config.json
├── trainpool.pkl
├── testset.pkl
├── image/           # image\_trainpool.pkl, image\_testset.pkl
├── text/            # text\_trainpool.pkl, text\_testset.pkl
├── video/           # video\_trainpool.pkl, video\_testset.pkl
├── dist\_matrices/
│   ├── ground\_distance\_matrix.pkl
│   ├── noisy\_distance\_matrix.pkl
│   └── ...
└── femmir\_pair\_lists\_<...>/
├── all\_positive\_pairs\_0.pkl
├── all\_negative\_pairs\_0.pkl
└── ...

````

---

## 📌 FemmIR Pair Generation

The `create_femmir_pairs.py` script supports:

✅ Generating training pairs for SimGNN using CED distance supervision  
✅ Sampling **positive/negative pairs** stratified by modality  
✅ Saving **query-indexed dictionaries** for reuse:

```python
{ qid: [(cid, ced, modality_pair), ...] }
````

This format supports fast loading and flexible resampling during training.

---

### 🔧 Controlled via `config.json`

```json
{
  "dataset_dir": "dataset/071420250000",
  "relevance_threshold": 3,
  "pair_save_dir": "femmir_pair_lists_NOS_mod_seed=42_NOS=10",
  ...
}
```

Saved files:

* `all_positive_pairs_0.pkl`, `all_negative_pairs_0.pkl`, ...
* One file per batch of \~1000 items for efficient memory use

---

## ⚙️ Stratified Train/Validation Splits

Defined in `set_train_test_data_and_distance_matrix.py`, using:

```python
stratified_train_validation_split_by_modality(...)
```

### 🧪 Balanced Sampling Modes

| Mode               | Behavior                                         |
| ------------------ | ------------------------------------------------ |
| `undersample=True` | Match the smallest modality (e.g. text)          |
| `oversample=True`  | Upsample smaller modalities to match the largest |
| `balanced=False`   | Keep natural modality proportions                |

Applies to both **train/val split** and **k-fold cross-validation**.

---

## ⚠️ Pairwise Explosion Control

To avoid computing the full O(N²) matrix:

✅ Pair sampling is done **per query**, limited to a few targets
✅ Samples are **modality-stratified**
✅ Supports **parallel batch processing** using `joblib.Parallel` (e.g. `n_jobs=8`)

This keeps memory and compute usage under control.

---

## 🔄 Pair Reuse Strategy

To avoid recomputing:

* Intermediate dictionaries are saved for each batch.
* Later SimGNN training can load and combine them per fold.

Example:

```python
with open("femmir_pair_lists/all_positive_pairs_0.pkl", "rb") as f:
    pos_dict = pickle.load(f)
```

---

## ✅ Status

* [x] Distance matrix computed (`noisy_distance_matrix.pkl`)
* [x] Query-target pair lists generated by CED thresholds
* [x] Stratified sampling supports both balancing and k-folds
* [x] Pair files saved for SimGNN training


---

**Important Note (for MuQNOL dataset):**

> ✅ For MuQNOL retrieval tasks, we only need the *dist_matrix* (already precomputed and saved, it's a proxy for CED).
> ✅ Updated in new create_harg.py and ced.py, we are not creating 
    - train_pairs/ and 
    - test_pairs/ 
for SIMGNN in these files, yet.

---


## 📌 Configuration

All settings are in a single **`config.json`** file. Specially, for values of different keys:

```
"dataset/071420250000" == random seed: 0.1237
"dataset/071320252100" == random seed: 0.1234
"dataset/112420230000" == random seed: 0.1234   // same as "dataset/071320252100"


"k_values": [50, 100],
"k_values": -1,
"k_values": 0,
```
---


## 📌 Future Work

* Add support for **graph-level augmentations**
* Experiment with **different GNN backbones** (beyond SimGNN)
* Analyze impact of **sampling strategies** on retrieval performance

---
