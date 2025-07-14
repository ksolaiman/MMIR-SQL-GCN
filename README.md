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

"dataset/071420250000" == random seed: 0.1237
"dataset/071320252100" == random seed: 0.1234
"dataset/112420230000" == random seed: 0.1234   // same as "dataset/071320252100"


"k_values": [50, 100],
"k_values": -1,
"k_values": 0,