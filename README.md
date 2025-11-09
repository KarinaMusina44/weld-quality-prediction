# weld-quality-prediction
## Backlog 
### ```2025-10-15``` — Kick-off Meeting 
* Created the Git repository and discussed project organization.
* Decided that each member will explore the dataset individually (preprocessing + PCA) on their own branch, to become familiar with the data and the problem.
* Next meeting scheduled in one week ```2025-10-22```: 
    * Share results and decide which ML approaches to try.
    * Merge the preprocessing versions (final merging strategy to be decided next week).
    * Assign ML methods to each member.
    * Define the testing pipeline and evaluation metrics.

--- 
---

### ```2025-10-22``` - Data Preprocessing & Technical Setup 

#### Preprocessing

##### Data Cleaning and Missing Values

* Replace `'N'` with `NaN`.
* If a value is given as an *interval*, take its **mean**.
* If a value is given as `"< a"`, replace it with **a / 2**.
* Remove the suffix `"(Hv)"` + create a binary flag column `is_hv`. (just in case)
* Drop columns with **more than 60% missing values**.
* For columns with **40–60% missing values**, identify the cause and apply a context-specific treatment.
* Convert relevant columns to **categorical** types.

---

#### Target Selection

* *Yield strength* and *UTS* are **strongly correlated** → choose *Yield strength* (larger sample size).
* Note a **negative correlation** between *Yield/UTS* and *Elongation/Reduction*.
* Justify target selection with a **correlation analysis** (matrix + PCA before training). (and with the litterature: Cool's thesis )

---

#### PCA

* Perform a first PCA including physical properties to visualize variance and relationships among features.

---

#### Evaluation Metric

* Define a common evaluation metric across all models (still to be decided).

---

#### Technical Organization

##### File Structure

* Create a dedicated folder **`utils/`** to centralize helper functions.
* Keep notebooks focused on **pipeline logic** (function calls only).


##### Functions to Implement (in Pipeline Order)

1. **`import_data(path)`**

   * Load the dataset.
   * Clean column names → English, lowercase, no accents/spaces (use underscores).
   * Replace `'N'` with `NaN`.
   * Load column names from `config.json`.

2. **`handle_intervals(df)`**

   * Manage interval values and `"<value"` cases (replace with value / 2).
   * Remove temporary columns after cleaning.

3. **`handle_nitrogen(df)`**

   * Replace outlier nitrogen values with `NaN`.

4. **`handle_hardness(df)`**

   * Keep only the first hardness value.
   * Optionally add a binary flag column.

5. **`drop_non_informative_columns(df)`**

   * Drop columns with > 60% missing values.
   * Investigate columns with 40–60% missing data before deletion.
   * Remove  `weld_id`.

6. **`drop_not_chosen_target(df)`**

   * Remove non-selected targets. 
   * Justify with correlation matrix + initial PCA.

7. **Feature Engineering**

   * To be documented after a short literature review.

8. **Train/Test Split**

9. **Outlier Handling**

10. **Encoding (One-Hot)**

11. **Imputation & Scaling**

12. **Collinearity Management & PCA**

---

#### Notebook Organization

##### Notebook 1 — Preprocessing

* **Introduction:** context + objectives.
* **Sections:**

  1. Data exploration / diagnostics
  2. Preprocessing pipeline

##### Notebook 2 — Model Training

* Model development and evaluation.

---

#### Next Steps

* Clearly distribute preprocessing responsibilities among team members.
* Explore **semi-supervised learning** approach (unlabeled data).
* Propose **feature-engineering** ideas.
* Address **collinearity** (PCA + correlation matrix).
* Decide on **evaluation metrics** for model comparison.


### ```2025-10-28``` - Tasks distribution
* Sarah  => 1-6
* Albane => 7-8
* Karina => 9-10
* Eliott => 11-12

