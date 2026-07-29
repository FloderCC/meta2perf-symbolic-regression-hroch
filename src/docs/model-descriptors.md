# Model descriptors and complexity

The following notation is adopted throughout the document:

- **n**: number of training samples  
- **m**: number of input features  
- **c**: number of output classes  

We report three complementary views:

1. the model-specific structural-size proxy used by the code,
2. training and inference complexities (Big-O),
3. other qualitative descriptors (regularization, robustness, representational capacity),
4. and finally the **encoding scheme** used in the meta-dataset.

---

## 1. Structural-size proxy

The raw value below is transformed with the natural logarithm before it is added to the meta-dataset as `Processing Units Number`. It is not a hardware-unit, FLOP, latency, or comparable parameter count.

**Table 1 – Implemented structural-size proxy per model.**

| Model | Formula | Explanation |
|-------|---------|-------------|
| LR | `c × m + 1` | Implemented linear-model proxy. |
| Ridge | `c × m + 1` | Implemented linear-model proxy. |
| SGD | `c × m + 1` | Implemented linear-model proxy. |
| Perceptron | `c × m + 1` | Implemented linear-model proxy. |
| LinearSVC | `c × m + 1` | Implemented linear-model proxy. |
| DT | `2^{max_depth} - 1` | Each node performs a split decision. |
| ET | `2^{max_depth} - 1` | Same single-tree proxy as DT. |
| RF | `n_estimators × (2^{max_depth} - 1)` | Tree ensembles; each estimator follows the same structure as DT. |
| ETs | `n_estimators × (2^{max_depth} - 1)` | Tree ensembles; each estimator follows the same structure as DT. |
| AB | `n_estimators` | Implemented one-unit-per-estimator proxy. |
| HGB | `max_iter × (2^{max_depth} - 1)` | Each iteration adds one tree to the ensemble. |
| Bagging | `n_estimators × 2^{160}` | Each estimator runs independently; default DT(`max_depth = 160`). |
| GaussianNB | `c × m × 2` | Learns mean and variance per feature and class. |
| BernoulliNB | `c × m + 1` | Implemented BernoulliNB proxy. |
| MLP | `Σ hidden_layer_sizes` | Each neuron is a processing unit. |
| DNN | `Σ hidden_layer_sizes` | Each neuron is a processing unit. |

---

## 2. Training and inference complexities

**Table 2 – Training and inference complexities of the evaluated models regarding `n` (samples) and `m` (features).**

| Model | `train_complexity` | `inf_complexity` |
|-------|--------------------|------------------|
| LR | `O(n × m)` | `O(m)` |
| Ridge | `O(n × m²)` | `O(m)` |
| SGD | `O(n × m)` | `O(m)` |
| Perceptron | `O(n × m)` | `O(m)` |
| LinearSVC | `O(n × m)` | `O(m)` |
| DT | `O(n × m × log m)` | `O(log m)` |
| ET | `O(n × m × log m)` | `O(log m)` |
| RF | `O(n × m × log m)` | `O(log m)` |
| ETs | `O(n × m × log m)` | `O(log m)` |
| AB | `O(n × m)` | `O(m)` |
| HGB | `O(m)` | `O(log m)` |
| Bagging | `O(n × m × log m)` | `O(log m)` |
| GaussianNB | `O(n × m)` | `O(m)` |
| BernoulliNB | `O(n × m)` | `O(m)` |
| MLP | `O(n × m)` | `O(n × m)` |
| DNN | `O(n × m)` | `O(n × m)` |


---

## 3. Other model descriptors

**Table 3 – Other model descriptors.**

| Model | `model_type` | `regularization` | `rob_outliers` | `repr_capacity` |
|-------|---------------|------------------|----------------|-----------------|
| LR | Linear | L1, L2, or None (depends on `penalty`) | No | Low |
| Ridge | Linear | L2 (if `alpha ≠ 0.1`), None (if `alpha = 0.1`) | No | Low |
| SGD | Linear | L2 (default) | Medium | Medium |
| Perceptron | Linear | L1, L2, or ElasticNet (depends on `penalty`) | Medium | Medium |
| LinearSVC | Linear | L2 | No | Medium |
| DT | Tree-based | None | Yes | Medium |
| ET | Tree-based | None | Yes | Medium |
| RF | Ensemble | None | Yes | High |
| ETs | Ensemble | None | Yes | High |
| AB | Ensemble | None | No | Medium |
| HGB | Ensemble | L2 (if `l2_regularization ≠ 0.0`), None (if `l2_regularization = 0.0`) | Yes | High |
| Bagging | Ensemble | None | Yes | Medium |
| GaussianNB | Probabilistic | None | No | Low |
| BernoulliNB | Probabilistic | L2 (if `alpha ≠ 0.1`), None (if `alpha = 0.1`) | No | Low |
| MLP | Neural Network | L2 (if `alpha ≠ 0.0001`), None (if `alpha = 0.0001`) | No | Very High |
| DNN | Neural Network | L2 (if `alpha ≠ 0.0001`), None (if `alpha = 0.0001`) | No | Very High |

---

## 4. Encoding scheme

All qualitative descriptors above were **numerically encoded** in the meta-dataset as follows. These compact ordinal codes are modeling heuristics: equal spacing and monotonic effects have not been validated.

### 4.1 Model type

- Linear → **0**  
- Probabilistic → **1**  
- Tree-based → **2**  
- Ensemble → **3**  
- Neural Network → **4**

### 4.2 Training complexity (function of n and m)

- `O(m)` → **0**  
- `O(n × m)` → **1**  
- `O(n × m × log(m))` → **2**  
- `O(n × m²)` → **3**

### 4.3 Inference complexity (function of n and m)

- `O(log(m))` → **0**  
- `O(m)` → **1**  
- `O(n × m)` → **2**

### 4.4 Regularization mechanism code

- None → **0**  
- Laplace → **1**  
- L1 → **2**  
- L2 → **3**  
- ElasticNet → **4**

### 4.5 Robustness to outliers

- No → **0**  
- Medium → **1**  
- Yes → **2**

### 4.6 Representational capacity

- Low → **0**  
- Medium → **1**  
- High → **2**  
- Very High → **3**
