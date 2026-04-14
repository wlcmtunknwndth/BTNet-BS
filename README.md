# BTNet — Binomial-Tree Neural Networks for Option Pricing
# BTNet — Нейронные сети на основе биномиального дерева для оценки опционов

Neural network architectures for pricing European and American put options, whose structure is derived analytically from the Cox–Ross–Rubinstein (CRR) binomial model.  
Нейросетевые архитектуры для оценки европейских и американских опционов пут, структура которых аналитически выведена из биномиальной модели Кокса–Росса–Рубинштейна (CRR).

Based on / На основе статьи:
> Шорохов С.Г. Оценка финансовых деривативов нейронной сетью на основе биномиального дерева // International Journal of Open Information Technologies. – 2024. – Т. 12, № 5. – С. 108–114.

---

## Key idea / Ключевая идея

Instead of training a generic network on option prices, the architecture is built analytically: each backward-induction step of the binomial tree becomes one network layer with financially meaningful fixed weights (risk-neutral probabilities, discount factors).  
Вместо того чтобы обучать произвольную нейросеть на ценах опционов, архитектура строится аналитически: каждый шаг обратного хода биномиального дерева — это один слой с весами, имеющими прямой финансовый смысл (риск-нейтральные вероятности, множители дисконтирования).

- **European / Европейский** — backward induction as a stack of size-2 conv layers / обратный ход как последовательность свёрточных слоёв с фильтром размера 2
- **American / Американский** — maxout layers implementing `max(hold, exercise)` / maxout-слои, реализующие `max(держать, исполнить)`

---

## Project structure / Структура проекта

```
btnn-bs/
├── btnn_bs/                    # Python package / Python-пакет
│   ├── layers.py               # ConvLayer, DenseLayer, MaxoutLayer
│   ├── model_european.py       # BTNetEuropean
│   ├── model_american.py       # BTNetAmerican
│   ├── tree.py                 # MyIBT_CRR — reference binomial tree / референсное дерево
│   ├── analytics.py            # bs_put_price, american_put_prices_binomial
│   ├── training.py             # train_BTNet
│   ├── plotting.py             # result visualisation / визуализация результатов
│   └── quantlib/               # QuantLib integration, optional / интеграция с QuantLib
│       └── __init__.py
├── btnn-bs-torch-v2.ipynb      # main notebook / основной ноутбук
├── BTNN_BS_Eur_v4.ipynb        # European-only, earlier version / европейский, ранняя версия
├── pyproject.toml
├── environment.yml
└── requirements.txt
```

---

## Installation / Установка

Clone and install in editable mode:  
Клонируйте репозиторий и установите пакет в режиме разработки:

```bash
git clone https://github.com/username/btnn-bs.git
cd btnn-bs
pip install -e .
```

For QuantLib price verification (optional):  
Для верификации цен через QuantLib (опционально):

```bash
pip install -e ".[quantlib]"
```

---

## Usage / Использование

### European put / Европейский опцион пут

```python
import numpy as np
from btnn_bs import BTNetEuropean, bs_put_price, train_BTNet

S0, T, r, sig, n_dim = 0.5, 1.0, 0.05, 0.25, 9

model = BTNetEuropean(n_dim, S0, sig, T, t0=0.0, r=r)

K_train = np.random.uniform(0.25, 0.75, 500).reshape(-1, 1)
prices_train = bs_put_price(S0, K_train, T, r, sig).reshape(-1, 1)

train_BTNet(model, K_train, prices_train, epochs=200)

K_test = np.array([[0.4], [0.5], [0.6]])
predictions = model.predict(K_test)
```

### American put / Американский опцион пут

```python
from btnn_bs import BTNetAmerican, american_put_prices_binomial, train_BTNet

model = BTNetAmerican(n_dim, S0, sig, T, t0=0.0, r=r)

prices_train = np.array([
    american_put_prices_binomial(S0, k, T, r, sig, n=100)
    for k in K_train.flatten()
]).reshape(-1, 1)

train_BTNet(model, K_train, prices_train, epochs=200)
predictions = model.predict(K_test)
```

### QuantLib verification / Верификация через QuantLib

```python
from btnn_bs import run_quantlib_benchmark, error_stats, print_comparison_table

ql_res = run_quantlib_benchmark(
    S0, K_test, T, r, sig,
    amer_crr_steps=500,
    include_baw=True,
)

print_comparison_table(
    american={
        "BTNetAmerican": error_stats(predictions.flatten(), ql_res.ql_american_crr),
    }
)
```

---

## Experimental results / Экспериментальные результаты

Standard parameter set for reproducibility:  
Стандартный набор параметров для воспроизведения результатов:

| Parameter / Параметр | Value / Значение |
|---|---|
| S₀ | 0.5 |
| T | 1.0 |
| r | 0.05 |
| σ | 0.25 |
| n (tree depth / глубина дерева) | 9 |
| Strike range / Диапазон страйков K | [0.25, 0.75] |
| Training set size / Размер обучающей выборки | 500 |
| Epochs / Эпох обучения | 200 |

**European / Европейский** — training loss ~10⁻⁶, error vs Black–Scholes at the level of numerical noise.  
**European / Европейский** — loss после обучения ~10⁻⁶, ошибка относительно формулы Блэка–Шоулза на уровне численного шума.

**American / Американский** — training loss ~10⁻⁵, MAE vs QuantLib CRR (500 steps) in the range 10⁻³–10⁻⁴ depending on tree depth.  
**American / Американский** — loss ~10⁻⁵, MAE относительно QuantLib CRR (~500 шагов) порядка 10⁻³–10⁻⁴ в зависимости от глубины дерева.

---

## Layer reference / Описание слоёв

| Class / Класс | Description / Описание |
|---|---|
| `DenseLayer` | Linear layer with configurable activation and weight initialisation / Линейный слой с настраиваемой активацией и инициализацией весов |
| `ConvLayer` | 1D convolution with kernel size 2 — one discounting step / 1D-свёртка с фильтром 2 — один шаг дисконтирования |
| `MaxoutLayer` | Convolution + linear branch with element-wise `max` — one American backward-induction step / Свёртка + линейная ветка с поэлементным `max` — один шаг American backward induction |

---

## CI

Three parallel jobs on GitHub Actions:  
Три параллельных задания на GitHub Actions:

| Job | What it does / Что делает |
|---|---|
| **lint** | `ruff check btnn_bs/` |
| **check** | Installs package, verifies imports (CPU torch) / Устанавливает пакет, проверяет импорты |
| **notebook** | Executes `btnn-bs-torch-v2.ipynb`, keeps artifact 7 days / Выполняет ноутбук, артефакт хранится 7 дней |

---

## Dependencies / Зависимости

- **PyTorch** — deep learning framework / основной фреймворк
- **NumPy, SciPy** — numerical computation / численные вычисления
- **Matplotlib** — visualisation / визуализация
- **QuantLib** — price verification, optional / верификация цен, опционально

---

## Mathematical basis / Математическая основа

CRR price move factors:  
Множители движения цены CRR:
```
u = exp(σ√Δt)
d = exp(−σ√Δt)
```

Risk-neutral probabilities / Риск-нейтральные вероятности:
```
πᵤ = (exp(rΔt) − d) / (u − d)
π_d = 1 − πᵤ
```

European backward step / Обратный ход для европейского опциона:
```
V(i,j) = exp(−rΔt) · [πᵤ · V(i+1,j+1) + π_d · V(i+1,j)]
```

American backward step / Обратный ход для американского опциона:
```
V(i,j) = max{ exp(−rΔt) · [πᵤ · V(i+1,j+1) + π_d · V(i+1,j)],  K − S(i,j) }
```

---

## Author / Автор

Петров Артем Евгеньевич, НКНбд-01-22  
**Научный руководитель / Supervisor**: к.ф.н., доцент Шорохов С.Г.  
Российский университет дружбы народов имени Патриса Лумумбы

## License / Лицензия

MIT
