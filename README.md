# BTNet — нейронные сети на основе биномиального дерева для оценки опционов

Реализация нейросетевых архитектур для оценки европейских и американских опционов пут, структура которых напрямую выведена из биномиальной модели Кокса–Росса–Рубинштейна (CRR). Проект реализует подход из статьи:

> Шорохов С.Г. Оценка финансовых деривативов нейронной сетью на основе биномиального дерева // International Journal of Open Information Technologies. – 2024. – Т. 12, № 5. – С. 108–114.

## Ключевая идея

Вместо того чтобы обучать произвольную нейросеть на ценах опционов, архитектура строится аналитически: каждый шаг обратного хода биномиального дерева — это один слой сети с фиксированными весами. Это даёт интерпретируемость «из коробки»: веса имеют прямой финансовый смысл (риск-нейтральные вероятности, множители дисконтирования).

- **Европейский опцион** → обратный ход реализован как последовательность свёрточных слоёв с фильтром размера 2
- **Американский опцион** → вместо свёртки — maxout-слои, реализующие `max(держать, исполнить)`

## Структура проекта

```
btnn-bs/
├── btnn_bs/                    # Python-пакет
│   ├── layers.py               # ConvLayer, DenseLayer, MaxoutLayer
│   ├── model_european.py       # BTNetEuropean
│   ├── model_american.py       # BTNetAmerican
│   ├── tree.py                 # MyIBT_CRR — референсное биномиальное дерево
│   ├── analytics.py            # bs_put_price, american_put_prices_binomial
│   ├── training.py             # train_BTNet
│   ├── plotting.py             # визуализация результатов
│   └── quantlib/               # интеграция с QuantLib (опционально)
│       └── __init__.py
├── btnn-bs-torch-v2.ipynb      # основной ноутбук
├── BTNN_BS_Eur_v4.ipynb        # европейский опцион, ранняя версия
├── pyproject.toml
├── environment.yml
└── requirements.txt
```

## Установка

```bash
git clone https://github.com/username/btnn-bs.git
cd btnn-bs

pip install -e .
```

Для работы с QuantLib (верификация цен):

```bash
pip install -e ".[quantlib]"
```

## Использование

### Европейский опцион пут

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

### Американский опцион пут

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

### Верификация через QuantLib

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

## Экспериментальные результаты

Стандартный набор параметров для воспроизведения результатов:

| Параметр | Значение |
|----------|----------|
| S₀ | 0.5 |
| T | 1.0 |
| r | 0.05 |
| σ | 0.25 |
| n (шагов дерева) | 9 |
| Диапазон страйков K | [0.25, 0.75] |
| Размер обучающей выборки | 500 |
| Эпох обучения | 200 |

**Европейский пут** — loss после обучения ~10⁻⁶, ошибка относительно формулы Блэка–Шоулза на уровне численного шума.

**Американский пут** — loss ~10⁻⁵, MAE относительно QuantLib CRR (~500 шагов) порядка 10⁻³–10⁻⁴ в зависимости от глубины дерева сети.

## Архитектуры слоёв

| Класс | Описание |
|-------|----------|
| `DenseLayer` | Полносвязный слой с настраиваемой активацией и возможностью инициализации весов |
| `ConvLayer` | 1D-свёртка с фильтром размера 2 (один шаг дисконтирования) |
| `MaxoutLayer` | Свёртка + линейный слой с поэлементным `max` (один шаг American backward induction) |

## CI

Три параллельных задания на GitHub Actions:

- **lint** — `ruff check btnn_bs/`
- **check** — установка пакета, проверка импортов (CPU torch)
- **notebook** — выполнение `btnn-bs-torch-v2.ipynb`, артефакт сохраняется 7 дней

## Зависимости

- **PyTorch** — основной фреймворк
- **NumPy, SciPy** — численные вычисления
- **Matplotlib** — визуализация
- **QuantLib** — верификация цен (опционально)

## Автор

Петров Артем Евгеньевич, НКНбд-01-22  
**Научный руководитель**: к.ф.н., доцент Шорохов С.Г.  
Российский университет дружбы народов имени Патриса Лумумбы

## Лицензия

MIT
