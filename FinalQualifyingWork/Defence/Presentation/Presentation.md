---
marp: true
theme: default
paginate: true
style: |
  :root {
    --ink: #1f2937;
    --muted: #64748b;
    --blue: #2f6f9f;
    --blue-dark: #1e3a5f;
    --blue-soft: #edf5fb;
    --teal: #5d9b96;
    --teal-soft: #eef7f6;
    --line: #d7e2ec;
    --paper: #ffffff;
    --wash: #f7f9fc;
  }
  section {
    font-family: "Arial", "Helvetica", sans-serif;
    font-size: 23px;
    color: var(--ink);
    background: var(--wash);
    padding: 42px 58px 38px 58px;
  }
  section::before {
    content: none !important;
    display: none !important;
  }
  section::after {
    color: #8a97a8;
    font-size: 17px;
    right: 24px;
    bottom: 18px;
  }
  h1 {
    font-size: 39px;
    line-height: 1.12;
    color: var(--blue-dark);
    margin-bottom: 22px;
  }
  h2 {
    font-size: 32px;
    line-height: 1.12;
    color: var(--blue-dark);
    margin: 0 0 22px;
  }
  h2::after {
    content: "";
    display: block;
    width: 86px;
    height: 4px;
    margin-top: 10px;
    background: linear-gradient(90deg, #6fa8cc, #9fc8c4);
    border-radius: 4px;
  }
  strong { color: var(--blue); }
  code {
    background: #eef4fa;
    color: #0b4f83;
    border-radius: 4px;
    padding: 0 4px;
  }
  ul, ol { font-size: 23px; line-height: 1.34; }
  li { margin-bottom: 8px; }
  table {
    width: 100%;
    font-size: 20px;
    border-collapse: collapse;
    background: var(--paper);
    box-shadow: 0 8px 20px rgba(15, 23, 42, 0.07);
  }
  th {
    background: var(--blue-soft);
    color: var(--blue-dark);
    font-weight: 700;
  }
  th, td {
    border: 1px solid var(--line);
    padding: 8px 10px;
  }
  img {
    max-height: 330px;
    display: block;
    margin: 0 auto;
    background: #ffffff;
    border: 1px solid #d7e1ec;
    border-radius: 6px;
    padding: 8px;
    box-shadow: 0 10px 24px rgba(15, 23, 42, 0.08);
  }
  section.title {
    background: #eef6fb;
    color: var(--ink);
    padding: 70px 68px;
  }
  section.title h1 { color: var(--blue-dark); font-size: 43px; max-width: 820px; }
  section.title h1::after {
    content: "";
    display: block;
    width: 130px;
    height: 5px;
    margin-top: 24px;
    background: linear-gradient(90deg, #6fa8cc, #9fc8c4);
    border-radius: 5px;
  }
  section.title strong { color: var(--blue); }
  section.title::after { color: #91a3b7; }
  .subtitle { font-size: 24px; line-height: 1.42; margin-top: 30px; color: #45566d; }
  .note { font-size: 19px; color: var(--muted); margin-top: 12px; }
  .two { display: grid; grid-template-columns: 1fr 1fr; gap: 28px; align-items: start; }
  .box {
    background: var(--paper);
    border: 1px solid #d7e1ec;
    border-radius: 8px;
    padding: 17px 20px;
    box-shadow: 0 10px 22px rgba(15, 23, 42, 0.07);
  }
  .metric {
    font-size: 32px;
    color: var(--blue);
    font-weight: 700;
  }
  .warning {
    background: var(--teal-soft);
    border-left: 6px solid var(--teal);
    padding: 14px 18px;
    margin-top: 16px;
    box-shadow: 0 8px 18px rgba(93, 155, 150, 0.12);
  }
  section.compact table { font-size: 18px; }
  section.evidence table { width: 40%; margin-top: 10px; font-size: 17px; }
  section.evidence th, section.evidence td { padding: 5px 8px; }
  section.evidence img { max-height: 275px; }
  section.evidence .note { font-size: 18px; margin-top: 9px; }
  section.wide-figure img { max-height: 320px; }
  section.final ol { font-size: 22px; }
  section.final p { font-size: 22px; }
---

<!-- _class: title -->

# Оценка американских опционов нейронной сетью на основе биномиального дерева

<div class="subtitle">
Петров Артём Евгеньевич, НКНбд-01-22<br>
Научный руководитель: Шорохов С.Г.<br>
РУДН, 2026
</div>

---

## 1. Постановка задачи

- Объект исследования: **методы оценки опционов**
- Предмет исследования: **BTNet для американского пут-опциона и его чувствительностей**
- Американский пут допускает досрочное исполнение
- Поэтому задача оценки является задачей **оптимальной остановки**
- Для американского пута в общем случае нет простой аналитической формулы

<div class="note">В работе QuantLib CRR(500) используется как практический численный ориентир, а не как точное решение.</div>

---

## 2. Цель и задачи

**Цель:** реализовать и численно верифицировать BTNet для оценки американских пут-опционов и анализа греческих символов через autograd.

1. Изучить Black-Scholes, CRR и нейросетевую эквивалентность BTNet
2. Реализовать `btnn_bs` на Python/PyTorch
3. Проверить прайсинг European / American put
4. Вычислить Delta, Gamma, Vega и Theta через autograd
5. Исследовать перенос весов `BTNetEuropean → BTNetAmerican`

---

## 3. Границы собственного вклада

<div class="two">
<div class="box">

**Не заявляется**

- новая теория BTNet
- новый метод оценки всех опционов
- замена CRR / QuantLib

</div>
<div class="box">

**Сделано в ВКР**

- реализация `btnn_bs`
- верификация BTNetAmerican
- расчет греков через autograd
- эксперимент переноса весов
- выявление ограничения по Gamma

</div>
</div>

<div class="note">Теоретическая основа BTNet взята из работы Шорохова С. Г.</div>

---

<!-- _class: compact -->

## 4. Почему BTNet

| Критерий | CRR | Обычная нейросеть | BTNet |
|---|---|---|---|
| Интерпретация весов | высокая | обычно низкая | высокая при CRR-инициализации |
| Раннее исполнение | естественно | требует обучения | `MaxoutLayer` |
| Autograd | не базовая форма | доступен | доступен |
| Основной риск | дискретизация | black box | кусочно-линейность, глубина `n` |

<div class="note">BTNet не отменяет CRR: она записывает его как интерпретируемую дифференцируемую сеть.</div>

---

<!-- _class: compact -->

## 5. Нейросетевая запись CRR

| Элемент CRR | Элемент BTNet | Финансовый смысл |
|---|---|---|
| `max(K - S, 0)` | DenseLayer + ReLU | выплата в листьях дерева |
| дисконтированное ожидание | ConvLayer | шаг обратной индукции |
| `max(continuation, exercise)` | MaxoutLayer | раннее исполнение |
| корень дерева | выход сети | текущая цена |

<div class="note">Для американского опциона maxout является нейросетевой записью рекурсии Беллмана.</div>

---

<!-- _class: wide-figure -->

## 6. Архитектура BTNetAmerican

![w:820px](../../figures/fig_2_2_btnet_american_architecture.png)

<div class="note">Каждый слой соответствует шагу обратной индукции, а maxout выбирает между продолжением и исполнением.</div>

---

## 7. Методика верификации

<div class="two">
<div>

**Базовый сценарий**

- `S0 = 0.5`
- `T = 1`
- `r = 0.05`
- `sigma = 0.25`
- `n = 9`
- `K ∈ [0.25; 0.75]`

</div>
<div>

**Что проверяется**

- European: Black-Scholes / QuantLib
- American: QuantLib CRR(500)
- метрики: MAE, RMSE, max|err|
- греки: autograd vs finite differences

</div>
</div>

---

<!-- _class: evidence -->

## 8. Прайсинг American put

![w:780px](../../figures/fig_3_2_price_quantlib_errors.png)

| Инициализация | MAE | RMSE | max\|err\| |
|---|---:|---:|---:|
| Analytical W | **2.84·10⁻⁴** | **4.06·10⁻⁴** | **1.10·10⁻³** |
| Transferred W | 4.38·10⁻⁴ | 6.81·10⁻⁴ | 1.71·10⁻³ |

---

<!-- _class: evidence wide-figure -->

## 9. Перенос весов: цена зависит от режима

![w:780px](../../figures/fig_3_2_sigma_sweep_prices_errors.png)

| σ | Analytical W: MAE | Transferred W: MAE |
|---:|---:|---:|
| 0.10 | 2.66e-5 | 2.62e-5 |
| 0.25 | **2.84e-4** | 4.38e-4 |
| 0.60 | 1.47e-3 | **1.24e-3** |
| 0.90 | **2.06e-3** | 5.79e-3 |

<div class="note">Перенос может локально улучшить цену, но не является надежной процедурой инициализации.</div>

---

<!-- _class: compact -->

## 10. Почему перенос весов нестабилен

| W | w0 | w1 | Сумма | Интерпретация |
|---|---:|---:|---:|---|
| CRR | 0.4847 | 0.5097 | 0.9944 | риск-нейтральное ожидание |
| Transfer | 0.4889 | 0.5063 | 0.9952 | подстроен под европейскую цену |
| Разность | +0.0042 | -0.0034 | +0.0008 | смещение continuation value |

- Для европейской задачи такой сдвиг может быть полезен
- Для американской задачи он меняет сравнение `continuation` и `exercise`
- Для Vega / Theta фиксированный перенесенный W теряет канал зависимости от параметров

---

## 11. Греческие символы: главный отрицательный результат

| Грек | MAE | max\|diff\| |
|---|---:|---:|
| Delta | 7.92·10⁻³ | 9.25·10⁻² |
| Gamma | **0** | **0** |
| Vega | 3.31·10⁻³ | 3.65·10⁻² |
| Theta | 4.13·10⁻⁴ | 4.57·10⁻³ |

<div class="warning">
Gamma = 0 почти всюду не является ошибкой кода: это следствие кусочно-линейных ReLU/maxout.
</div>

---

<!-- _class: final -->

## 12. Выводы

1. Цель работы достигнута: BTNet реализована и верифицирована для American put
2. Аналитическая CRR-инициализация дает точность **MAE = 2.84·10⁻⁴**
3. Перенос весов European → American является нестабильной эвристикой
4. Текущая BTNet пригодна для воспроизводимого прайсинга и анализа отдельных чувствительностей
5. Для полноценного риск-менеджмента требуется модификация архитектуры из-за Gamma

**Перспективы:** гладкие max/ReLU, анализ глубины `n`, локальная волатильность, экзотические опционы.
