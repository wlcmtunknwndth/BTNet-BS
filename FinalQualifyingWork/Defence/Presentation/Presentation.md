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
  ul, ol { font-size: 23px; line-height: 1.32; }
  li { margin-bottom: 7px; }
  table {
    width: 100%;
    font-size: 19px;
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
    padding: 7px 9px;
  }
  img {
    max-height: 315px;
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
    padding: 72px 68px;
  }
  section.title h1 { color: var(--blue-dark); font-size: 42px; max-width: 820px; }
  section.title h1::after {
    content: "";
    display: block;
    width: 130px;
    height: 5px;
    margin-top: 24px;
    background: linear-gradient(90deg, #6fa8cc, #9fc8c4);
    border-radius: 5px;
  }
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
  section.figure img { max-height: 430px; }
  section.evidence img { max-height: 280px; }
  section.evidence table { font-size: 18px; }
  .final-grid {
    display: grid;
    grid-template-columns: 1fr 190px;
    gap: 28px;
    align-items: start;
  }
  .qr-card {
    background: var(--paper);
    border: 1px solid #d7e1ec;
    border-radius: 8px;
    padding: 14px;
    text-align: center;
    box-shadow: 0 10px 22px rgba(15, 23, 42, 0.07);
  }
  .qr-card img {
    width: 132px;
    max-height: 132px;
    padding: 0;
    border: none;
    border-radius: 0;
    box-shadow: none;
  }
  .qr-card .label {
    font-size: 15px;
    line-height: 1.2;
    color: var(--muted);
    margin-top: 8px;
  }
---

<!-- _class: title -->

# Оценка американских опционов нейронной сетью на основе биномиального дерева

<div class="subtitle">
Петров Артём Евгеньевич, НКНбд-01-22<br>
Научный руководитель: Шорохов С.Г.<br>
РУДН, 2026
</div>

---

## 1. Что такое американский опцион

- **Пут-опцион** дает право продать актив по заранее заданной цене `K`
- **Европейский** опцион исполняется только в дату экспирации
- **Американский** опцион можно исполнить в любой момент до экспирации
- Поэтому при оценке нужно выбирать:

<div class="box">

продолжать держать опцион **или** исполнить его сейчас

</div>

<div class="note">Именно право досрочного исполнения делает американский пут задачей оптимальной остановки.</div>

---

## 2. Почему оценка сложна

- Для европейского опциона есть аналитическая формула Black-Scholes
- Для американского пута простой замкнутой формулы в общем случае нет
- Цена зависит от будущей траектории базового актива и решения об исполнении
- На практике используют численные методы: CRR, конечные разности, Monte Carlo

**Задача ВКР:** реализовать модели архитектуры, основанной на биномиальном дереве метода CRR, обучить европейскую модель и проверить американскую модель.

---

<!-- _class: figure -->

## 3. Биномиальное дерево CRR

![w:960px](../../figures/fig_1_crr_tree.png)

---

## 4. Формулы CRR для American put

Параметры дерева:

| Параметр | Формула |
|---|---|
| шаг времени | `dt = T / n` |
| рост | `u = exp(sigma * sqrt(dt))` |
| падение | `d = exp(-sigma * sqrt(dt))` |
| вероятность | `p = (exp(r dt) - d) / (u - d)` |

Рекурсия для американского пута:

`V(i,j) = max( K - S(i,j), exp(-r dt) * (p V_up + (1-p) V_down) )`

---

## 5. Что такое BTNet

**BTNet = Binomial Tree Neural Network.**

Это нейросетевая архитектура, прямой проход которой повторяет обратную индукцию CRR-дерева.

| CRR | BTNet |
|---|---|
| терминальная выплата `max(K-S,0)` | первый слой |
| дисконтированное ожидание | линейный фильтр `W` |
| шаги назад по дереву | последовательность слоев |
| досрочное исполнение | `maxout` |

<div class="note">Теоретическая основа BTNet взята из работы Шорохова С. Г.</div>

---

## 6. Реализация в ВКР

<div class="two">
<div class="box">

**`btnn_bs`**

программный пакет на Python/PyTorch для воспроизводимых экспериментов

**`BTNetEuropean`**

модель для европейского пут-опциона

</div>
<div class="box">

**`BTNetAmerican`**

модель для американского пут-опциона с maxout-слоями

**Greeks**

Delta, Gamma, Vega, Theta — чувствительности цены к рыночным параметрам

</div>
</div>

---

## 7. Ключевая особенность архитектуры

![w:930px](../../figures/fig_2_2_btnet_american_architecture_simple.png)

---

<!-- _class: evidence -->

## 8. Проверка точности цены

![w:760px](../../figures/fig_3_2_price_quantlib_errors.png)

| Инициализация | MAE | RMSE | max\|err\| |
|---|---:|---:|---:|
| Analytical W | **2.84·10⁻⁴** | **4.06·10⁻⁴** | **1.10·10⁻³** |
| Transferred W | 4.38·10⁻⁴ | 6.81·10⁻⁴ | 1.71·10⁻³ |

<div class="note">Ориентир: QuantLib CRR(500), не точное аналитическое решение.</div>

---

## 9. Важные результаты экспериментов

<div class="two">
<div class="box">

**Перенос весов**

- European → American не дает стабильного улучшения
- при `sigma = 0.90` ошибка заметно растет
- аналитическая CRR-инициализация надежнее

</div>
<div class="box">

**Ограничение для рисков**

- чувствительности можно считать через autograd
- Gamma = 0 почти всюду
- причина: ReLU/maxout кусочно-линейны

</div>
</div>

<div class="warning">Это ограничение важно для риск-менеджмента: для полноценного delta-gamma hedging архитектуру нужно сглаживать.</div>

---

## 10. Что сделано и выводы

<div class="final-grid">
<div>

**В работе выполнено:**

1. Реализован пакет `btnn_bs` на Python/PyTorch
2. Построены модели `BTNetEuropean` и `BTNetAmerican`
3. Обучена европейская модель и проверена американская модель
4. Проведена верификация цены относительно Black-Scholes и QuantLib CRR(500)
5. Проведен эксперимент переноса весов European → American

**Основные выводы:** аналитическая CRR-инициализация дает точную и интерпретируемую цену, перенос весов нестабилен, а Gamma требует гладкой модификации архитектуры.

</div>
<div class="qr-card">

![QR код репозитория](../../figures/github_repo_qr.png)

<div class="label">GitHub<br>BTNet-BS</div>

</div>
</div>
