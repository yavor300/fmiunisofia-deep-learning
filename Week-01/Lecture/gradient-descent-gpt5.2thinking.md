## Лекция 2 (20.02.2026): Градиентно спускане и логистична регресия

Източник: Лекция2(GD).pdf 

---

### 1) Recap: Logistic Regression (binary classification)

**Модел**

* $z = w^T x + b$
* $\hat{y} = a = \sigma(z)$
* $\sigma(z) = \frac{1}{1 + e^{-z}}$

**Loss за един пример (cross-entropy / log loss)**

* $\mathcal{L}(a, y) = -\big(y\log(a) + (1-y)\log(1-a)\big)$

**Cost за $m$ примера**

* $J(w,b) = \frac{1}{m}\sum_{i=1}^{m}\mathcal{L}\big(a^{(i)}, y^{(i)}\big)$

Цел: намираме $w,b$, които минимизират $J(w,b)$. 

---

### 2) Градиентно спускане (Gradient Descent)

Итеративно обновяване:

* $w := w - \alpha \frac{\partial J(w,b)}{\partial w}$
* $b := b - \alpha \frac{\partial J(w,b)}{\partial b}$

където $\alpha$ е learning rate (стъпка). 

---

### 3) Изчислителен граф и backpropagation (интуиция)

#### 3.1) Пример с граф

Функция:

* $J(a,b,c) = 3a + bc$

Междинни стъпки:

* $u = bc$
* $v = a + u$
* $J = 3v$

Идея: смятаме производните назад по веригата (chain rule). 

#### 3.2) Общ принцип (chain rule)

* $\frac{\partial J}{\partial x}$ се получава като произведение от локалните производни по пътя от $x$ към $J$.

#### 3.3) “Upstream” × “Local” gradient

Backprop работи като:

* (градиент отдясно) $\times$ (локална производна на операцията)

Типично поведение през операции:

* **add gate**: разпределя градиента към входовете
* **mul gate**: мащабира градиента (умножава по другия вход)
* **max gate**: пропуска градиента само по активния (максималния) клон 

#### 3.4) Важно правило: сумиране при разклонения

Ако променлива влияе на $J$ по повече от един път, общият ѝ градиент е **сума** от приносите по всички пътища. 

---

### 4) Сигмоида: производна и backprop през нея

Сигмоида:

* $\sigma(z) = \frac{1}{1 + e^{-z}}$

Производна:

* $\frac{d\sigma}{dz} = \sigma(z)\big(1-\sigma(z)\big)$

В лекцията се разглежда граф на $\sigma$ като серия от операции (exp, +1, reciprocal) и как градиентът се връща назад през тях. 

---

### 5) Backprop за логистична регресия (ключов резултат)

За един пример:

* $z = w^T x + b$
* $a = \sigma(z)$
* $\mathcal{L}(a,y) = -\big(y\log(a) + (1-y)\log(1-a)\big)$

Ключов резултат:

* $\frac{\partial \mathcal{L}}{\partial z} = a - y$

Често се записва като:

* $dz = a - y$
* $dw = x \cdot dz$  (поелементно за всеки компонент на $w$)
* $db = dz$ 

---

### 6) Обучение с $m$ примера (усредняване)

Идея: cost и градиенти са средни стойности върху всички примери.

Псевдопроцедура:

1. Инициализация: $J=0$, $dw=0$, $db=0$
2. За $i=1..m$:

   * $z^{(i)} = w^T x^{(i)} + b$
   * $a^{(i)} = \sigma(z^{(i)})$
   * $J += \mathcal{L}(a^{(i)}, y^{(i)})$
   * $dz^{(i)} = a^{(i)} - y^{(i)}$
   * $dw += x^{(i)} dz^{(i)}$
   * $db += dz^{(i)}$
3. Усредняване: $J/=m$, $dw/=m$, $db/=m$
4. Update:

   * $w := w - \alpha dw$
   * $b := b - \alpha db$ 

---

### 7) Векторизация (по-бърза имплементация)

Нотация:

* $X \in \mathbb{R}^{n_x \times m}$ (колони = примери)
* $w \in \mathbb{R}^{n_x \times 1}$, $b \in \mathbb{R}$

Forward:

* $Z = w^T X + b$
* $A = \sigma(Z)$

Gradients:

* $dZ = A - Y$
* $dw = \frac{1}{m} X dZ^T$
* $db = \frac{1}{m}\sum dZ$

Update:

* $w := w - \alpha dw$
* $b := b - \alpha db$ 

---

### 8) Какво да запомниш

* Основни формули: $z=w^Tx+b$, $a=\sigma(z)$, $\mathcal{L}$, $J$
* Gradient Descent update с learning rate $\alpha$
* Най-важното за логистична регресия: **$dz = a - y$**
* При разклонения: **градиентите се сумират**
* Векторизация: $Z=w^TX+b$, $A=\sigma(Z)$, $dZ=A-Y$, $dw=\frac1mXdZ^T$, $db=\frac1m\sum dZ$
