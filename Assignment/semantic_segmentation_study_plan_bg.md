# Учебен план за подготовка по използваните модели и архитектури
## Проект: Semantic Segmentation върху Cityscapes Dataset

**Цел на плана:** да подготви кратко, но систематично разбиране на използваните модели, архитектури, loss функции, метрики и експерименти, така че да можеш уверено да отговаряш на въпроси от преподавателя по проекта.

**Фокус на проекта:** semantic segmentation на градски сцени от Cityscapes Dataset с 19 класа. Най-добрият модел в проекта е `046_fpn_efficientnet_b3_ce_dice`, който използва **FPN + EfficientNet-B3 + Cross Entropy + Dice Loss + Cosine Annealing**.

---

## 0. Как да използваш този план

Препоръчителен подход:

1. Първо научи основната задача: какво е semantic segmentation и защо Cityscapes е подходящ dataset.
2. След това мини през архитектурите в ред: CNN basics → FCN → U-Net → U-Net++ → FPN → PSPNet → DeepLabV3+ → SegFormer.
3. За всяка архитектура си подготви кратък отговор на 4 въпроса:
   - Каква е основната идея?
   - Какъв проблем решава?
   - Какви са силните ѝ страни?
   - Защо я пробвахме в проекта?
4. Накрая научи loss функциите, метриките и как да обясниш защо най-добрият модел е FPN + EfficientNet-B3.

---

# Част I — Основи на задачата

## 1. Какво е semantic segmentation?

### Какво трябва да знаеш

Semantic segmentation е задача, при която **всеки пиксел** от изображението се класифицира към предварително дефиниран клас.

В проекта:

- вход: RGB изображение от градска сцена;
- изход: segmentation mask;
- брой класове: 19 Cityscapes training classes;
- всеки пиксел получава клас като `road`, `sidewalk`, `building`, `car`, `person`, `traffic sign` и др.

### Как да го обясниш на преподавателя

> Semantic segmentation не класифицира цялото изображение, а класифицира всеки пиксел. Така моделът не казва само „има кола“, а посочва кои точно пиксели принадлежат на път, кола, сграда, тротоар и т.н.

### Разлика спрямо други задачи

| Задача | Какво връща моделът |
|---|---|
| Image classification | Един клас за цялото изображение |
| Object detection | Bounding boxes + класове |
| Semantic segmentation | Клас за всеки пиксел |
| Instance segmentation | Маска за всяка отделна инстанция |
| Panoptic segmentation | Комбинация от semantic + instance segmentation |

### Възможен въпрос

**Въпрос:** Защо това е semantic segmentation, а не instance segmentation?  
**Отговор:** Защото всички пиксели от един и същ клас се третират еднакво. Ако има две коли, semantic segmentation ги маркира като клас `car`, но не ги разделя като `car_1` и `car_2`.

---

## 2. Dataset: Cityscapes

### Какво трябва да знаеш

Cityscapes съдържа реални градски сцени и pixel-level анотации. В проекта се използват:

```text
leftImg8bit_trainvaltest.zip
gtFine_trainvaltest.zip
```

Структурата е:

```text
data/raw/cityscapes/
├── leftImg8bit/
│   ├── train/
│   ├── val/
│   └── test/
└── gtFine/
    ├── train/
    ├── val/
    └── test/
```

### Как са използвани split-овете

- `train` — обучение;
- `val` — оценка и сравнение на моделите;
- `test` — inference/demo, защото публичният test split няма реални ground-truth masks.

### Какво е `labelTrainIds`?

Cityscapes има различни варианти на label masks. За training обикновено се използват `labelTrainIds`, които map-ват оригиналните класове към 19 training classes и използват ignore index за невалидни пиксели.

### Как да го обясниш

> Използвахме `leftImg8bit` като входни изображения и `gtFine` като pixel-level masks. За обучение се използват `labelTrainIds`, защото те свеждат Cityscapes labels до стандартните 19 класа, върху които се тренират semantic segmentation модели.

### Възможни въпроси

**Въпрос:** Защо не използваме test split за финални метрики?  
**Отговор:** Защото публичният test split няма реални ground-truth masks. Затова сравняваме моделите върху validation split-а.

**Въпрос:** Защо Cityscapes е подходящ за този проект?  
**Отговор:** Защото съдържа реални улични сцени и pixel-level анотации, което го прави стандартен dataset за semantic segmentation в контекста на autonomous driving и scene understanding.

---

## 3. Подготовка на данните

### Използвани preprocessing стъпки

- проверка за image-mask двойки;
- resize до `512×1024`;
- random crop до `512×512`;
- ImageNet normalization;
- augmentations:
  - `resize_only`;
  - `basic_aug`;
  - `strong_aug`.

### Защо resize и crop?

Оригиналните Cityscapes изображения са големи. Resize и crop намаляват нужната GPU памет и позволяват по-стабилно обучение с batch size 4.

### Как да го обясниш

> Resize и crop са компромис между качество и GPU памет. Full-resolution обучение би било по-добро за дребни обекти, но е много по-скъпо. Затова използвахме resize 512×1024 и crop 512×512.

### Възможен въпрос

**Въпрос:** Какъв е недостатъкът на crop 512×512?  
**Отговор:** Може да се загуби част от глобалния контекст на сцената, особено при wide street images. Също така малки и далечни обекти могат да станат още по-трудни за разпознаване.

---

# Част II — Основи на архитектурите

## 4. CNN основа за segmentation

### Какво трябва да знаеш

Convolutional Neural Networks извличат features чрез convolutional filters. При classification CNN обикновено завършва с fully connected layers, но при segmentation трябва да запазим spatial информацията и да върнем маска със същата или подобна пространствена резолюция.

### Основни понятия

| Понятие | Обяснение |
|---|---|
| Convolution | Извлича локални features |
| Pooling / downsampling | Намалява размерите и увеличава receptive field |
| Upsampling | Увеличава пространствената резолюция |
| Skip connection | Пренася детайлна информация от encoder към decoder |
| Encoder | Извлича features |
| Decoder | Възстановява segmentation mask |

### Как да го обясниш

> При segmentation не е достатъчно да знаем какво има в изображението. Трябва да знаем и къде се намира. Затова архитектурите използват encoder за извличане на features и decoder за възстановяване на pixel-level prediction.

---

## 5. FCN — Fully Convolutional Network

### Основна идея

FCN заменя fully connected слоевете с convolutional слоеве, така че моделът да може да прави dense pixel-wise predictions.

### Защо е важен?

FCN е една от първите архитектури, които показват как classification CNN може да се адаптира за segmentation.

### Ключови идеи

- няма final fully connected classifier;
- output-ът е spatial feature map;
- използва upsampling за връщане към размера на изображението;
- използва skip connections за по-добри детайли.

### Как да го обясниш

> FCN е основата на модерните segmentation модели. Идеята е да направим мрежата fully convolutional, за да получим prediction за всеки пиксел, а не само един клас за цялото изображение.

### Възможен въпрос

**Въпрос:** Защо fully connected слоевете са проблем при segmentation?  
**Отговор:** Те губят пространствената структура и обикновено изискват фиксиран размер на входа. При segmentation трябва да запазим пространствената информация.

---

## 6. Tiny U-Net

### Основна идея

Tiny U-Net е опростена версия на U-Net с по-малко слоеве и параметри. Използва се като лек baseline или за проверка дали pipeline-ът работи.

### Защо го използвахме?

- лесен за обучение;
- бърз за експерименти;
- полезен като sanity check;
- показва дали dataset loader, loss, metrics и training loop работят.

### Как да го обясниш

> Tiny U-Net не е очаквано да бъде най-добрият модел. Той служи като по-лек neural baseline, с който проверяваме дали цялата training инфраструктура работи правилно.

---

## 7. U-Net

### Основна идея

U-Net е encoder-decoder архитектура със skip connections между encoder и decoder частите.

### Структура

```text
Input → Encoder/downsampling → Bottleneck → Decoder/upsampling → Segmentation mask
              │                                  ↑
              └──────── skip connections ────────┘
```

### Защо U-Net работи добре?

- encoder-ът извлича high-level semantic features;
- decoder-ът възстановява spatial resolution;
- skip connections връщат детайлите, изгубени при downsampling;
- подходящ е за pixel-level задачи.

### Как да го обясниш

> U-Net комбинира глобална информация от дълбоките слоеве с локални детайли от ранните слоеве чрез skip connections. Това помага да се получат по-точни граници на обектите.

### Възможни въпроси

**Въпрос:** Каква е ролята на skip connections в U-Net?  
**Отговор:** Те пренасят high-resolution features от encoder-а към decoder-а, за да се възстановят по-добре границите и малките детайли.

**Въпрос:** Защо U-Net е добър baseline?  
**Отговор:** Защото е стабилна, добре позната encoder-decoder архитектура, която често работи добре при segmentation задачи.

---

## 8. U-Net++

### Основна идея

U-Net++ разширява U-Net чрез по-гъсти и nested skip connections между encoder и decoder блоковете.

### Какво подобрява?

В стандартния U-Net има semantic gap между encoder features и decoder features. U-Net++ се опитва да намали този gap чрез междинни convolutional blocks по skip пътищата.

### Как да го обясниш

> U-Net++ е развитие на U-Net. Вместо директни skip connections използва по-гъсти междинни връзки, които правят feature maps по-съвместими между encoder и decoder.

### В проекта

U-Net++ с ResNet34 и CE+Dice е в Top 5 по mean IoU, но не достига FPN и DeepLabV3+ с EfficientNet-B3.

---

## 9. FPN — Feature Pyramid Network

### Основна идея

FPN комбинира feature maps от различни мащаби. Това позволява моделът да използва както high-level semantic информация, така и по-детайлна spatial информация.

### Защо е полезен за Cityscapes?

Cityscapes съдържа обекти с различни размери:

- големи: road, building, sky;
- средни: car, bus, vegetation;
- малки: traffic light, traffic sign, person, rider, bicycle.

FPN е подходящ, защото multi-scale features помагат за различни размери на обекти.

### Как да го обясниш

> FPN изгражда пирамида от features. Така моделът може да разпознава едновременно големи обекти като път и сгради, както и по-малки обекти като знаци и пешеходци.

### Защо FPN беше най-добър в проекта?

Най-добрият модел е:

```text
046_fpn_efficientnet_b3_ce_dice
```

Конфигурация:

- architecture: FPN;
- encoder: EfficientNet-B3;
- weights: ImageNet;
- loss: Cross Entropy + Dice;
- scheduler: Cosine Annealing;
- mean IoU: 0.728307.

### Кратък отговор за защита

> Най-добрият резултат идва от FPN, защото архитектурата агрегира features от различни мащаби, а EfficientNet-B3 предоставя силен pretrained encoder. Това е важно за Cityscapes, където има както големи класове като road/building, така и дребни обекти като traffic sign и person.

---

## 10. PSPNet

### Основна идея

PSPNet използва Pyramid Pooling Module, който събира контекст от различни spatial мащаби.

### Какъв проблем решава?

При segmentation често е важно не само как изглежда локален patch, но и какъв е глобалният контекст. Например обект, който изглежда като част от път, може да бъде sidewalk или terrain в зависимост от сцената.

### Как да го обясниш

> PSPNet добавя pyramid pooling, за да включи глобален контекст. Това помага на модела да разбира сцената като цяло, а не само локални детайли.

### Разлика спрямо FPN

| FPN | PSPNet |
|---|---|
| Комбинира feature maps от различни encoder нива | Използва pooling на различни spatial мащаби |
| Силен за multi-scale feature fusion | Силен за global context |

---

## 11. DeepLabV3+

### Основна идея

DeepLabV3+ използва atrous/dilated convolutions и ASPP — Atrous Spatial Pyramid Pooling — за улавяне на контекст на различни мащаби, след което има decoder за по-добри граници.

### Ключови компоненти

| Компонент | Роля |
|---|---|
| Atrous convolution | Увеличава receptive field без силно намаляване на резолюцията |
| ASPP | Събира multi-scale context чрез различни dilation rates |
| Decoder | Възстановява по-добри spatial детайли |

### Какво е atrous/dilated convolution?

Това е convolution, при която има „разстояния“ между kernel елементите. Така kernel-ът вижда по-голяма област без да увеличава броя параметри толкова много.

### Как да го обясниш

> DeepLabV3+ използва dilated convolutions, за да разшири receptive field-а, и ASPP, за да гледа сцената на различни мащаби. Decoder-ът помага за по-добри граници на segmentation masks.

### В проекта

DeepLabV3+ се представя много силно:

- `047_deeplabv3plus_efficientnet_b3_ce_dice` е вторият най-добър модел;
- `043_deeplabv3plus_resnet101_ce_dice` също е в Top 5.

### Възможен въпрос

**Въпрос:** Защо DeepLabV3+ е подходящ за Cityscapes?  
**Отговор:** Защото Cityscapes има обекти на различни мащаби и сложен контекст. ASPP помага за multi-scale context, а decoder-ът подобрява границите.

---

## 12. SegFormer

### Основна идея

SegFormer е transformer-based segmentation архитектура. Тя използва hierarchical Transformer encoder и лек MLP decoder.

### Какво я различава от CNN моделите?

CNN моделите разчитат на convolutional filters и локални операции. Transformer-based моделите могат по-добре да моделират дълги зависимости и глобален контекст.

### Как да го обясниш

> SegFormer използва Transformer encoder, който може да улавя по-глобални зависимости в изображението. Това го прави различен от U-Net, FPN и DeepLabV3+, които са CNN-базирани архитектури.

### В проекта

`049_segformer_mit_b1_ce_dice_optional` е третият най-добър модел и най-добрият transformer-based експеримент.

### Възможен въпрос

**Въпрос:** Защо SegFormer не е най-добър, въпреки че е по-модерен?  
**Отговор:** Възможно е да изисква повече tuning, повече epochs, по-силен encoder като MiT-B2/B3 или по-голям resolution. В текущите ограничения FPN + EfficientNet-B3 се оказа по-добър практически избор.

---

# Част III — Encoders

## 13. Какво е encoder?

Encoder-ът е feature extractor. Той превръща изображението в feature maps, които съдържат по-абстрактна информация.

В проекта са използвани pretrained encoders:

- ResNet18;
- ResNet34;
- ResNet50;
- ResNet101;
- EfficientNet-B3;
- MiT-B1 за SegFormer.

### Защо pretrained weights?

Pretrained ImageNet weights помагат, защото encoder-ът вече е научил базови visual features като edges, textures, shapes и object parts.

### Как да го обясниш

> Вместо да започнем обучението от нула, използваме encoder, предварително обучен върху ImageNet. Така моделът започва с вече полезни визуални features и се адаптира към segmentation задачата.

---

## 14. ResNet encoders

### Основна идея

ResNet използва residual connections, които позволяват обучение на по-дълбоки мрежи.

### Какво е residual connection?

Вместо слой да учи директно mapping `H(x)`, той учи residual функция `F(x)`, а output-ът е:

```text
output = F(x) + x
```

Това помага срещу vanishing gradient и прави по-дълбоките мрежи по-лесни за обучение.

### Разлика между ResNet34, ResNet50, ResNet101

| Encoder | Характеристика |
|---|---|
| ResNet34 | по-лек, стабилен baseline |
| ResNet50 | по-дълбок, повече параметри |
| ResNet101 | още по-дълбок, по-силен, но по-тежък |

### Как да го обясниш

> ResNet използва skip/residual връзки вътре в encoder-а. Това улеснява обучението на по-дълбоки модели и позволява по-добро извличане на features.

---

## 15. EfficientNet-B3

### Основна идея

EfficientNet използва compound scaling — балансирано увеличава depth, width и input resolution.

### Защо EfficientNet-B3 се представи силно?

EfficientNet-B3 е по-мощен feature extractor от ResNet34, но остава относително ефективен спрямо размера си. В комбинация с FPN и DeepLabV3+ даде най-добрите резултати.

### Как да го обясниш

> EfficientNet-B3 предоставя силни pretrained features и добър баланс между сложност и качество. При FPN това е особено полезно, защото feature maps от различни мащаби са по-информативни.

---

## 16. MiT-B1 encoder при SegFormer

### Основна идея

MiT означава Mix Transformer. Това е hierarchical transformer encoder, използван от SegFormer.

### Защо е подходящ?

Той извлича features на различни spatial нива, подобно на CNN pyramid, но с transformer-based attention механизми.

### Как да го обясниш

> MiT-B1 е transformer encoder, който създава multi-scale feature representations. Това позволява на SegFormer да работи като segmentation модел без тежък convolutional decoder.

---

# Част IV — Loss функции

## 17. Cross Entropy Loss

### Основна идея

Cross Entropy е стандартна loss функция за classification. При semantic segmentation тя се прилага pixel-wise.

### Какво оптимизира?

За всеки пиксел моделът предсказва вероятност за всеки клас. Cross Entropy наказва модела, ако истинският клас има ниска вероятност.

### Как да го обясниш

> Pixel-wise Cross Entropy третира всеки пиксел като отделна classification задача.

### Силен плюс

- стабилна;
- лесна за оптимизация;
- стандартна за multi-class segmentation.

### Слабост

- може да бъде доминирана от големи класове като road, building и sky.

---

## 18. Dice Loss

### Основна идея

Dice Loss измерва overlap между predicted mask и ground truth mask.

### Защо е полезна?

При class imbalance Dice помага, защото се интересува от припокриването, а не само от броя правилни пиксели.

### Как да го обясниш

> Dice Loss се фокусира върху това колко добре предсказаната област се припокрива с истинската област. Затова е полезна при segmentation, особено когато класовете са небалансирани.

---

## 19. Focal Loss

### Основна идея

Focal Loss намалява влиянието на лесните примери и фокусира обучението върху трудните.

### Защо е полезна?

В Cityscapes има class imbalance. Големите класове са много по-чести, а малките класове като `traffic light`, `rider`, `motorcycle` са редки.

### Как да го обясниш

> Focal Loss помага при class imbalance, защото намалява тежестта на лесно класифицираните пиксели и насочва модела към трудните примери.

---

## 20. Lovasz Loss

### Основна идея

Lovasz Loss е loss функция, ориентирана към IoU/Jaccard optimization.

### Защо я пробвахме?

Основната метрика в проекта е `mean_iou`, затова е логично да се пробва loss, която е по-близка до IoU.

### Как да го обясниш

> Lovasz Loss е интересна, защото е свързана с оптимизацията на IoU. В проекта обаче CE+Dice се оказа по-надеждна комбинация.

---

## 21. Защо CE + Dice беше най-добра комбинация?

### Основна идея

Комбинацията Cross Entropy + Dice съчетава два типа сигнал:

- Cross Entropy — добър pixel-wise classification signal;
- Dice — overlap-based segmentation signal.

### Как да го обясниш

> CE+Dice работи добре, защото Cross Entropy учи модела да класифицира пикселите правилно, а Dice Loss го насърчава да прави по-добро припокриване на областите. Така се получава баланс между класификация и segmentation quality.

---

# Част V — Learning rate schedulers

## 22. Cosine Annealing

### Основна идея

Cosine Annealing плавно намалява learning rate по cosine крива.

### Защо е полезен?

- започва с по-голям learning rate;
- постепенно го намалява;
- помага за по-стабилна convergence;
- често работи добре при deep learning експерименти.

### Как да го обясниш

> Cosine Annealing позволява на модела да учи по-бързо в началото и по-фино в края, като плавно намалява learning rate.

---

## 23. Step Decay

### Основна идея

Step Decay намалява learning rate на фиксирани интервали.

### Как да го обясниш

> Step Decay прави дискретни намаления на learning rate, например на всеки N epochs. За разлика от Cosine Annealing, промяната не е плавна.

---

## 24. Reduce on Plateau

### Основна идея

Reduce on Plateau намалява learning rate, когато validation metric или validation loss спре да се подобрява.

### Как да го обясниш

> Reduce on Plateau реагира динамично. Ако моделът спре да подобрява validation loss, scheduler-ът намалява learning rate.

---

# Част VI — Метрики и анализ

## 25. Pixel Accuracy

### Основна идея

Pixel Accuracy измерва дела на правилно класифицираните пиксели.

### Проблем

Cityscapes е class-imbalanced. Ако моделът е много добър върху road и building, pixel accuracy може да е висока дори при слабо представяне върху малки класове.

### Как да го обясниш

> Pixel Accuracy не е достатъчна сама по себе си, защото фаворизира доминиращите класове. Затова използвахме mIoU като основна метрика.

---

## 26. IoU и mean IoU

### Какво е IoU?

IoU сравнява припокриването между predicted mask и ground truth mask:

```text
IoU = intersection / union
```

### Какво е mean IoU?

Mean IoU е средната IoU стойност по класове.

### Защо е основна метрика?

mIoU е по-справедлива при class imbalance, защото отчита performance по класове, а не само общ брой правилни пиксели.

### Как да го обясниш

> mIoU измерва качеството на segmentation по класове. Това е по-подходящо от pixel accuracy, защото не позволява големите класове напълно да доминират оценката.

---

## 27. Dice / mean Dice

### Основна идея

Dice измерва overlap между prediction и ground truth. Подобна е на IoU, но има различна формула и често е по-чувствителна към overlap quality.

### Как да го обясниш

> Dice е overlap метрика. Използваме я като допълнение към mIoU, за да проверим колко добре predicted regions съвпадат с истинските regions.

---

## 28. Confusion matrix анализ

### Какво показва confusion matrix?

Confusion matrix показва за всеки истински клас към кой клас е предсказан.

### Защо е полезна?

Тя помага да се открият типични грешки:

- `road` ↔ `sidewalk`;
- `terrain` ↔ `vegetation`;
- `person` ↔ `rider`;
- `truck` ↔ `bus`;
- `traffic sign` ↔ `pole` или background класове.

### Как да го обясниш

> Confusion matrix е полезна за error analysis, защото показва не само дали моделът греши, а и с какво бърка даден клас.

---

# Част VII — Как да защитиш експериментите

## 29. Каква беше експерименталната стратегия?

### Основни направления

1. Baseline модел.
2. Tiny U-Net / basic neural baseline.
3. U-Net family.
4. Multi-scale architectures: FPN, PSPNet, DeepLabV3+.
5. Transformer-based SegFormer.
6. Различни encoders.
7. Различни loss функции.
8. Различни preprocessing и augmentations.
9. Различни schedulers.

### Как да го обясниш

> Експериментите не са случайни. Те постепенно сравняват архитектура, encoder, loss функция, scheduler и preprocessing. Целта е да се разбере кое реално подобрява mIoU.

---

## 30. Защо най-добрият модел е FPN + EfficientNet-B3?

### Факти от проекта

Най-добрият модел:

```text
046_fpn_efficientnet_b3_ce_dice
```

Метрики:

```text
mean_iou = 0.728307
mean_dice = 0.833546
pixel_accuracy = 0.950271
```

### Обяснение

FPN е силен за multi-scale segmentation, а EfficientNet-B3 е силен pretrained encoder. CE+Dice дава стабилен training signal. Тази комбинация е особено подходяща за Cityscapes, защото dataset-ът има обекти с много различни размери.

### Кратък отговор за устен изпит

> FPN + EfficientNet-B3 беше най-добър, защото FPN комбинира features на различни мащаби, EfficientNet-B3 предоставя по-силен pretrained encoder, а CE+Dice балансира pixel-wise classification и overlap quality. Това доведе до най-висок mean IoU.

---

## 31. Защо DeepLabV3+ е много близо до FPN?

DeepLabV3+ също е multi-scale архитектура чрез ASPP. Тя е много подходяща за Cityscapes. Разликата спрямо FPN е, че DeepLabV3+ използва atrous convolutions и ASPP, докато FPN комбинира feature maps от различни encoder нива.

### Кратък отговор

> DeepLabV3+ е близо до FPN, защото също работи с multi-scale context. При текущите настройки FPN + EfficientNet-B3 даде малко по-добър резултат, вероятно заради по-ефективното feature aggregation в комбинация с EfficientNet-B3.

---

## 32. Защо 50 epochs не подобриха U-Net ResNet34?

В проекта `040_unet_resnet34_ce_dice_50ep` не подобрява значително резултата спрямо 30-epoch варианта.

### Възможно обяснение

- моделът вече е достигнал plateau;
- допълнителните epochs не добавят нова информация;
- може да има лек overfitting;
- ограничението може да идва от архитектурата, не от броя epochs.

### Кратък отговор

> Повече epochs не гарантират по-добър резултат. При U-Net ResNet34 моделът вероятно вече е достигнал plateau и ограничението е по-скоро архитектурно, а не времево.

---

## 33. Защо малките класове са проблемни?

### Причини

- заемат малко пиксели;
- срещат се по-рядко;
- често са далеч от камерата;
- могат да бъдат закрити;
- при resize/crop губят детайли.

### Примери

- traffic light;
- traffic sign;
- rider;
- bicycle;
- motorcycle;
- person.

### Кратък отговор

> Малките класове са трудни, защото имат малко пиксели и са по-редки. При resize и crop част от детайла се губи, което понижава IoU за тези класове.

---

# Част VIII — Въпроси, които преподавателят може да зададе

## 34. Бързи въпроси и готови отговори

### 1. Какво е semantic segmentation?

Semantic segmentation е pixel-wise classification задача, при която всеки пиксел получава семантичен клас.

### 2. Защо използвахте Cityscapes?

Защото съдържа реални градски сцени с pixel-level анотации и е стандартен dataset за autonomous driving и scene understanding.

### 3. Защо mIoU е основна метрика?

Защото оценява overlap по класове и е по-подходяща при class imbalance от pixel accuracy.

### 4. Защо pixel accuracy не е достатъчна?

Защото големи класове като road и building доминират броя пиксели и могат да дадат висока accuracy, дори моделът да е слаб върху редки класове.

### 5. Как работи U-Net?

U-Net има encoder-decoder структура със skip connections, които комбинират high-level features и spatial details.

### 6. Какво е FPN?

FPN е Feature Pyramid Network, която комбинира features от различни мащаби, за да разпознава обекти с различни размери.

### 7. Какво е ASPP в DeepLabV3+?

ASPP е Atrous Spatial Pyramid Pooling — модул, който използва dilated convolutions с различни dilation rates за multi-scale context.

### 8. Какво е SegFormer?

SegFormer е transformer-based segmentation архитектура с hierarchical Transformer encoder и лек MLP decoder.

### 9. Защо CE+Dice работи добре?

CE учи pixel-wise classification, а Dice оптимизира overlap. Комбинацията балансира двата аспекта.

### 10. Защо най-добрият модел е FPN + EfficientNet-B3?

Защото FPN агрегира multi-scale features, EfficientNet-B3 е силен pretrained encoder, а CE+Dice дава стабилен segmentation signal.

### 11. Защо използвахте pretrained encoders?

Защото pretrained encoders вече знаят базови визуални features, което ускорява и стабилизира обучението.

### 12. Какви са ограниченията на проекта?

Resize/crop вместо full resolution, class imbalance, ограничена GPU памет, липса на test ground truth и ограничен transformer search.

---

# Част IX — Мини програма за учене

## Ден 1 — Задача и данни

### Научи

- semantic segmentation;
- Cityscapes classes;
- image-mask pairs;
- labelTrainIds;
- train/val/test split.

### Провери се

Можеш ли за 1 минута да обясниш:

> Какво влиза в модела и какво излиза от него?

---

## Ден 2 — CNN, FCN и U-Net

### Научи

- convolution;
- downsampling;
- upsampling;
- encoder-decoder;
- skip connections;
- U-Net;
- Tiny U-Net;
- U-Net++.

### Провери се

Можеш ли да обясниш:

> Защо skip connections са важни за segmentation?

---

## Ден 3 — Multi-scale архитектури

### Научи

- FPN;
- PSPNet;
- DeepLabV3+;
- atrous/dilated convolutions;
- ASPP;
- pyramid pooling.

### Провери се

Можеш ли да сравниш:

> FPN vs DeepLabV3+?

---

## Ден 4 — Encoders и pretrained learning

### Научи

- ResNet;
- residual connections;
- EfficientNet;
- compound scaling;
- MiT-B1;
- ImageNet weights.

### Провери се

Можеш ли да обясниш:

> Защо EfficientNet-B3 подобри резултата спрямо ResNet34?

---

## Ден 5 — Loss функции и метрики

### Научи

- Cross Entropy;
- Dice Loss;
- Focal Loss;
- Lovasz Loss;
- CE+Dice;
- IoU;
- mean IoU;
- pixel accuracy;
- confusion matrix.

### Провери се

Можеш ли да отговориш:

> Защо mIoU е по-важна от pixel accuracy?

---

## Ден 6 — Експерименти и защита на резултатите

### Научи

- Top 10 модели;
- best model configuration;
- loss curves;
- generalization gap;
- грешки при малки класове;
- защо FPN + EfficientNet-B3 е final model.

### Провери се

Можеш ли да направиш 2-минутно обяснение:

> Как стигнахме до избора на финален модел?

---

## Ден 7 — Репетиция за защита

### Направи

1. Представи проекта за 5 минути.
2. Обясни best model за 1 минута.
3. Обясни U-Net, FPN и DeepLabV3+.
4. Обясни CE+Dice и mIoU.
5. Подготви отговор за ограниченията и бъдеща работа.

---

# Част X — Най-важните неща за запомняне

## Финален кратък summary

Проектът решава semantic segmentation върху Cityscapes, където всеки пиксел от градска сцена се класифицира към един от 19 класа. Изграденият pipeline използва PyTorch/SMP, YAML конфигурации, preprocessing, augmentations, pretrained encoders, различни loss функции и schedulers. Сравнени са U-Net, U-Net++, FPN, PSPNet, DeepLabV3+, Tiny U-Net и SegFormer. Най-добрият модел е `046_fpn_efficientnet_b3_ce_dice`, защото комбинира multi-scale feature aggregation, силен EfficientNet-B3 encoder и CE+Dice loss. Основната метрика е mean IoU, защото е по-подходяща от pixel accuracy при class imbalance.

## Едно изречение за финална защита

> Най-добрият резултат беше постигнат от FPN + EfficientNet-B3 + CE Dice, защото тази комбинация съчетава multi-scale feature aggregation, силни pretrained visual representations и loss функция, която балансира pixel-wise classification и overlap quality.

