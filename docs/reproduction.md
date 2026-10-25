# Как запустить проект

Нужны Python 3.12 и Linux или WSL/Ubuntu. Команды выполняются из корня проекта.

## Установка

```bash
python3.12 -m venv task_1/.venv-featuretools
source task_1/.venv-featuretools/bin/activate
python -m pip install -r task_1/requirements.txt
python -m pip check
python task_1/src/check_project.py --smoke --skip-artifacts
```

В Jupyter или VS Code нужно выбрать Python этой среды.
Один тест сохранённых checkpoint пропускается, когда моделей нет локально.

## Два режима ноутбуков

В первой кодовой ячейке каждого ноутбука:

```python
MODE = 'report'       # сохранённые результаты
# MODE = 'recompute'  # новый расчёт данных, признаков и моделей
RUN_NAME = 'repeat_01'
DOWNLOAD_DATA = True
INCLUDE_TEST = False
```

`report` выполняет ячейки без исходного датасета и моделей:

```bash
python task_1/src/run_report_notebooks.py
```

Команда выполняет все пять ноутбуков в новых ядрах, не меняя исходные файлы.
Для сохранения выполненных копий укажите отдельную папку:

```bash
python task_1/src/run_report_notebooks.py --output-dir task_1/.validation/reports
```

Обновить выводы непосредственно в исходных ноутбуках можно только явно:

```bash
python task_1/src/run_report_notebooks.py --in-place
```

Те же правила действуют для `run_experiment_notebooks.py --mode report`.
Флаг `--check` сохранён как явный вариант выполнения без записи.

`recompute` запускает расчёты по мере чтения ноутбука. Все этапы используют
один RUN_NAME и передают новые результаты дальше. Завершённые этапы повторно
проверяются, незавершённые останавливают цепочку. При ошибке старые CSV
не подставляются вместо новых результатов.

Полный проход пяти ноутбуков с новой оценкой ранее открытого test:

```bash
python task_1/src/run_experiment_notebooks.py \
  --mode recompute --run-name repeat_01 --download --include-test
```

Без `--include-test` новая test-оценка пропускается. Выполненные ноутбуки
нового запуска сохраняются в `task_1/results/reproduction/repeat_01/notebooks/`.
Для нового расчёта нужна отдельная чистая копия проекта. Существующие данные
и результаты не перезаписываются.

Runner фиксирует inline-бэкенд Matplotlib, чтобы графики сохранялись в выводах ноутбуков независимо от внешнего MPLBACKEND.

## Рисунок итогового сравнения

Диаграмма интервалов в README строится из опубликованных CSV:

```bash
python task_1/src/plot_final_intervals.py
```

Команда обновляет только docs/figures/c4_intervals.png. Обучение и новая оценка моделей не выполняются.

## Пути в опубликованных результатах

Пути в JSON и CSV относительны к папке task_1; например, results/c4/ находится внутри неё, а ../docs/ — в корне репозитория. Читатели сохранённых протоколов разрешают их относительно текущей копии проекта.

Из метаданных и архива ноутбуков удалены личные абсолютные пути. В [манифесте обезличивания](provenance/path_sanitization.json) сохранены исходные и новые SHA256; связанные хеши метаданных обновлены. Параметры обучения, метрики и хеши моделей, исходных данных и исторического кода не подменялись.

## Данные

Snapshot Yambda: `dd6f3a19eef5866e346c3270e098baa641a44948`.
Имена, размеры и SHA256 файлов закреплены в
[dataset_source.json](../task_1/dataset_source.json).
Вместо скачивания можно передать локальные файлы:

```bash
python task_1/src/run_experiment_notebooks.py \
  --mode recompute --run-name repeat_01 --include-test \
  --raw-events /absolute/path/to/multi_event.parquet \
  --artist-mapping /absolute/path/to/artist_item_mapping.parquet
```

`prepare_history.py` сохраняет listens/events и описание источников.
`prepare_legacy.py` создаёт samples и ручные признаки старого протокола.
EDA читается порциями целых пользователей; большие матрицы остаются на диске.

## Какие этапы повторяются

| Ноутбук | Вычисления в режиме recompute |
| --- | --- |
| dataset-analysis | История, разметка, сводки, дубликаты и семь EDA-графиков |
| model-baseline | Samples, 42 ручных входа, простые ориентиры, YetiRank и MultiClass, Optuna |
| experiment-1 | DFS130, два обучения, bootstrap, пять временных окон, построение формул |
| experiment-2 | 12 782 кандидата, top-k, признаки деревьев, полный DFS, пять контролей, интервалы, диагностика start_ts, семь моделей на test |
| final-comparison | Исправленное время запроса, 42/49/462 входа, контроль 48 входов без текущего канала, интервалы, test, важности |

Повторный поиск по деревьям может выбрать другое число входов. Оно сохраняется
в новом протоколе. Основное исправленное сравнение использует фиксированный
опубликованный список 462 формул, как и исходное исследование.

Весь граф доступен без Jupyter. По умолчанию команда только печатает план:

```bash
python task_1/src/run_all_experiments.py --run-name repeat_01 --download --include-test
```

Для выполнения добавить `--execute`. Отдельные ветви выбираются через
`--experiments`, например `--experiments bootstrap rolling`.

## Основное сравнение: ручные 48 и DFS 462

Из исходных данных, без нового исторического поиска и Optuna:

```bash
python task_1/src/run_all_experiments.py --run-name repeat_main \
  --experiments channel-control --download --include-test --execute
```

Граф готовит историю и C4-пулы, обучает три первоначальных модели и дополнительный контроль без текущего канала. В итоговом сравнении используются ручные 48 и DFS 462; модели 42/49 остаются предыдущими этапами. Test — ранее открытый период.

### Первоначальный маршрут трёх моделей

Без дополнительного контроля 48:

```bash
python task_1/src/reproduce.py --run-name repeat_main --stage all --download
```

Команда готовит историю, обучает три ранкера и оценивает их на прежнем test.
Этапы отдельно:

```bash
python task_1/src/prepare_history.py --download
python task_1/src/run_c4.py --run-name repeat_main --max-training-seconds 7200
python task_1/src/run_c4_test.py --run-name repeat_main_test --training-run repeat_main
```

Результаты: `task_1/results/c4/repeat_main/` и
`task_1/results/c4_test/repeat_main_test/`. Таблицы — comparison.csv
и bootstrap_summary.csv. Завершение — completed=true в status.json и result.json.

## Контроль без текущего канала

Готовое сравнение 48 ручных признаков с DFS читается из results/c4/no_current_channel/. Исторические модели 42/49/462 сохранены отдельно.

План повторного контроля на готовых C4-пулах печатается без обучения:

```bash
python task_1/src/run_channel_control.py --training-run repeat_main --run-name repeat_channel
```

Для нового fit нужно явно добавить --stage fit. Отдельная оценка ранее открытого test:

```bash
python task_1/src/run_channel_control.py --stage evaluate --include-test \
  --training-run repeat_main --test-run repeat_main_test --run-name repeat_channel
```

В recompute итоговый notebook использует тот же RUN_NAME: контроль зависит от новых C4-пулов, а при INCLUDE_TEST=True — и от нового test этого запуска. Готовые старые метрики вместо нового расчёта не подставляются.

## Время и ресурсы

Основной исходный train/validation после подготовки истории занял 67.04 минуты,
test — 7.47 минуты. Полный повтор всей истории с Optuna, временными окнами,
широким DFS и контролями занимает значительно больше времени.

Поиск Optuna ограничен двумя часами; полные обучения учитываются отдельно.
Контроли ограничены шестью часами, фаза трёх основных моделей — двумя часами.
Ориентир: 12 GiB RAM и не менее 40 GB свободного места; промежуточные файлы
всех ветвей требуют дополнительного диска.

## Проверка пересчёта

```bash
python task_1/src/verify_reproduction.py --output-dir task_1/.validation/synthetic_01
```

Проверка создаёт искусственный журнал и временную копию: подготовка истории,
EDA, 42/49/462 признака, три обучения, искусственный test и важности.
Исследовательские данные и модели не используются.

Основное исходное C4-сравнение независимо повторено в отдельной Python 3.12.13 venv: шесть полных матриц manual49/DFS462 для train/validation/test совпали побайтно. Три модели повторили 791/685/1043 дерева, все validation/test-прогнозы и обе метрики. Ручной контроль без текущего канала дополнительно предсказал на свежем test Pool: все 3 160 068 прогнозов совпали с сохранённой оценкой. [Подтверждение](results/reproduction/main_verification.json).

Это полный повтор основного C4-сравнения первоначальной публикации. Исторический Optuna и все старые временные окна полностью не переобучались. Новые версии отчётного режима проверены отдельно; новый CLI контроля 48 в текущей ветке прошёл план и тесты без повторного fit. [Более ранние проверки](provenance/audit_reproduction_20261025.json).

Метрики сохранённого исследования: [основной протокол](c4.md),
[test](c4-test.md), [выбранные признаки](feature-analysis.md).
