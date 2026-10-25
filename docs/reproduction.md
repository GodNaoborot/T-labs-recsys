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
| final-comparison | Исправленное время запроса, 42/49/462 входа, три обучения, интервалы, test, важности |

Повторный поиск по деревьям может выбрать другое число входов. Оно сохраняется
в новом протоколе. Основное исправленное сравнение использует фиксированный
опубликованный список 462 формул, как и исходное исследование.

Весь граф доступен без Jupyter. По умолчанию команда только печатает план:

```bash
python task_1/src/run_all_experiments.py --run-name repeat_01 --download --include-test
```

Для выполнения добавить `--execute`. Отдельные ветви выбираются через
`--experiments`, например `--experiments bootstrap rolling`.

## Только основное сравнение

Без нового исторического поиска и Optuna:

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

На полном журнале повторены история, EDA, samples, ручные признаки,
train/validation DFS130 и подготовка Pool. Все значения истории и ручных
признаков совпали; DFS-матрицы совпали побайтно. Новые validation Pool дали
точно те же прогнозы двух сохранённых моделей. Полный повтор всех исследовательских обучений в чистой копии
пока не выполнен. [Фактические проверки](provenance/audit_reproduction_20261025.json).

Метрики сохранённого исследования: [основной протокол](c4.md),
[test](c4-test.md), [выбранные признаки](feature-analysis.md).
