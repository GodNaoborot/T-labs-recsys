"""Печать Markdown-сводки завершенных окон без изменения README и ноутбуков."""
import argparse
import pandas as pd

from rolling_windows import ROOT, RESULTS, WINDOWS, RUN_NAME, RUN_HELP, window_spec


def result_view(summary):
    """Свести две длины блока в одну строку на окно и objective."""
    bounds = summary.pivot(index=['window', 'objective'], columns='block_days', values='upper_bound')
    bounds = bounds.rename(columns={3: 'upper_3', 4: 'upper_4'})
    values = summary.groupby(['window', 'objective']).first()
    view = values[['base_ndcg', 'dfs_ndcg', 'relative_loss', 'groups']].join(bounds).reset_index()
    view['passes_both_lengths'] = view[['upper_3', 'upper_4']].lt(.03).all(axis=1)
    for name in ['relative_loss', 'upper_3', 'upper_4']:
        view[name + '_pct'] = 100 * view[name]
    return view


def markdown_table(view):
    """Вывести все результаты с единым округлением, без выбора удачных окон."""
    rows = ['| Train | Val | Модель | NDCG manual | NDCG DFS | Потеря | U95 L3 | U95 L4 | Обе проверки |',
            '| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |']
    specs = {window_spec(*bounds)['id']: window_spec(*bounds) for bounds in WINDOWS}
    for value in view.itertuples():
        spec = specs[value.window]
        train, val = ('–'.join(map(str, spec[key])) for key in ('train_days', 'val_days'))
        model = 'YetiRank' if value.objective == 'ranker' else 'MultiClass'
        result = 'U95 < 3%' if value.passes_both_lengths else 'H0 не отвергнута для всех блоков'
        rows.append(f'| {train} | {val} | {model} | {value.base_ndcg:.5f} | {value.dfs_ndcg:.5f} | '
                    f'{value.relative_loss_pct:.3f}% | {value.upper_3_pct:.3f}% | '
                    f'{value.upper_4_pct:.3f}% | {result} |')
    return '\n'.join(rows)


def report_text():
    """Явно отделить незавершенный запуск от финального результата пяти окон."""
    path = RESULTS / 'summary.csv'
    count = pd.read_csv(path)['window'].nunique() if path.exists() else 0
    text = ['## Переобучение на пяти временных окнах',
            f'Завершено окон: **{count} из {len(WINDOWS)}**. До завершения всех окон общий вывод не делается.',
            'Train содержит 45 дней, val — следующие 15 дней; дни в таблице нумеруются с 1 включительно. '
            'На каждом окне заново обучаются manual/DFS YetiRank и MultiClass с прежними параметрами. '
            'Признаки используют всю доступную прошлую историю, включая период до train; GAP 1800 с.',
            'На каждом val выполняются 20 000 парных циклических bootstrap-реплик для блоков 3 и 4 дня. '
            'Общий критерий для objective: U95 < 3% во всех 5 окнах при обеих длинах.']
    if path.exists():
        text.append(markdown_table(result_view(pd.read_csv(path))))
    text.extend(report_limits())
    return '\n\n'.join(text)


def report_limits():
    """Описать пределы вывода и указать артефакты именно выбранного запуска."""
    location = RESULTS.relative_to(ROOT.parent).as_posix()
    environment = f'TLAB_ROLLING_RUN={RUN_NAME} ' if RUN_NAME is not None else ''
    return ['Bootstrap внутри каждого окна условен на его обученных моделях. Пересекающиеся train '
             'делают результаты зависимыми; это проверка временной устойчивости данного протокола, '
             'а не доказательство для всей популяции Yambda. Последний val пересекается с исходным val. '
             'Показаны все окна и обе длины блока; индивидуальные границы не являются совместным 95% интервалом.',
             f'Протокол и артефакты: [rolling_windows]({location}/). '
             f'Просмотр статуса из task_1: `{environment}.venv-featuretools/bin/python '
             'src/run_rolling_windows.py --status`. Завершенный прогон только читается; '
             'продолжение частичного прогона требует точного совпадения protocol.json.']


def main():
    """Напечатать публичную сводку; сохранение текста задает вызывающая сторона."""
    parser = argparse.ArgumentParser(description=__doc__, epilog=RUN_HELP,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.parse_args()
    print(report_text())


if __name__ == '__main__':
    main()
