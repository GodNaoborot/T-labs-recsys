"""Собрать исследовательский отчёт из опубликованных таблиц, без обучения.

python task_1/reproduction/build_editorial_report.py --figures-only
python task_1/reproduction/build_editorial_report.py

Для графиков нужны matplotlib и pandas, для PDF - reportlab и шрифты DejaVu Sans, Serif и Sans Mono.
Графики можно подготовить отдельно, затем собрать PDF без matplotlib.
"""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import os
from xml.sax.saxutils import escape

ROOT = Path(__file__).resolve().parents[2]
CONTROL = Path('task_1/results/c4/no_current_channel')
FIG = Path('docs/figures/research_editorial')
BLUE, ORANGE, INK = '#31596C', '#BA5F42', '#192D30'
PAPER = '#F6F3EB'


def records(root, relative):
    with (root / relative).open(encoding='utf-8', newline='') as stream:
        return list(csv.DictReader(stream))


def load(root, relative):
    return json.loads((root / relative).read_text(encoding='utf-8'))


def figures(root):
    """Графики используют только небольшие сохранённые CSV."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import sys
    sys.path.insert(0, str(root / 'task_1/src'))
    from plot_final_intervals import plot_intervals
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                         'axes.spines.top': False, 'axes.spines.right': False,
                         'axes.labelcolor': INK, 'text.color': INK,
                         'axes.titleweight': 'normal', 'savefig.facecolor': PAPER, 'figure.facecolor': PAPER, 'axes.facecolor': PAPER})
    out = root / FIG
    out.mkdir(parents=True, exist_ok=True)

    data = pd.read_csv(root / 'docs/results/dataset-analysis/ratio_counts.csv')
    grouped = data.groupby('played_ratio_pct')['count'].sum()
    fig, ax = plt.subplots(figsize=(9, 3.1))
    ax.bar(grouped.index, grouped / grouped.sum() * 100, width=1, color=BLUE)
    ax.set(xlim=(-1, 160), xlabel='Доля прослушивания, %', ylabel='Доля listen-событий, %',
           title='Ранние пропуски и прослушивания до конца')
    ax.axvline(35, color=ORANGE, linestyle='--', label='Порог разметки 35%')
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out / 'listening_ratio.png', dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 2.25))
    ax.set(xlim=(75, 175), ylim=(-.9, 1.3))
    ax.axhline(0, color='#A9B5C0', linewidth=2)
    for x, color, label in [(100, BLUE, 'История: 100 < t'),
                             (120, '#8998A5', 'Граница t = 120\nсобытие исключено'),
                             (150, ORANGE, 'Целевой пакет: 150')]:
        ax.scatter([x], [0], s=170, color=color, zorder=3)
        ax.text(x, .28, label, ha='center', va='bottom', fontsize=10)
        ax.text(x, -.24, str(x), ha='center', va='top', fontsize=10)
    ax.annotate('', xy=(150, -.7), xytext=(120, -.7),
                arrowprops={'arrowstyle': '->', 'color': ORANGE, 'lw': 2})
    ax.text(135, -.9, 'Предсказываем следующий пакет', ha='center', fontsize=9)
    ax.axis('off')
    fig.tight_layout()
    fig.savefig(out / 'time_protocol.png', dpi=200)
    plt.close(fig)

    entities = pd.read_csv(root / 'docs/results/feature-analysis/entity_summary.csv')
    labels = ['Пользователь', 'Трек', 'Пользователь - трек', 'Пользователь - исполнитель']
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.2))
    for ax, column, title, suffix in [(axes[0], 'feature_count', 'Число отобранных признаков', ''),
                                     (axes[1], 'importance_sum', 'Суммарная важность модели', '%')]:
        y = np.arange(4)
        values = entities[column].to_numpy()
        ax.barh(y, values, color=[BLUE, '#6D95B3', ORANGE, '#E4A980'])
        ax.set_yticks(y, labels if ax is axes[0] else [''] * 4)
        ax.invert_yaxis()
        ax.set_title(title, fontsize=10)
        ax.set_xlim(0, values.max() * 1.25)
        for i, v in enumerate(values):
            ax.text(v + values.max() * .025, i, f'{v:.2f}{suffix}' if suffix else str(int(v)), va='center')
        ax.grid(axis='x', alpha=.15)
    fig.tight_layout()
    fig.savefig(out / 'entities.png', dpi=200)
    plt.close(fig)

    windows = pd.read_csv(root / 'docs/results/feature-analysis/window_summary.csv')
    finite = windows[windows.window_days.notna()]
    fig, ax = plt.subplots(figsize=(9, 3))
    ax.bar(finite.window_days, finite.importance_sum, color=BLUE)
    ax.set(xlabel='Длина окна, дни', ylabel='Сумма важностей, %',
           title='Среди окон заметнее короткая история')
    ax.set_xticks([1, 3, 7, 14, 21, 30])
    ax.text(.98, .93, f'Полная история: {windows.iloc[0].importance_sum:.2f}%\nпоказана отдельно от окон',
            transform=ax.transAxes, ha='right', va='top', fontsize=9)
    fig.tight_layout()
    fig.savefig(out / 'windows.png', dpi=200)
    plt.close(fig)

    intervals = pd.read_csv(root / CONTROL / 'bootstrap_summary.csv', float_precision='round_trip')
    pairs = [('manual48_no_current_channel', 'D462-t', 'Ручные48 - DFS462')]
    fig = plot_intervals(intervals[intervals.split == 'validation'],
                         intervals[intervals.split == 'test'], pairs=pairs, colors=(BLUE, ORANGE))
    fig.savefig(out / 'intervals.png', dpi=190)
    plt.close(fig)

def report(root, output):
    """Таблицы читаются из файлов; никакие метрики заново не оцениваются."""
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
    from reportlab.platypus import Flowable
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Table,
                                   TableStyle, Image, PageBreak, KeepTogether)

    font_candidates = [Path(os.environ.get('REPORT_FONT_DIR', '')),
                       Path('/usr/share/fonts/truetype/dejavu'),
                       Path(os.environ.get('WINDIR', 'C:/Windows')) / 'Fonts',
                       Path('//wsl.localhost/Ubuntu/usr/share/fonts/truetype/dejavu')]
    font_dir = next((p for p in font_candidates if (p / 'DejaVuSans.ttf').exists()), None)
    if font_dir is None:
        raise FileNotFoundError('Укажите REPORT_FONT_DIR с DejaVuSans.ttf и DejaVuSans-Bold.ttf')
    pdfmetrics.registerFont(TTFont('Report', str(font_dir / 'DejaVuSans.ttf')))
    pdfmetrics.registerFont(TTFont('ReportBold', str(font_dir / 'DejaVuSans-Bold.ttf')))
    pdfmetrics.registerFontFamily('Report', normal='Report', bold='ReportBold', italic='Report', boldItalic='ReportBold')
    pdfmetrics.registerFont(TTFont('ReportSerif', str(font_dir / 'DejaVuSerif.ttf')))
    pdfmetrics.registerFont(TTFont('ReportSerifBold', str(font_dir / 'DejaVuSerif-Bold.ttf')))
    pdfmetrics.registerFont(TTFont('ReportMono', str(font_dir / 'DejaVuSansMono.ttf')))
    pdfmetrics.registerFontFamily('ReportSerif', normal='ReportSerif', bold='ReportSerifBold',
                                italic='ReportSerif', boldItalic='ReportSerifBold')
    ink = colors.HexColor(INK)
    styles = {
        'body': ParagraphStyle('body', fontName='Report', fontSize=9.9, leading=14.9,
                               textColor=ink, spaceAfter=8),
        'small': ParagraphStyle('small', fontName='Report', fontSize=8.1, leading=11.7,
                                textColor=colors.HexColor('#526475'), spaceAfter=6),
        'title': ParagraphStyle('title', fontName='ReportSerif', fontSize=26, leading=32,
                                textColor=ink, spaceAfter=14),
        'h1': ParagraphStyle('h1', fontName='ReportSerif', fontSize=22, leading=27,
                             textColor=ink, spaceAfter=12),
        'h2': ParagraphStyle('h2', fontName='ReportBold', fontSize=10.8, leading=15,
                             textColor=ink, spaceBefore=11, spaceAfter=7),
        'cell': ParagraphStyle('cell', fontName='Report', fontSize=8.4, leading=12, textColor=ink),
        'head': ParagraphStyle('head', fontName='ReportBold', fontSize=8.3, leading=11.5,
                               textColor=colors.white),
    }
    story = []
    pagenames = []

    def p(text, style='body'):
        # Типографические тире заменены обычными дефисами для переносимого рендера.
        text = text.replace('\u2011', '-').replace('−', '-').replace('–', '-').replace('—', '-')
        story.append(Paragraph(text, styles[style]))

    def h(text):
        p(text, 'h2')

    def page(title, first=False):
        if not first:
            story.append(PageBreak())
        story.append(Mark(len(pagenames)))
        pagenames.append(title)
        p(title, 'title' if first else 'h1')

    styles['num'] = ParagraphStyle('num', parent=styles['cell'], fontName='ReportMono', alignment=TA_RIGHT)
    styles['head'].textColor = ink

    def table(rows, widths):
        def numeric(value):
            try:
                float(str(value).replace(' ', '').replace('%', ''))
                return True
            except ValueError:
                return False
        right = [all(numeric(row[j]) for row in rows[1:]) for j in range(len(rows[0]))]
        converted = [[Paragraph(escape(str(value)).replace('\n', '<br/>'),
                                styles['head' if i == 0 else ('num' if right[j] else 'cell')])
                      for j, value in enumerate(row)] for i, row in enumerate(rows)]
        t = Table(converted, colWidths=widths, repeatRows=1, hAlign='LEFT')
        t.setStyle(TableStyle([
            ('VALIGN', (0, 0), (-1, -1), 'TOP'),
            ('LEFTPADDING', (0, 0), (-1, -1), 5), ('RIGHTPADDING', (0, 0), (-1, -1), 5),
            ('TOPPADDING', (0, 0), (-1, -1), 6), ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
            ('LINEABOVE', (0, 0), (-1, 0), 1.0, ink),
            ('LINEBELOW', (0, 0), (-1, 0), .5, ink),
            ('LINEBELOW', (0, -1), (-1, -1), .8, ink)]))
        story.extend([t, Spacer(1, 9)])

    styles['caption'] = ParagraphStyle('caption', parent=styles['small'],
                                           fontSize=7.7, leading=11, spaceAfter=10)

    def box(kind, text):
        # Один спокойный акцент для идеи; выводы остаются обычными тезисами.
        if kind == 'Результат':
            p(text)
            return
        note = Paragraph(text.replace('−', '-').replace('–', '-').replace('—', '-'),
                         styles['body'])
        t = Table([[note]], colWidths=[A4[0] - 96], hAlign='LEFT')
        t.setStyle(TableStyle([
            ('LINEBEFORE', (0, 0), (0, -1), 1.7, colors.HexColor(ORANGE if kind == 'Ограничение' else BLUE)),
            ('LEFTPADDING', (0, 0), (-1, -1), 12),
            ('RIGHTPADDING', (0, 0), (-1, -1), 5),
            ('TOPPADDING', (0, 0), (-1, -1), 4),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 2)]))
        story.extend([t, Spacer(1, 10)])

    class Mark(Flowable):
        def __init__(self, index):
            super().__init__()
            self.index = index
            self.width = self.height = 0

        def draw(self):
            state['section'] = self.index

    state = {'section': -1}

    class Cover(Flowable):
        """Обложка с типографической композицией размеров двух наборов."""
        width, height = 499, 660

        def drawOn(self, canvas, x, y, _sW=0):
            state['section'] = -1
            c = canvas
            c.saveState()
            c.setFillColor(colors.HexColor(INK))
            c.rect(0, 0, *A4, fill=1, stroke=0)
            cream = colors.HexColor(PAPER)
            muted = colors.HexColor('#9BB1A8')
            c.setFillColor(muted)
            c.setFont('Report', 8)
            c.drawString(48, 786, 'ИССЛЕДОВАНИЕ / РЕКОМЕНДАТЕЛЬНЫЕ СИСТЕМЫ')
            c.setStrokeColor(colors.HexColor('#4F6760'))
            c.line(48, 770, 547, 770)
            c.setFillColor(cream)
            c.setFont('ReportSerif', 38)
            for line, baseline in [('Автоматические', 683), ('признаки для', 635), ('рекомендаций', 587)]:
                c.drawString(48, baseline, line)
            c.setFont('Report', 12)
            c.setFillColor(muted)
            c.drawString(50, 544, 'Yambda flat 50m · Deep Feature Synthesis · CatBoost')
            # Линия времени и четыре ветви формул - схематический мотив, не данные.
            c.setStrokeColor(colors.HexColor('#52756A'))
            for i in range(4):
                y0 = 460 - i * 22
                c.line(48, y0, 155 + i * 40, y0)
                c.line(155 + i * 40, y0, 310, 427)
            c.line(310, 427, 547, 427)
            for x0 in [330, 365, 400, 435, 470, 505, 540]:
                c.setFillColor(colors.HexColor('#C9D7BC') if x0 in [365, 470] else colors.HexColor('#52756A'))
                c.rect(x0 - 2, 425, 4, 4, fill=1, stroke=0)
            c.setFillColor(muted)
            c.setFont('Report', 8)
            c.drawString(48, 353, 'СВЯЗИ МЕЖДУ СУЩНОСТЯМИ → ФОРМУЛЫ → РАНЖИРОВАНИЕ')
            c.setStrokeColor(colors.HexColor('#4F6760'))
            c.line(48, 311, 547, 311)
            for x0, number, label in [(48, '48', 'ручных признаков'), (230, '462', 'признака DFS'), (426, '30', 'дней в переборе')]:
                c.setFillColor(cream)
                c.setFont('ReportMono', 42)
                c.drawString(x0, 244, number)
                c.setFillColor(muted)
                c.setFont('Report', 8)
                c.drawString(x0, 218, label)
            c.setFillColor(cream)
            c.setFont('Report', 10)
            c.drawString(48, 143, 'Автоматизировать выбор признаков.')
            c.drawString(48, 126, 'Проверить качество. Разобраться со временем.')
            c.setFillColor(muted)
            c.setFont('Report', 8)
            c.drawString(48, 48, 'GODNAOBOROT / ИССЛЕДОВАТЕЛЬСКИЙ ОТЧЁТ')
            c.drawRightString(547, 48, 'YAMBDA / FLAT 50M')
            c.restoreState()

    class ScoreStrip(Flowable):
        def __init__(self):
            super().__init__()
            self.width, self.height = 499, 83

        def draw(self):
            c = self.canv
            for x0, name, value, color in [
                (0, 'Ручные48', metric(manual, 'test')['NDCG@10'], BLUE),
                (255, 'DFS462', metric(dfs, 'test')['NDCG@10'], ORANGE)]:
                c.setFillColor(colors.HexColor(color))
                c.setFont('ReportBold', 8)
                c.drawString(x0, 69, name)
                c.setFont('ReportMono', 26)
                c.drawString(x0 - 1, 34, f'{float(value):.6f}')
                c.setFont('Report', 7.5)
                c.setFillColor(colors.HexColor('#69756F'))
                c.drawString(x0, 15, 'NDCG@10 / повторный test')

    story.extend([Cover(), PageBreak()])

    def image(name, width=490):
        from PIL import Image as PILImage
        path = root / FIG / name
        with PILImage.open(path) as raw:
            w, height = raw.size
        story.append(Image(str(path), width=width, height=width * height / w))
        captions = {
            'listening_ratio.png': '01 / Распределение доли прослушивания',
            'time_protocol.png': '02 / Момент запроса и доступная история',
            'entities.png': '03 / Состав 462 признаков и важности по сущностям',
            'windows.png': '04 / Важности формул с разной длиной окна',
            'intervals.png': '05 / Парные интервалы разницы NDCG@10',
        }
        p(captions.get(name, name), 'caption')

    def source(text):
        p('Источник: ' + escape(text), 'small')

    def num(value, digits=6):
        return f'{float(value):.{digits}f}'

    comp = records(root, CONTROL / 'comparison.csv')
    ci = records(root, CONTROL / 'bootstrap_summary.csv')
    snap = load(root, 'docs/results/dataset-analysis/snapshot.json')
    entity = records(root, 'docs/results/feature-analysis/entity_summary.csv')
    def metric(variant, split):
        return next(x for x in comp if x['variant'] == variant and x['split'] == split)
    manual, dfs = 'manual48_no_current_channel', 'D462-t'

    page('Автоматическое создание\nпризнаков для RecSys', first=True)
    p('Исследование на Yambda flat 50m', 'h2')
    h('Постановка задачи')
    p('Для пары пользователь-трек нужно оценить релевантность по истории, доступной до момента запроса. '
      'Проверяю, можно ли автоматизировать построение признаков для CatBoostRanker и получить качество, '
      'близкое к расширенному ручному набору. Отдельный вопрос - как выбирать временные окна, '
      'не ограничиваясь несколькими привычными значениями.')
    p('<b>Идея:</b> задать сущности и допустимые операции, перечислить формулы DFS с окнами 1-30 дней '
      'и признаками свежести, затем сократить набор по обученной модели. В итоговом сравнении '
      'используются 48 ручных и 462 автоматических признака на одинаковых строках.')
    h('Основной результат')
    table([['Модель', 'Validation NDCG@10', 'Повторный test NDCG@10'],
           ['Ручные48', num(metric(manual, 'validation')['NDCG@10']), num(metric(manual, 'test')['NDCG@10'])],
           ['DFS462', num(metric(dfs, 'validation')['NDCG@10']), num(metric(dfs, 'test')['NDCG@10'])]], [130, 180, 180])
    p('Точечное качество близко. DFS не показал преимущества перед расширенным ручным набором, '
      'но позволил автоматически построить интерпретируемые признаки и указал на полезность недавней '
      'истории пользователя. Близость оценок сама по себе не доказывает эквивалентность методов.')
    p('Основную пару ручные48-DFS462 выбрали после открытия test: у ручного контроля исключили текущий '
      'канал взаимодействия, доступность которого в момент прогноза не обоснована. Поэтому повторный '
      'test здесь служит исследовательской проверкой, а не новым слепым подтверждением.')
    h('Что входит в отчёт')
    p('Данные и метки; диагностика времени; подход к признакам; схема эксперимента; метрики и интервалы; '
      'вычислительная стоимость; выводы и воспроизведение. Результаты прежнего некорректного cutoff '
      'используются только для объяснения найденной проблемы.')
    source('no_current_channel/comparison.csv; подробные пути к данным отчёта приведены в конце.')

    page('Данные и целевая переменная')
    p('Использована версия Yambda flat 50m, а не весь Yambda-5B [1, 2]. '
      'Исходный журнал содержит прослушивания и реакции: like, dislike, unlike, undislike. '
      'Идентификаторы пользователей, треков и исполнителей позволяют строить прямые и косвенные связи.')
    table([['Характеристика', 'В выбранной версии'],
           ['Все события', f"{snap['rows']:,}".replace(',', ' ')],
           ['Прослушивания / реакции', f"{snap['listen_rows']:,} / {snap['reaction_rows']:,}".replace(',', ' ')],
           ['Пользователи / треки', f"{snap['users']:,} / {snap['items']:,}".replace(',', ' ')],
           ['Полные дубликаты строк', f"{snap['duplicated_rows']:,}; сохранены".replace(',', ' ')]], [230, 260])
    h('Как получены метки')
    p('Готовой метки релевантности для выбранной задачи в данных нет. Для каждого прослушивания '
      'реакции той же пары привязываются к ближайшему следующему концу прослушивания, если попадают '
      'в восстановленный интервал. Приоритет разметки: <b>like - 2</b>; иначе '
      '<b>dislike или доля прослушивания меньше 35% - 0</b>; иначе <b>1</b>. '
      'Если привязаны обе реакции, приоритет у like. Unlike/undislike напрямую метку не задают.')
    image('listening_ratio.png')
    p('Распределение сосредоточено около ранних пропусков и конца трека. Порог 35% - ручная эвристика, '
      'а не выведенная из этого графика оптимальная граница. Значения выше 100% встречаются в исходном '
      'журнале. Сравнение относится к этой разметке поведения, а не к независимо измеренной полезности рекомендаций.')
    source('docs/results/dataset-analysis/{snapshot.json,ratio_counts.csv}; task_1/src/c4_targets.py.')

    page('Почему пришлось изменить время запроса')
    p('В ранней версии признаки рассчитывались к восстановленному началу прослушивания. '
      'Для его вычисления использовалась <b>текущая</b> доля прослушивания R. Таким образом, '
      'момент расчёта уже зависел от будущего исхода целевого события.')
    h('Механика канала')
    p('Сохранённая формула: <b>S = E - 5 - ceil(L * R / 100)</b>, где E - исходный timestamp '
      'прослушивания, L - длина трека в секундах. Она совпала с восстановленным временем на всех '
      '2 039 017 train- и 2 882 983 validation-строках прежнего сравнения.')
    p('Исходные времена и длины лежат на пятисекундной сетке. Поэтому остаток пользовательской '
      'свежести F по модулю 5 зависел от текущего R: <b>F mod 5 = (-ceil(L * R / 100)) mod 5</b>. '
      'Даже если R не подать в модель напрямую, он может попасть в неё через временной признак.')
    h('Эксперименты диагностики')
    table([['Проверка на прежней validation', 'Наблюдение'],
           ['Таблица частот по фазе F mod 5, построенная только на train', 'ROC AUC для label=0 против остальных: 0.7542'],
           ['Перестановка свежести внутри (L, floor(F/5)); ручная модель заморожена', 'NDCG: 0.849833 → 0.806796'],
           ['Та же перестановка; DFS-модель заморожена', 'NDCG: 0.847202 → 0.802563']], [305, 185])
    p('Перестановка не использовала метки, меняла свежесть максимум на 4 секунды и оставляла '
      'прочие входы фиксированными. Это показывает чувствительность обученных моделей к найденному '
      'каналу. Падение не является оценкой причинной доли утечки и не равно качеству после переобучения.')
    h('Что изменено')
    p('Момент запроса заменён предыдущим исходным timestamp пользователя. После этого заново построены '
      'целевые строки, признаки, split и группы и переобучены модели. Восстановленное начало больше '
      'не задаёт cutoff признаков; оно сохранено в привязке реакций. Старые высокие метрики не используются '
      'как основной результат исследования.')
    source('task_1/results/analysis/start_ts_{formula,phase,phase_permutation,dfs_phase_permutation}_20261024_v1; docs/start-time-audit.md.')

    page('Подход: прогноз следующего пакета')
    p('Прослушивания пользователя с одинаковым исходным timestamp объединяются в пакет. '
      'Для целевого пакета момент запроса <b>t</b> равен предыдущему <b>строго меньшему</b> timestamp '
      'того же пользователя. Первый пакет исключается. История содержит только исходные события '
      'с timestamp &lt; t; пакет прямо на границе t тоже исключён. Временное окно: <b>[t-w, t)</b>.')
    image('time_protocol.png')
    p('Для равных исторических timestamp порядок задаёт исходный ID строки, без сортировки по метке '
      'или доле прослушивания. Внутренний порядок событий в пятисекундном бине неизвестен. '
      'Интерпретация исходного timestamp как времени доступности результата остаётся допущением: '
      'карточка данных однозначно не описывает момент отправки и доступности логов.')
    table([['Часть', 'Cutoff, дни с начала', 'Строки', 'Группы'],
           ['Train', '[225, 270)', '2 021 867', '40 000'],
           ['Validation', '[270, 285)', '2 879 535', '57 097'],
           ['Повторный test', 'От 285 до конца', '3 160 068', '61 964']], [96, 150, 125, 119])
    p('В train и validation исходный timestamp целевого пакета должен быть меньше правой границы минус '
      'GAP 1800 секунд. У test конечного GAP нет. Из train выбраны 40 000 целых смешанных групп '
      'с seed 42; validation и test взяты полностью в рамках правил отбора.')
    p('Если прикреплённый к цели like/dislike уже произошёл раньше t, строка исключается: '
      '448 в train, 170 в validation и 191 в test. Это правило зависит от исхода и меняет популяцию оценки.')
    p('История начинается с начала журнала и обновляется внутри validation/test по мере доступности '
      'событий. Веса модели фиксированы. Текущий label и доля прослушивания не входят в признаки. '
      'Длина целевого трека считается заранее известным каталоговым атрибутом; текущий is_organic исключён.')
    source('task_1/src/c4_targets.py; docs/c4.md; docs/c4-test.md; no_current_channel/result.json.')

    page('Подход: ручные и автоматические признаки')
    h('Ручные48')
    p('Ручной набор содержит длину трека и 47 исторических признаков. К исходным item-, user-item- '
      'и user-artist-признакам после анализа DFS добавлены 7 признаков недавнего поведения пользователя. '
      'Таким образом, это усиленный контроль, сформированный с учётом результатов автоматического поиска.')
    table([['История', 'Примеры и окна'],
           ['Трек', 'Число прослушиваний, skip-rate, like/listen, средняя доля прослушивания; 1/7/30/90 дней'],
           ['Пользователь-трек', 'Число прослушиваний и пропусков; 7/14/30/90 дней; затухание 1/7/30 дней; давность и состояние реакций'],
           ['Пользователь-исполнитель', 'Счётчики, пропуски и затухание по исполнителю'],
           ['Пользователь', 'Доля прослушивания и skip последнего события; skip-rate последних 5/20 событий и 24 часов; число событий за последние 24 часа; давность']], [145, 345])
    h('DFS462')
    p('Deep Feature Synthesis [3] строит формулы через связи между сущностями. Здесь используются '
      '<b>Users, Items, UserItem, UserArtist</b> и журнал Interactions. Сначала агрегируем доступную '
      'историю по ключу сущности, затем присоединяем результат к целевой строке. '
      'Так средняя историческая метка пары user-item становится признаком следующего взаимодействия.')
    p('Featuretools [4] создаёт определения формул. Собственный кумулятивный исполнитель считает '
      'поддерживаемую грамматику, переиспользуя состояние между запросами. Это ускорение выбранного '
      'подмножества формул, а не универсальная реализация всех возможностей библиотеки.')
    p('Числовые поля истории DFS: <b>label, track_length_seconds, is_organic</b>. Операции: count, '
      'mean, std, min, max, допустимые произведения и давность первого/последнего события. '
      'Историческая непрерывная доля прослушивания напрямую в DFS не включена, хотя ручной набор '
      'её использует. Исходные представления сравниваемых методов поэтому различаются.')
    source('task_1/src/rolling_manual_features.py; cumulative_features.py; docs/feature-analysis.md; no_current_channel/protocol.json.')

    page('Эксперименты: поиск и отбор формул')
    p('Для каждой из четырёх сущностей перечислены 103 базовые формулы в 31 режиме: вся история '
      'и окна от 1 до 30 дней. Добавлены два признака давности для каждой сущности и два текущих '
      'контекстных поля. Всего <b>4 × (103 × 31 + 2) + 2 = 12 782 кандидата</b>. '
      'Произведения агрегатов строятся внутри одного окна; сочетания разных окон не перебираются.')
    table([['Этап', 'Что сделано'],
           ['Подвыборка', '9 999 train-строк / 183 группы; 5 000 validation-строк / 96 групп'],
           ['Проверка top-k', 'Важность LossFunctionChange на train; размер набора сравнивался по validation NDCG'],
           ['Сокращение', 'Top-k не улучшил полный набор. Оставлены все 462 признака, использованные в разбиениях 91-деревной модели'],
           ['Итоговый протокол', 'Фиксированный список перенесён на исправленный cutoff без повторного отбора или тюнинга']], [122, 368])
    p('Отбор первоначально выполнялся при прежнем восстановленном cutoff. Поэтому 462 - '
      'фиксированный унаследованный набор, а не оптимальное число признаков для исправленного времени. '
      'Текущие контекстные поля в отобранные 462 не вошли; исторический is_organic сохранился.')
    image('entities.png')
    p('В набор вошли 132 пользовательских, 148 item-, 59 user-item- и 123 user-artist-признака. '
      'Число признаков не определяет их важность: пользовательская ветвь получила около половины '
      'суммарной PredictionValuesChange итогового DFS-ранкера.')
    p('Автоматизация перечисляет формулы и окна внутри заданной грамматики. Сущности, исходные поля, '
      'операции, временной протокол и вычислительный бюджет по-прежнему задаются вручную.')
    source('docs/results/feature-analysis/entity_summary.csv; docs/feature-analysis.md; temporal_topk_20261023_v2 и temporal_used_20261023_v1.')

    page('Какие признаки выбрала модель')
    image('windows.png')
    p('51 признак использует всю доступную историю, 411 - конечное окно. В отобранном наборе '
      'представлены все длины от 1 до 30 дней. Самый заметный оконный блок - один день '
      '(18.92% суммы важностей); вся история даёт 38.55%, включая признаки давности.')
    h('Примеры с понятным смыслом')
    top = records(root, 'docs/results/feature-analysis/top_features.csv')
    table([['Признак', 'Интерпретация', 'Важность'],
           ['Давность последнего события пользователя', 'Насколько недавно пользователь слушал музыку', f"{float(top[0]['importance']):.2f}%"],
           ['max(label) × mean(label), пользователь, 1 день', 'Сочетание лучшей и средней исторической реакции за сутки', f"{float(top[1]['importance']):.2f}%"],
           ['mean(label), пользователь-трек, вся история', 'Средняя оценка предыдущих прослушиваний этой пары', f"{float(top[2]['importance']):.2f}%"],
           ['mean(label), пользователь, 3 дня', 'Недавний общий характер реакций пользователя', f"{float(top[3]['importance']):.2f}%"]], [190, 225, 75])
    p('Средняя метка user-item считается по предыдущим доступным событиям пары, без текущей цели. '
      'Для неё полезны повторные прослушивания одного трека. Пользовательская свежесть и недавние '
      'пропуски описывают общий контекст поведения; именно этот блок затем усилил ручной контроль.')
    p('Важности - описание обученной модели [6]. Окна и агрегаты коррелируют, поэтому сумма важностей '
      'не измеряет причинный вклад и не доказывает оптимальность одного дня. Для такого вывода '
      'понадобилось бы отдельное сравнение с удалением/заменой блоков и новым закрытым периодом.')
    source('docs/results/feature-analysis/{window_summary.csv,top_features.csv,prediction_values_change.csv}.')

    page('Эксперименты: модель и оценка')
    h('Общий протокол обучения')
    p('Обе модели - CatBoostRanker с YetiRank [5]. Целевые строки, группы, метки и временные '
      'границы совпадают. Границы квантования обучаются на train. Для ручных48 физическая '
      'матрица содержит 49 колонок, но текущий is_organic исключён через ignored_features=[0].')
    table([['Параметр', 'Общее значение'],
           ['Глубина / learning_rate', '6 / 0.1'], ['Максимум деревьев / early stopping', '1600 / 80'],
           ['Seed / CPU-потоки', '42 / 4'], ['Bootstrap / subsample / rsm', 'MVS / 0.8 / 1.0'],
           ['border_count / random_strength', '254 / 1.0'], ['l2_leaf_reg', '1e-20; унаследованная конфигурация']], [250, 240])
    p('Параметры перенесены из предыдущих сравнений. После исправления cutoff новый поиск '
      'гиперпараметров не проводился. Это ограниченная проверенная конфигурация, а не гарантированно '
      'лучший бустинг для каждой системы признаков.')
    h('Метрики и кандидаты')
    p('<b>NDCG@10</b> считается с линейной релевантностью 0/1/2: '
      'DCG@10 = sum(label_i / log2(i+1)); NDCG = DCG / IDCG. '
      'Итог - среднее по группам с одинаковым весом. Точные ничьи score разрешаются общей '
      'фиксированной перестановкой seed 42. Встроенная early-stopping-метрика CatBoost разрешает их иначе.')
    p('Дополнительная метрика - <b>QueryAUC:type=Ranking</b> по исходным score. '
      'Она не подменяется бинарным ROC AUC. Основное сравнение относится к актуальным ранкерам.')
    p('Группа - пользователь и день cutoff: uid*1000 + floor(t/DAY). Удалены группы с постоянной '
      'меткой. Кандидаты - фактически наблюдавшиеся прослушивания, без negative sampling. '
      'Они относятся к разным моментам запроса одного дня. Это оценка ранжирования наблюдавшихся '
      'событий, а не выдачи всего каталога в один момент.')
    source('no_current_channel/protocol.json; task_1/src/train_dfs_catboost.py; c4_targets.py; документация CatBoost [5].')

    page('Результаты: ручные48 и DFS462')
    story.append(ScoreStrip())
    rows = [['Часть / модель', 'NDCG@10', 'QueryAUC']]
    for split, title in [('validation', 'Validation'), ('test', 'Повторный test')]:
        for variant, label in [(manual, 'ручные48'), (dfs, 'DFS462')]:
            rec = metric(variant, split)
            rows.append([f'{title}: {label}', num(rec['NDCG@10']), num(rec['QueryAUC'])])
    table(rows, [250, 120, 120])
    image('intervals.png')
    p('Разница определена как <b>ручные48 - DFS462</b>: на validation -0.00003418, '
      'на повторном test +0.00058196 NDCG. Для парного циклического блочного bootstrap по дням '
      'использованы 20 000 реплик, блоки 3 и 4 дня, seed 42 и двусторонние 95% percentile-интервалы.')
    rows = [['Часть', 'Блок', 'Нижняя граница', 'Верхняя граница']]
    for rec in ci:
        rows.append(['Val' if rec['split'] == 'validation' else 'Повторный test',
                     rec['block_days'] + ' дня', num(rec['ci_low'], 8), num(rec['ci_high'], 8)])
    table(rows, [130, 80, 140, 140])
    p('Validation-интервалы включают ноль. На test они положительны, но нижние границы '
      'практически нулевые. Это не основание заявлять устойчивое превосходство ручного метода '
      'или эквивалентность систем: пара выбрана после открытия test, seed обучения один, '
      'а интервалы условны на уже обученных моделях. Поправка на множественные сравнения не применялась.')
    source('task_1/results/c4/no_current_channel/{comparison.csv,bootstrap_summary.csv}.')

    page('Вычислительная стоимость')
    h('Наблюдавшееся время')
    # Значения времён генерации читаются ниже из небольших JSON с учётом их схемы.
    def seconds(name):
        obj = load(root, 'task_1/results/c4/c4_packet_20261024_v1/' + name)
        for key in ('elapsed_seconds', 'generation_seconds', 'wall_seconds'):
            if key in obj:
                return float(obj[key])
        raise KeyError(f'Нет времени генерации: {name}')
    generation = []
    for kind, label in [('manual', 'Ручная матрица: 49 колонок'), ('dfs', 'DFS: 462 колонки')]:
        generation.append([label, f"{seconds(f'generation_{kind}_train.json'):.2f} с",
                           f"{seconds(f'generation_{kind}_val.json'):.2f} с"])
    table([['Построение матрицы', 'Train', 'Validation']] + generation, [250, 120, 120])
    table([['Обучение', 'Fit, минуты', 'Деревья'],
           ['Ручные48', f"{float(metric(manual, 'validation')['fit_seconds']) / 60:.2f}", metric(manual, 'validation')['tree_count']],
           ['DFS462', f"{float(metric(dfs, 'validation')['fit_seconds']) / 60:.2f}", metric(dfs, 'validation')['tree_count']]], [250, 120, 120])
    p('Ручные48 переиспользуют матрицу49 с игнорируемой колонкой. Их fit выполнялся с пониженным '
      'приоритетом параллельно аудиту, поэтому эти времена не являются чистым сравнением скорости. '
      'Генерация DFS в наблюдавшемся прогоне дороже. Число колонок само по себе не измеряет '
      'потребление всей памяти WSL.')
    p('Этап train/validation трёх первоначальных исправленных моделей, включая построение признаков, '
      'квантование и сохранение, занял 67.04 минуты после подготовки истории. Пиковый anonymous RSS '
      'процессов этого этапа составил 5.74 GiB; это не вся память WSL. Test-этап занял 7.47 минуты. '
      'Дополнительный fit ручных48 учитывается отдельно.')
    p('Это подтверждает выполнимость выбранного ограниченного набора на миллионах целевых строк. '
      'Стоимость всех 12 782 кандидатов на полном объёме не измерена: широкий поиск выполнялся '
      'на малой подвыборке. Перенос на миллиарды событий этими замерами не проверен.')
    source('generation_{manual,dfs}_{train,val}.json; c4_packet_20261024_v1/result.json; c4_test_20261024_v1/result.json; no_current_channel/comparison.csv.')

    page('Воспроизведение эксперимента')
    p('Репозиторий: <link href="https://github.com/GodNaoborot/T-labs-recsys" color="#285A83">'
      'github.com/GodNaoborot/T-labs-recsys</link>. Главные инструкции - '
      '<link href="https://github.com/GodNaoborot/T-labs-recsys/blob/main/docs/reproduction.md" color="#285A83">'
      'docs/reproduction.md</link>, итоговый расчёт - '
      '<link href="https://github.com/GodNaoborot/T-labs-recsys/blob/main/task_1/notebooks/final-comparison.ipynb" color="#285A83">'
      'final-comparison.ipynb</link>. '
      'Быстрый режим ноутбуков читает опубликованные таблицы и строит графики без моделей и нового обучения.')
    h('Установка и запуск отчётов')
    p('Python 3.12, Linux или WSL/Ubuntu. Команды выполняются из корня клонированного проекта. '
      'Создаётся отдельное окружение; выполненные копии ноутбуков сохраняются в отдельную папку.')
    table([['Шаг', 'Команда'],
           ['Окружение', 'python3.12 -m venv task_1/.venv-featuretools'],
           ['Активация', 'source task_1/.venv-featuretools/bin/activate'],
           ['Зависимости', 'python -m pip install -r task_1/requirements.txt'],
           ['Проверка установки', 'python -m pip check'],
           ['Отчёты без fit', 'python task_1/src/run_report_notebooks.py\n  --output-dir task_1/.validation/reports']], [122, 368])
    h('Повтор из исходных данных')
    p('Закреплённая версия данных скачивается по манифесту. Новый run-name сохраняет расчёт отдельно '
      'от опубликованных результатов. Следующая команда запускает обучение; --include-test '
      'означает повторную оценку уже открытого периода.')
    p('<font size="8.4">python task_1/src/run_all_experiments.py --run-name repeat_main<br/>'
      '&nbsp;&nbsp;--experiments channel-control --download --include-test --execute</font>')
    p('В терминале это одна команда. Граф готовит историю, целевые строки и матрицы, обучает '
      'три первоначальных модели исправленного протокола и контроль ручные48. Исторический '
      'поиск и Optuna в эту ветвь не входят. Для нового расчёта нужна отдельная чистая копия; '
      'ориентир ресурсов инструкции - 12 GiB RAM и не менее 40 GB свободного диска.')
    h('Что проверено фактически')
    p('В отдельной чистой копии повторены исходные три модели исправленного протокола: шесть матриц '
      'побайтно совпали, validation/test-предсказания и метрики воспроизвелись. Для ручных48 опубликованы '
      'исходные fit/evaluate-скрипты, параметры и хеши; свежие test-предсказания совпали точно. '
      'Новый переносимый CLI проверен на синтетическом цикле, но его полный повтор на реальных данных '
      'не выполнялся. Все исторические обучения и Optuna заново не повторялись.')
    source('docs/reproduction.md; docs/results/reproduction/main_verification.json; task_1/reproduction/channel_control_{fit,evaluate}.py.')

    page('Выводы и границы исследования')
    h('Что удалось проверить')
    p('<b>1. Автоматическое построение формул работает на выбранном масштабе.</b> '
      'Перечислены 12 782 кандидата; фиксированный набор из 462 признаков посчитан к миллионам '
      'целевых строк с переиспользованием исторических статистик.')
    p('<b>2. Качество близко к расширенному ручному контролю.</b> '
      'На validation оценки практически совпали; на повторном test ручные48 выше на 0.000582 NDCG. '
      'Доказательства превосходства DFS или формальной эквивалентности нет.')
    p('<b>3. DFS полезен как инструмент поиска.</b> Он выявил заметные пользовательские признаки '
      'свежести и недавних реакций. Их смысл можно объяснить, а несколько простых признаков '
      'позволили усилить ручной набор. Это наблюдение о ходе исследования, не причинная абляция '
      'каждого найденного признака.')
    p('<b>4. Проверка времени оказалась существеннее прироста метрики.</b> '
      'Восстановленный cutoff зависел от текущего исхода; найденный канал исключён из нового '
      'момента расчёта. Предыдущие высокие оценки не считаются валидным подтверждением качества.')
    h('Что ограничивает вывод')
    p('Один датасет, один актуальный временной split и seed обучения; повторно использованные '
      'validation/test; подбор основной пары после открытия test. Старые дополнительные окна '
      'не являются повторением финального исправленного эксперимента.')
    p('Метки получены эвристически; восстановленное время остаётся в привязке реакций. '
      'Доступность результатов в исходный timestamp и каталоговой длины - допущения. '
      'Отбор смешанных user-day групп и исключение ранних реакций зависят от исходов. '
      'Оценка не распространяется автоматически на полный каталог и реальные выдачи.')
    p('Наборы используют разные числовые представления истории. Формулы и гиперпараметры '
      'перенесены из прежних этапов; новый оптимальный набор не искался. Не проверялись '
      'миллиарды событий, другие датасеты или широкий набор обучающих seed.')
    h('Что проверять дальше')
    p('Зафиксировать текущие два метода и оценить их на новом закрытом периоде с несколькими seed; '
      'согласовать семантику времени с владельцами данных; отдельно проверить непрерывную долю '
      'прослушивания в грамматике DFS. Эти проверки предложены как продолжение и не включены '
      'в полученные результаты.')

    p('Дальше можно совместить LLM с DFS: по схеме сущностей и описанию полей LLM предлагает новые связи и формулы, а существующий конвейер проверяет и рассчитывает поддерживаемые признаки. Кандидаты проходят проверку доступности во времени. Сравнить DFS и LLM+DFS стоит при одинаковом бюджете на новом закрытом периоде. Этот вариант пока не реализован и не оценивался.')

    page('Источники и проверяемые артефакты')
    references = [
        ('1', 'Ploshkin A. et al. Yambda-5B: A Large-Scale Multi-modal Dataset for Ranking And Retrieval. 2025, v2.', 'https://arxiv.org/abs/2505.22238v2', 'Описание исходного датасета и его полей; здесь выбран собственный протокол оценки.'),
        ('2', 'Yandex. Yambda dataset card; закреплённая ревизия dd6f3a19eef5866e346c3270e098baa641a44948.', 'https://huggingface.co/datasets/yandex/yambda/blob/dd6f3a19eef5866e346c3270e098baa641a44948/README.md', 'Схема, варианты данных и лицензия Apache-2.0. Исходный Yambda не включается в кодовую MIT-лицензию проекта.'),
        ('3', 'Kanter J. M., Veeramachaneni K. Deep Feature Synthesis: Towards Automating Data Science Endeavors. DSAA, 2015.', 'https://www.maxkanter.com/papers/DSAA_DSM_2015.pdf', 'Реляционное построение признаков через связи и композиции операций.'),
        ('4', 'Featuretools. Deep Feature Synthesis API.', 'https://docs.featuretools.com/en/stable/generated/featuretools.dfs.html', 'Определения DFS-формул и параметры генерации.'),
        ('5', 'CatBoost. Ranking: objectives and metrics.', 'https://catboost.ai/docs/en/concepts/loss-functions-ranking', 'YetiRank, NDCG и QueryAUC Ranking.'),
        ('6', 'CatBoost. Feature importance.', 'https://catboost.ai/docs/en/concepts/fstr', 'Интерпретация PredictionValuesChange и LossFunctionChange.'),
    ]
    for number, title, url, use in references:
        p(f'<b>[{number}] {escape(title)}</b> ' + escape(use))
        # Короткая видимая подпись сохраняет длинный pinned URL в кликабельной аннотации.
        p(f'<link href="{escape(url)}" color="#285A83">Открыть первоисточник</link>', 'small')
    h('Откуда взяты числа и рисунки')
    p('Все пути ниже относительны корню репозитория. Сборщик читает небольшие JSON/CSV, '
      'а не веса моделей или матрицы взаимодействий.', 'small')
    for title, path in [
            ('Главная таблица и интервалы', 'task_1/results/c4/no_current_channel/'),
            ('Диагностика времени', 'task_1/results/analysis/; docs/start-time-audit.md'),
            ('Выбранные формулы и важности', 'docs/results/feature-analysis/'),
            ('Агрегаты анализа данных', 'docs/results/dataset-analysis/'),
            ('Полный повтор и область проверки', 'docs/results/reproduction/main_verification.json'),
            ('Исходные скрипты контроля', 'task_1/reproduction/channel_control_fit.py; channel_control_evaluate.py'),
            ('Сборка этого отчёта', 'task_1/reproduction/build_editorial_report.py')]:
        p('<b>' + escape(title) + ':</b> ' + escape(path), 'small')

    output.parent.mkdir(parents=True, exist_ok=True)
    def background(canvas, doc):
        canvas.saveState()
        canvas.setFillColor(colors.HexColor(PAPER))
        canvas.rect(0, 0, *A4, fill=1, stroke=0)
        canvas.restoreState()

    def footer(canvas, doc):
        if state['section'] < 0:
            return
        canvas.saveState()
        section = state['section']
        rubrics = ['ЗАДАЧА', 'ДАННЫЕ', 'ДИАГНОСТИКА', 'ВРЕМЯ', 'ПРИЗНАКИ',
                   'ПОИСК', 'ИНТЕРПРЕТАЦИЯ', 'ЭКСПЕРИМЕНТ', 'РЕЗУЛЬТАТЫ',
                   'СТОИМОСТЬ', 'ВОСПРОИЗВЕДЕНИЕ', 'ВЫВОДЫ', 'ИСТОЧНИКИ']
        canvas.setFillColor(colors.HexColor('#69756F'))
        canvas.setFont('Report', 7.2)
        canvas.drawString(48, 808, 'RECSYS / FEATURE ENGINEERING')
        canvas.drawRightString(547, 808, 'YAMBDA FLAT 50M')
        canvas.setStrokeColor(colors.HexColor('#C9CEC3'))
        canvas.line(48, 794, 547, 794)
        canvas.setFillColor(colors.HexColor(ORANGE))
        canvas.setFont('ReportMono', 8)
        canvas.drawString(48, 774, f'{section + 1:02d}')
        canvas.setFillColor(colors.HexColor('#69756F'))
        canvas.setFont('Report', 7.6)
        canvas.drawString(73, 774, rubrics[section])
        # Тонкая шкала разделов связывает отдельные страницы в один отчёт.
        segment = 499 / len(rubrics)
        for i in range(len(rubrics)):
            canvas.setStrokeColor(colors.HexColor(INK if i == section else '#CDD2C7'))
            canvas.setLineWidth(1.6 if i == section else .6)
            canvas.line(48 + i * segment, 42, 48 + (i + 1) * segment - 3, 42)
        canvas.setFont('Report', 7)
        canvas.setFillColor(colors.HexColor('#69756F'))
        canvas.drawString(48, 25, 'АВТОМАТИЧЕСКИЕ ПРИЗНАКИ / ИССЛЕДОВАНИЕ')
        canvas.setFont('ReportMono', 10)
        canvas.drawRightString(547, 24, f'{doc.page - 1:02d}')
        canvas.restoreState()

    class EditorialDocument(SimpleDocTemplate):
        def afterPage(self):
            footer(self.canv, self)

    document = EditorialDocument(str(output), pagesize=A4, rightMargin=48, leftMargin=48,
                                 topMargin=84, bottomMargin=59, title='Автоматическое создание признаков для RecSys',
                                 author='GodNaoborot', subject='Yambda flat 50m: ручные48 и DFS462',
                                 pageCompression=1)
    document.build(story, onFirstPage=background, onLaterPages=background)
    print(json.dumps({'output': str(output), 'bytes': output.stat().st_size,
                      'sha256': hashlib.sha256(output.read_bytes()).hexdigest(),
                      'expected_sections': pagenames}, ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--figures-only', action='store_true')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    if args.figures_only:
        figures(root)
    else:
        report(root, args.output or root / 'output/pdf/recsys_feature_generation_editorial.pdf')


if __name__ == '__main__':
    main()
