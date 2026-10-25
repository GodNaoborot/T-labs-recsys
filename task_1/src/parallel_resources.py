"""Допуск независимых CatBoost CPU-fit по памяти и числу потоков.

Модуль не запускает процессы и не читает результаты: контроллер передает
завершенные профили, текущий RSS и системные счетчики в явном виде.
"""
from collections import deque
from math import isfinite
from numbers import Integral, Real


THREADS = {'ranker': 4, 'pointwise': 8}
BUDGET_OVERHEAD_GIB = 0.35
SWAP_WINDOW_SECONDS = 10.0
SWAP_THRESHOLD_BYTES = 256 * 1024 ** 2


def _finite_number(value, name):
    """Проверить конечное неотрицательное число без логических значений."""
    if isinstance(value, bool) or not isinstance(value, Real) or not isfinite(value) or value < 0:
        raise ValueError(f'{name}: нужно конечное неотрицательное число')
    return float(value)


def _positive_count(value, name):
    """Проверить положительное целое число строк или доступных слотов."""
    if isinstance(value, bool) or not isinstance(value, Integral) or value <= 0:
        raise ValueError(f'{name}: нужно положительное целое число')
    return int(value)


def _job_key(job):
    """Проверить описание полного fit и вернуть тип признаков и objective."""
    required = {'window', 'kind', 'objective', 'train_rows', 'val_rows'}
    if not isinstance(job, dict) or not required.issubset(job):
        raise ValueError(f'job должен содержать поля {sorted(required)}')
    if job['kind'] not in {'manual', 'dfs'} or job['objective'] not in THREADS:
        raise ValueError('Неизвестные kind или objective')
    for name in ['train_rows', 'val_rows']:
        _positive_count(job[name], name)
    return job['kind'], job['objective']


def _scaled_peak(job, profile):
    """Перенести наблюденный пик по большему отношению размеров train и val."""
    peak = _finite_number(profile['peak_rss_gib'], 'peak_rss_gib')
    if peak <= 0:
        raise ValueError('peak_rss_gib должен быть положительным')
    scale = max(job['train_rows'] / profile['train_rows'],
                job['val_rows'] / profile['val_rows'])
    return peak * scale


def peak_budget_gib(job, profiles) -> float | None:
    """Вернуть наибольший перенесенный пик + 0.35 GiB; без профиля вернуть None."""
    key = _job_key(job)
    estimates = [_scaled_peak(job, profile) for profile in profiles if _job_key(profile) == key]
    return max(estimates) + BUDGET_OVERHEAD_GIB if estimates else None


def _threads(job):
    """Вернуть зафиксированные 4 потока ranker или 8 потоков pointwise."""
    return THREADS[_job_key(job)[1]]


def can_launch(job, active, profiles, available_gib, cpu_limit=12,
               reserve_gib=1.5, max_active=2) -> bool:
    """Допустить новый fit, оставив запас на рост RSS уже активных процессов."""
    available = _finite_number(available_gib, 'available_gib')
    reserve = _finite_number(reserve_gib, 'reserve_gib')
    cpu_limit = _positive_count(cpu_limit, 'cpu_limit')
    max_active = _positive_count(max_active, 'max_active')
    if len(active) >= max_active:
        return False
    budget = peak_budget_gib(job, profiles)
    budgets = [peak_budget_gib(running, profiles) for running in active]
    if budget is None or any(value is None for value in budgets):
        return False
    if _threads(job) + sum(_threads(running) for running in active) > cpu_limit:
        return False
    growth = sum(max(0.0, value - _finite_number(running['rss_gib'], 'rss_gib'))
                 for running, value in zip(active, budgets))
    return available >= budget + growth + reserve


class MemoryPressureGuard:
    """Наблюдать свежую нехватку памяти; абсолютный swap-used не учитывается."""
    def __init__(self):
        """Начать наблюдение без предыдущих счетчиков и низких значений памяти."""
        self.reset()

    def reset(self):
        """Очистить историю после fallback, чтобы старый сигнал не повторялся."""
        self._samples = deque()
        self._low_samples = 0
        self.reason = None

    def _record_swap(self, swap_out_bytes, now):
        """Сохранить cumulative swap-out и оставить наблюдения последних 10 секунд."""
        if isinstance(swap_out_bytes, bool) or not isinstance(swap_out_bytes, Integral) or swap_out_bytes < 0:
            raise ValueError('swap_out_bytes: нужен неотрицательный целый счетчик')
        if self._samples and now < self._samples[-1][0]:
            raise ValueError('now должен монотонно возрастать')
        if self._samples and swap_out_bytes < self._samples[-1][1]:
            self._samples.clear()
        self._samples.append((now, int(swap_out_bytes)))
        while self._samples[0][0] < now - SWAP_WINDOW_SECONDS:
            self._samples.popleft()
        return int(swap_out_bytes) - self._samples[0][1]

    def observe(self, available_gib, swap_out_bytes, now) -> bool:
        """Сигнализировать о двух low-memory сэмплах либо свежем swap-out при RAM < 2."""
        available = _finite_number(available_gib, 'available_gib')
        now = _finite_number(now, 'now')
        recent_swap = self._record_swap(swap_out_bytes, now)
        self._low_samples = self._low_samples + 1 if available < 1.0 else 0
        low_memory = self._low_samples >= 2
        swapping = recent_swap > SWAP_THRESHOLD_BYTES and available < 2.0
        self.reason = 'low_available' if low_memory else 'recent_swap_out' if swapping else None
        return low_memory or swapping
