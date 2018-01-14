from .math_and_types import floor
from time import time


class ProgressMeter:
    def __init__(self, total: int, leader='', print_out=True):
        if total <= 0:
            raise ValueError(f"invalid total: {total}")
        self._total = total
        self._leader = leader
        self._print = print_out
        self._complete = 0
        self._initial_time = time()
        self._total_time = 0
        self._last_string = ''
        self._update_and_print()

    def _update_last_string(self):
        elapsed = time() - self._initial_time
        self._total_time = elapsed
        percent = round(self._complete / self._total * 100, 1)
        est_per = 0 if self._complete == 0 else self._total_time / self._complete
        est_remain = est_per * (self._total - self._complete)
        self._last_string = f"{self._leader}{self._complete} / {self._total} = {percent}% " \
                            f"| {round(self._total_time, 2)}s total " \
                            f"| ~{round(est_per, 2)}s/iter " \
                            f"| ~{round(est_remain, 2)}s remaining"

    def _update_and_print(self):
        print('\b' * len(self._last_string) + '\r', end='')
        self._update_last_string()
        print(self._last_string, end='')

    def update(self, completed=1):
        if not self.done:
            self._complete += completed
        if not self._print:
            return
        self._update_and_print()

    def finish(self):
        self._complete = self._total
        self.update()
        if self._print:
            print()

    def set_leader(self, new_leader: str):
        self._leader = new_leader

    @property
    def done(self):
        if self._complete >= self._total:
            return True
        else:
            return False
