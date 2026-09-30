"""Keep a laptop training run inside a thermal budget.

⚠️ WRITTEN BY AI -- 2026-09-24. UNVERIFIED against a long run.

⚠️ There is NO API to cap GPU utilisation on Apple Silicon. Nothing in PyTorch,
Metal or macOS lets a process ask for "50% of the GPU". What actually controls
temperature is:

    1. DUTY CYCLE -- compute for a while, then idle. This is the real lever,
       and it is the only one that bounds sustained heat regardless of what
       the model is doing.
    2. THREAD COUNT -- fewer CPU threads, fewer cores kept hot.
    3. BATCH SIZE -- smaller batches finish sooner and leave gaps.

This module implements (1) and reads macOS's own throttling signal so the run
backs off further once the system is already unhappy.

⚠️ A duty cycle of 0.5 does not cost 2x wall-clock and nothing else -- it IS
2x wall-clock. Thermal safety on a laptop is paid for in time, and the only
question is whether you would rather pay it in pauses or in the OS throttling
you unpredictably halfway through an epoch.
"""

from __future__ import annotations

import re
import subprocess
import time

from lightning.pytorch.callbacks import Callback

#macOS records a throttling event here; the value is a percentage of full speed
SPEED_LIMIT_PATTERN = re.compile(r"CPU_Speed_Limit\s*=\s*(\d+)")

#how far below 100 the speed limit must fall before the duty cycle is cut
THROTTLE_TRIGGER = 100


def read_cpu_speed_limit():
    """Read macOS's current CPU speed limit, as a percentage.

    `pmset -g therm` reports nothing at all until the system has recorded a
    thermal event, so "no reading" is the healthy case, not an error.

    RETURNS:
        int or None: the percentage, or None if the OS has recorded nothing
    """
    try:
        out = subprocess.run(
            ["pmset", "-g", "therm"], capture_output=True, text=True, timeout=5
        )
    except Exception:
        #not macOS, or pmset unavailable -- the caller treats this as no signal
        return None

    match = SPEED_LIMIT_PATTERN.search(out.stdout)
    return int(match.group(1)) if match else None


class ThermalThrottle(Callback):
    """Pace training to a target duty cycle, and back off when macOS throttles.

    After every batch the callback sleeps for however long it takes to hold the
    requested ratio of compute to wall-clock. At `duty_cycle=0.5` a 0.4 s step
    is followed by a 0.4 s pause.

    ⚠️ The pause happens between BATCHES, so the granularity is one step. A
    very large batch holds the GPU at full load for its whole duration no
    matter what this is set to -- if steps are long, lower the batch size too.

    ATTRS:
        duty_cycle (float): target fraction of wall-clock spent computing
        cooldown_every_seconds (float): how often to take a longer pause
        cooldown_seconds (float): how long that pause lasts
        effective_duty (float): what was actually achieved, measured
    """

    def __init__(
        self,
        duty_cycle=0.6,
        cooldown_every_minutes=0.0,
        cooldown_seconds=0.0,
        poll_seconds=60.0,
    ):
        """
        ARGS:
            duty_cycle (float): 1.0 disables pacing; 0.5 idles half the time
            cooldown_every_minutes (float): 0 disables the periodic long pause
            cooldown_seconds (float): length of that pause
            poll_seconds (float): how often to re-read the OS throttling signal
        """
        super().__init__()

        #a duty of 0 would sleep forever; above 1 is not a thing
        if not 0 < duty_cycle <= 1.0:
            raise ValueError(f"duty_cycle must be in (0, 1], got {duty_cycle}")

        self.duty_cycle = duty_cycle
        self.cooldown_every_seconds = cooldown_every_minutes * 60.0
        self.cooldown_seconds = cooldown_seconds
        self.poll_seconds = poll_seconds

        self._step_started = None
        self._compute_seconds = 0.0
        self._paused_seconds = 0.0
        self._run_started = None
        self._last_cooldown = None
        self._last_poll = 0.0

        #halved whenever macOS reports it is already throttling us
        self._backoff = 1.0
        self.effective_duty = None

    def on_train_start(self, trainer, pl_module):
        """Announce the budget, so a slow run is never a surprise."""
        self._run_started = time.perf_counter()
        self._last_cooldown = self._run_started

        if self.duty_cycle >= 1.0:
            print("[thermal] duty cycle 1.0 -- no pacing, the machine runs flat out")
        else:
            #say the cost in the same breath as the setting
            print(
                f"[thermal] duty cycle {self.duty_cycle:.2f} -- training will take "
                f"~{1 / self.duty_cycle:.1f}x longer than unthrottled"
            )
        if self.cooldown_every_seconds > 0:
            print(
                f"[thermal] plus a {self.cooldown_seconds:.0f}s pause every "
                f"{self.cooldown_every_seconds / 60:.0f} min"
            )

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        self._step_started = time.perf_counter()

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        """Sleep for whatever holds the duty cycle, then handle cooldowns."""
        if self._step_started is None:
            return

        elapsed = time.perf_counter() - self._step_started
        self._compute_seconds += elapsed

        #re-read the OS signal occasionally; the call is cheap but not free
        now = time.perf_counter()
        if now - self._last_poll > self.poll_seconds:
            self._last_poll = now
            self._update_backoff()

        #compute:idle of d:(1-d) means idling elapsed * (1/d - 1)
        target = self.duty_cycle * self._backoff
        if target < 1.0:
            pause = elapsed * (1.0 / target - 1.0)
            time.sleep(pause)
            self._paused_seconds += pause

        #a longer pause lets the chassis actually shed heat, which a string of
        #short gaps between batches does not
        if self.cooldown_every_seconds > 0:
            if time.perf_counter() - self._last_cooldown > self.cooldown_every_seconds:
                print(f"[thermal] cooldown pause {self.cooldown_seconds:.0f}s")
                time.sleep(self.cooldown_seconds)
                self._paused_seconds += self.cooldown_seconds
                self._last_cooldown = time.perf_counter()

    def _update_backoff(self):
        """Halve the duty cycle while macOS reports it is throttling us.

        If the OS is already limiting clock speed, the thermal budget has
        been exceeded and the configured duty cycle was too generous.
        """
        limit = read_cpu_speed_limit()

        #None means macOS has recorded no thermal event, which is the good case
        if limit is None or limit >= THROTTLE_TRIGGER:
            if self._backoff < 1.0:
                print("[thermal] pressure cleared, restoring the configured duty cycle")
            self._backoff = 1.0
            return

        if self._backoff == 1.0:
            print(
                f"[thermal] macOS reports CPU_Speed_Limit={limit}% -- halving the "
                f"duty cycle until it clears"
            )
        self._backoff = 0.5

    def on_train_epoch_end(self, trainer, pl_module):
        """Report what the pacing actually achieved, not what was asked for."""
        total = self._compute_seconds + self._paused_seconds
        if total <= 0:
            return

        #the measured number is the one worth trusting; sleep overshoots
        self.effective_duty = self._compute_seconds / total
        print(
            f"[thermal] epoch: {self._compute_seconds / 60:.1f} min computing, "
            f"{self._paused_seconds / 60:.1f} min paused "
            f"(effective duty {self.effective_duty:.2f})"
        )
        self._compute_seconds = 0.0
        self._paused_seconds = 0.0
