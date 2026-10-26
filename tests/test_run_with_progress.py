"""Tests for run_with_progress, with a fake Tk window instead of a display."""

import threading
import time
import unittest
from unittest.mock import MagicMock, patch

from forensics_app.tools import dialogs
from forensics_app.tools.dialogs import TaskCancelled, run_with_progress


class FakeWindow:
    """Stand-in for tk.Toplevel that runs ``after`` callbacks when the test pumps it."""

    def __init__(self, _parent=None) -> None:
        self.callbacks = []
        self.protocols = {}
        self.destroyed = False

    def after(self, _ms, callback) -> None:
        self.callbacks.append(callback)

    def protocol(self, name, callback) -> None:
        self.protocols[name] = callback

    def destroy(self) -> None:
        self.destroyed = True

    def title(self, *_args) -> None:
        pass

    def transient(self, *_args) -> None:
        pass

    def resizable(self, *_args) -> None:
        pass

    def grab_set(self) -> None:
        pass


class RunWithProgressWindowTests(unittest.TestCase):
    def setUp(self) -> None:
        self.windows = []

        def make_window(parent):
            window = FakeWindow(parent)
            self.windows.append(window)
            return window

        patch.object(dialogs.tk, "Toplevel", side_effect=make_window).start()
        self.ttk = patch.object(dialogs, "ttk").start()
        self.addCleanup(patch.stopall)
        self.parent = MagicMock()
        self.parent.wait_window.side_effect = self.pump
        self.on_open = None  # optional action run once the window exists

    def pump(self, window) -> None:
        """Play the Tk event loop: run scheduled callbacks until the window closes."""
        if self.on_open is not None:
            self.on_open(window)
        deadline = time.monotonic() + 5
        while not window.destroyed:
            self.assertLess(time.monotonic(), deadline, "the progress window never closed")
            pending, window.callbacks = window.callbacks, []
            for callback in pending:
                callback()
            time.sleep(0.005)

    def cancel_button_command(self):
        return self.ttk.Button.call_args.kwargs["command"]

    def test_returns_the_task_result_off_the_ui_thread(self) -> None:
        threads = []

        def task(report, cancelled):
            threads.append(threading.current_thread())
            report(1, 2, "Halfway")
            return "done"

        self.assertEqual(run_with_progress(self.parent, "Work", task), "done")
        self.assertIsNot(threads[0], threading.current_thread())
        self.assertTrue(self.windows[0].destroyed)

    def test_progress_moves_the_bar_and_label(self) -> None:
        release = threading.Event()

        def task(report, cancelled):
            report(3, 4, "Image 3 of 4")
            release.wait(2)
            return None

        def check_progress(window) -> None:
            # Let the worker report, then run the poll callback once before finishing
            deadline = time.monotonic() + 2
            while not self.ttk.Progressbar.return_value.configure.called and time.monotonic() < deadline:
                pending, window.callbacks = window.callbacks, []
                for callback in pending:
                    callback()
                time.sleep(0.005)
            release.set()

        self.on_open = check_progress
        run_with_progress(self.parent, "Work", task)
        self.ttk.Progressbar.return_value.configure.assert_any_call(maximum=4, value=3)
        self.ttk.Label.return_value.configure.assert_any_call(text="Image 3 of 4")

    def test_task_errors_are_raised_in_the_caller(self) -> None:
        def task(report, cancelled):
            raise ValueError("bad image")

        with self.assertRaisesRegex(ValueError, "bad image"):
            run_with_progress(self.parent, "Work", task)

    def test_cancel_button_stops_the_task(self) -> None:
        def task(report, cancelled):
            deadline = time.monotonic() + 2
            while not cancelled() and time.monotonic() < deadline:
                time.sleep(0.005)
            raise TaskCancelled()

        self.on_open = lambda _window: self.cancel_button_command()()
        with self.assertRaises(TaskCancelled):
            run_with_progress(self.parent, "Work", task)
        self.ttk.Button.return_value.configure.assert_called_with(state="disabled")

    def test_closing_the_window_cancels_even_if_the_task_finishes(self) -> None:
        self.on_open = lambda window: window.protocols["WM_DELETE_WINDOW"]()
        with self.assertRaises(TaskCancelled):
            run_with_progress(self.parent, "Work", lambda report, cancelled: "ignored")


if __name__ == "__main__":
    unittest.main()
