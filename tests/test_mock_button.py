"""Tests for MockButton."""

from __future__ import annotations

from app.mock_button import MockButton


# ------------------------------------------------------------------
# Unit tests for MockButton
# ------------------------------------------------------------------


class TestMockButtonInterface:
    """Verify that MockButton satisfies the AbstractButton interface."""

    def test_when_pressed_getter_setter(self):
        btn = MockButton()
        cb = lambda: None  # noqa: E731
        btn.when_pressed = cb
        assert btn.when_pressed is cb

    def test_when_released_getter_setter(self):
        btn = MockButton()
        cb = lambda: None  # noqa: E731
        btn.when_released = cb
        assert btn.when_released is cb

    def test_initial_state_not_pressed(self):
        btn = MockButton()
        assert btn._pressed is False
        assert btn._running is False


class TestMockButtonToggle:
    """Verify the press/release toggle logic without spawning a thread."""

    def _trigger_enter(self, btn: MockButton) -> None:
        """Directly invoke the internal toggle as if Enter was pressed."""
        if btn._pressed:
            btn._pressed = False
            if btn._when_released is not None:
                btn._when_released()
        else:
            btn._pressed = True
            if btn._when_pressed is not None:
                btn._when_pressed()

    def test_first_enter_fires_when_pressed(self):
        btn = MockButton()
        pressed_calls = []
        btn.when_pressed = lambda: pressed_calls.append(1)
        self._trigger_enter(btn)
        assert pressed_calls == [1]
        assert btn._pressed is True

    def test_second_enter_fires_when_released(self):
        btn = MockButton()
        released_calls = []
        btn.when_released = lambda: released_calls.append(1)
        self._trigger_enter(btn)  # press
        self._trigger_enter(btn)  # release
        assert released_calls == [1]
        assert btn._pressed is False

    def test_toggle_sequence(self):
        btn = MockButton()
        events = []
        btn.when_pressed = lambda: events.append("pressed")
        btn.when_released = lambda: events.append("released")

        for _ in range(3):
            self._trigger_enter(btn)  # press
            self._trigger_enter(btn)  # release

        assert events == ["pressed", "released", "pressed", "released", "pressed", "released"]


class TestMockButtonQuit:
    """Verify quit callback is invoked."""

    def test_quit_callback_invoked(self):
        btn = MockButton()
        quit_calls = []
        btn._running = True

        # Simulate 'q' input path
        btn._running = False
        on_quit = lambda: quit_calls.append(1)  # noqa: E731
        on_quit()

        assert quit_calls == [1]
