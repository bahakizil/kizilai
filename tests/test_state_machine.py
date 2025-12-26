"""Tests for call state machine."""

import pytest

from src.core.state_machine import CallState, CallStateMachine


class TestCallStateMachine:
    """Tests for CallStateMachine class."""

    def test_initial_state(self):
        """Should start in IDLE state."""
        sm = CallStateMachine("test-call")
        assert sm.state == CallState.IDLE

    def test_start_call(self):
        """Should transition from IDLE to LISTENING."""
        sm = CallStateMachine("test-call")
        assert sm.start_call() is True
        assert sm.state == CallState.LISTENING

    def test_invalid_transition(self):
        """Should reject invalid state transitions."""
        sm = CallStateMachine("test-call")
        # Can't go from IDLE to PROCESSING directly
        assert sm.transition(CallState.PROCESSING) is False
        assert sm.state == CallState.IDLE

    def test_full_conversation_flow(self):
        """Should support full conversation flow."""
        sm = CallStateMachine("test-call")

        # Start call
        assert sm.start_call() is True
        assert sm.state == CallState.LISTENING

        # User stops speaking -> start processing
        assert sm.user_stopped_speaking() is True
        assert sm.state == CallState.PROCESSING

        # AI starts speaking
        assert sm.ai_started_speaking() is True
        assert sm.state == CallState.SPEAKING

        # AI finishes speaking
        assert sm.ai_finished_speaking() is True
        assert sm.state == CallState.LISTENING

        # End call
        assert sm.end_call() is True
        assert sm.state == CallState.IDLE

    def test_barge_in(self):
        """Should handle user interruption (barge-in)."""
        sm = CallStateMachine("test-call")

        sm.start_call()
        sm.user_stopped_speaking()
        sm.ai_started_speaking()

        assert sm.state == CallState.SPEAKING
        assert sm.can_interrupt() is True

        # User starts speaking while AI is speaking
        assert sm.user_started_speaking() is True
        assert sm.state == CallState.INTERRUPTED

        # Then back to listening
        assert sm.transition(CallState.LISTENING) is True

    def test_is_active(self):
        """Should report active state correctly."""
        sm = CallStateMachine("test-call")
        assert sm.is_active() is False

        sm.start_call()
        assert sm.is_active() is True

        sm.end_call()
        assert sm.is_active() is False

    def test_listener_notification(self):
        """Should notify listeners on state change."""
        sm = CallStateMachine("test-call")
        transitions = []

        def listener(old_state, new_state):
            transitions.append((old_state, new_state))

        sm.add_listener(listener)
        sm.start_call()

        assert len(transitions) == 1
        assert transitions[0] == (CallState.IDLE, CallState.LISTENING)

    def test_remove_listener(self):
        """Should allow removing listeners."""
        sm = CallStateMachine("test-call")
        transitions = []

        def listener(old_state, new_state):
            transitions.append((old_state, new_state))

        sm.add_listener(listener)
        sm.remove_listener(listener)
        sm.start_call()

        assert len(transitions) == 0
