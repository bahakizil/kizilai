"""Call state machine for voice agent."""

from enum import Enum
from typing import Callable

from src.utils.logging import get_logger

logger = get_logger(__name__)


class CallState(Enum):
    """States of a voice call."""

    IDLE = "idle"  # Call not active
    LISTENING = "listening"  # Listening to user speech
    PROCESSING = "processing"  # Processing user input (STT -> LLM)
    SPEAKING = "speaking"  # AI is speaking (TTS output)
    INTERRUPTED = "interrupted"  # User interrupted AI speech


# Valid state transitions
VALID_TRANSITIONS: dict[CallState, set[CallState]] = {
    CallState.IDLE: {CallState.LISTENING},
    CallState.LISTENING: {CallState.PROCESSING, CallState.IDLE},
    CallState.PROCESSING: {CallState.SPEAKING, CallState.LISTENING, CallState.IDLE},
    CallState.SPEAKING: {CallState.INTERRUPTED, CallState.LISTENING, CallState.IDLE},
    CallState.INTERRUPTED: {CallState.LISTENING, CallState.IDLE},
}


class CallStateMachine:
    """Manages state transitions for a voice call.

    Ensures valid state transitions and notifies listeners
    when state changes occur.
    """

    def __init__(self, call_id: str):
        """Initialize state machine.

        Args:
            call_id: Unique identifier for the call
        """
        self.call_id = call_id
        self._state = CallState.IDLE
        self._listeners: list[Callable[[CallState, CallState], None]] = []

        logger.info("state_machine_initialized", call_id=call_id, state=self._state.value)

    @property
    def state(self) -> CallState:
        """Current state of the call."""
        return self._state

    def transition(self, new_state: CallState) -> bool:
        """Attempt to transition to a new state.

        Args:
            new_state: Target state

        Returns:
            True if transition was successful, False otherwise
        """
        if new_state not in VALID_TRANSITIONS.get(self._state, set()):
            logger.warning(
                "invalid_state_transition",
                call_id=self.call_id,
                from_state=self._state.value,
                to_state=new_state.value,
            )
            return False

        old_state = self._state
        self._state = new_state

        logger.info(
            "state_transition",
            call_id=self.call_id,
            from_state=old_state.value,
            to_state=new_state.value,
        )

        # Notify listeners
        for listener in self._listeners:
            try:
                listener(old_state, new_state)
            except Exception as e:
                logger.error("state_listener_error", error=str(e))

        return True

    def add_listener(self, listener: Callable[[CallState, CallState], None]) -> None:
        """Add a state change listener.

        Args:
            listener: Callback function(old_state, new_state)
        """
        self._listeners.append(listener)

    def remove_listener(self, listener: Callable[[CallState, CallState], None]) -> None:
        """Remove a state change listener."""
        if listener in self._listeners:
            self._listeners.remove(listener)

    def is_active(self) -> bool:
        """Check if call is in an active state (not IDLE)."""
        return self._state != CallState.IDLE

    def can_interrupt(self) -> bool:
        """Check if AI speech can be interrupted."""
        return self._state == CallState.SPEAKING

    def start_call(self) -> bool:
        """Start the call (transition to LISTENING)."""
        return self.transition(CallState.LISTENING)

    def end_call(self) -> bool:
        """End the call (transition to IDLE)."""
        return self.transition(CallState.IDLE)

    def user_started_speaking(self) -> bool:
        """Handle user starting to speak.

        If AI is speaking, triggers interruption.
        """
        if self._state == CallState.SPEAKING:
            return self.transition(CallState.INTERRUPTED)
        return True

    def user_stopped_speaking(self) -> bool:
        """Handle user finished speaking (start processing)."""
        if self._state in (CallState.LISTENING, CallState.INTERRUPTED):
            return self.transition(CallState.PROCESSING)
        return False

    def ai_started_speaking(self) -> bool:
        """Handle AI starting to speak."""
        if self._state == CallState.PROCESSING:
            return self.transition(CallState.SPEAKING)
        return False

    def ai_finished_speaking(self) -> bool:
        """Handle AI finished speaking."""
        if self._state == CallState.SPEAKING:
            return self.transition(CallState.LISTENING)
        return False
