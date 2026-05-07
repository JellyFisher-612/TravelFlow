from __future__ import annotations

import unittest
from unittest.mock import patch

from utils import circuit_breaker
from utils.circuit_breaker import CircuitBreaker, CircuitOpenError, CircuitState


class CircuitBreakerTests(unittest.TestCase):
    def test_circuit_starts_closed(self):
        breaker = CircuitBreaker()

        self.assertEqual(CircuitState.CLOSED, breaker.state)
        self.assertTrue(breaker.allow_call())
        self.assertEqual("closed", breaker.get_status()["state"])

    def test_circuit_opens_after_failure_threshold_consecutive_failures(self):
        breaker = CircuitBreaker(failure_threshold=3)

        with patch.object(circuit_breaker.time, "monotonic", side_effect=[1.0, 2.0, 3.0, 4.0]):
            breaker.record_failure()
            breaker.record_failure()
            self.assertEqual(CircuitState.CLOSED, breaker.state)

            breaker.record_failure()

        with patch.object(circuit_breaker.time, "monotonic", return_value=4.0):
            self.assertEqual(CircuitState.OPEN, breaker.state)
            self.assertFalse(breaker.allow_call())
            with self.assertRaises(CircuitOpenError):
                breaker.raise_if_open()

    def test_circuit_transitions_to_half_open_after_recovery_timeout(self):
        breaker = CircuitBreaker(failure_threshold=1, recovery_timeout_sec=10.0)

        with patch.object(circuit_breaker.time, "monotonic", side_effect=[100.0, 100.0]):
            breaker.record_failure()

        with patch.object(circuit_breaker.time, "monotonic", return_value=109.9):
            self.assertEqual(CircuitState.OPEN, breaker.state)
            self.assertFalse(breaker.allow_call())

        with patch.object(circuit_breaker.time, "monotonic", return_value=110.0):
            self.assertEqual(CircuitState.HALF_OPEN, breaker.state)
            self.assertTrue(breaker.allow_call())

    def test_circuit_closes_after_half_open_successes(self):
        breaker = CircuitBreaker(
            failure_threshold=1,
            recovery_timeout_sec=10.0,
            half_open_successes=2,
        )

        with patch.object(circuit_breaker.time, "monotonic", side_effect=[100.0, 100.0]):
            breaker.record_failure()
        with patch.object(circuit_breaker.time, "monotonic", return_value=110.0):
            self.assertEqual(CircuitState.HALF_OPEN, breaker.state)

        breaker.record_success()
        self.assertEqual(CircuitState.HALF_OPEN, breaker.state)

        breaker.record_success()
        self.assertEqual(CircuitState.CLOSED, breaker.state)
        self.assertTrue(breaker.allow_call())
        self.assertEqual(0, breaker.get_status()["failure_count"])

    def test_circuit_reopens_on_failure_during_half_open(self):
        breaker = CircuitBreaker(failure_threshold=1, recovery_timeout_sec=10.0)

        with patch.object(circuit_breaker.time, "monotonic", side_effect=[100.0, 100.0]):
            breaker.record_failure()
        with patch.object(circuit_breaker.time, "monotonic", return_value=110.0):
            self.assertEqual(CircuitState.HALF_OPEN, breaker.state)

        with patch.object(circuit_breaker.time, "monotonic", side_effect=[111.0, 111.0]):
            breaker.record_failure()

        with patch.object(circuit_breaker.time, "monotonic", return_value=111.0):
            self.assertEqual(CircuitState.OPEN, breaker.state)
            self.assertFalse(breaker.allow_call())


if __name__ == "__main__":
    unittest.main()
