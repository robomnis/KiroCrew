"""Command-authority boundary tests for coordinator-backed subagent mutations."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any

import pytest

from kiro_crew.run_coordinator import (
    CommandOperation,
    CommandStatus,
    MemoryRunCoordinator,
    SubmitRun,
)
from kiro_crew.subagent_command_authority import (
    AuthorityConflict,
    AuthorityOutcomeUncertain,
    AuthorityUnavailable,
    CommandIdentity,
    SubagentCommandAuthority,
)


class _FinishUnavailableCoordinator(MemoryRunCoordinator):
    async def finish_command(self, *args: Any, **kwargs: Any):
        raise OSError("coordinator write failed")


@dataclass
class _Info:
    id: str
    done: bool = False
    error: str = ""
    queued: bool = False
    _coordinator_waiting: bool = False


class _Manager:
    def __init__(self, *, register_spawn: bool = True) -> None:
        self.register_spawn = register_spawn
        self.spawn_calls: list[tuple[str, dict[str, Any]]] = []
        self.continue_calls: list[tuple[str, str, dict[str, Any]]] = []
        self.steer_calls: list[tuple[str, str]] = []
        self.followup_calls: list[tuple[str, str]] = []
        self.cancel_calls: list[str] = []
        self.release_calls: list[str] = []
        self.infos: dict[str, _Info] = {}

    def spawn(self, task: str, **kwargs: Any) -> _Info:
        self.spawn_calls.append((task, kwargs))
        info = _Info(
            kwargs["_preassigned_id"],
            queued=not self.register_spawn,
            _coordinator_waiting=not self.register_spawn,
        )
        if self.register_spawn:
            self.infos[info.id] = info
        return info

    def continue_conversation(self, conversation_id: str, task: str, **kwargs: Any) -> _Info:
        self.continue_calls.append((conversation_id, task, kwargs))
        info = _Info(kwargs["_preassigned_id"])
        self.infos[info.id] = info
        return info

    def get(self, run_id: str) -> _Info | None:
        return self.infos.get(run_id)

    async def steer_run(self, run_id: str, message: str) -> tuple[bool, str]:
        self.steer_calls.append((run_id, message))
        return True, "ok"

    async def follow_up_run(self, run_id: str, message: str) -> tuple[bool, str]:
        self.followup_calls.append((run_id, message))
        return True, "queued"

    async def cancel(self, run_id: str) -> bool:
        self.cancel_calls.append(run_id)
        return True

    def release_conversation(self, conversation_id: str) -> tuple[bool, str]:
        self.release_calls.append(conversation_id)
        return True, "released"


class _UnclaimableCoordinator(MemoryRunCoordinator):
    async def claim_command(self, command_id, owner):
        return None


class _FirstClaimUnavailableCoordinator(MemoryRunCoordinator):
    def __init__(self) -> None:
        super().__init__()
        self.claim_attempts = 0

    async def claim_command(self, command_id, owner):
        self.claim_attempts += 1
        if self.claim_attempts == 1:
            return None
        return await super().claim_command(command_id, owner)


class _RejectingManager(_Manager):
    def spawn(self, task: str, **kwargs: Any) -> _Info:
        self.spawn_calls.append((task, kwargs))
        return _Info(
            kwargs["_preassigned_id"],
            done=True,
            error="spawn refused by governance",
        )

    def continue_conversation(self, conversation_id: str, task: str, **kwargs: Any) -> _Info:
        self.continue_calls.append((conversation_id, task, kwargs))
        return _Info(
            kwargs["_preassigned_id"],
            done=True,
            error="conversation_busy: existing run",
        )


class _SlowCancelManager(_Manager):
    def __init__(self, clock: list[float]) -> None:
        super().__init__()
        self._clock = clock

    async def cancel(self, run_id: str) -> bool:
        self._clock[0] += 31.0
        return await super().cancel(run_id)


def _identity(suffix: str, *, key: str | None = None) -> CommandIdentity:
    return CommandIdentity(
        run_id=f"run-{suffix}",
        command_id=f"command-{suffix}",
        idempotency_key=key or f"key-{suffix}",
    )


async def _coordinator_with_target(
    run_id: str,
    *,
    clock: Any = None,
) -> MemoryRunCoordinator:
    coordinator = MemoryRunCoordinator(clock=clock) if clock is not None else MemoryRunCoordinator()
    result = await coordinator.submit(
        SubmitRun(
            run_id=run_id,
            command_id=f"seed:{run_id}",
            idempotency_key=f"seed:{run_id}",
            payload_hash="seed",
            payload_json="{}",
            parent_session="",
            agent="",
            task="seed",
            conversation_key="",
            operation=CommandOperation.SPAWN,
        )
    )
    assert result.value is not None
    return coordinator


@pytest.mark.asyncio
async def test_keyed_spawn_replay_invokes_sync_manager_once() -> None:
    manager = _Manager()
    authority = SubagentCommandAuthority(MemoryRunCoordinator(), manager)
    identity = _identity("spawn")

    first = await authority.spawn(
        identity,
        "inspect the tree",
        parent_session_key="dashboard:one",
        agent="reviewer",
    )
    replay = await authority.spawn(
        identity,
        "inspect the tree",
        parent_session_key="dashboard:one",
        agent="reviewer",
    )

    assert replay is first
    assert len(manager.spawn_calls) == 1
    called_task, called_kwargs = manager.spawn_calls[0]
    assert called_task == "inspect the tree"
    assert called_kwargs["parent_session_key"] == "dashboard:one"
    assert called_kwargs["agent"] == "reviewer"
    assert called_kwargs["_preassigned_id"] == "run-spawn"
    assert called_kwargs["_coordinator_admitted"] is True
    assert called_kwargs["_coordinator_command"].command_id == "command-spawn"
    assert called_kwargs["_coordinator_fence"].run_id == "run-spawn"
    assert called_kwargs["_coordinator_version"] == 1


@pytest.mark.asyncio
async def test_keyed_spawn_payload_conflict_fails_before_second_execution() -> None:
    manager = _Manager()
    authority = SubagentCommandAuthority(MemoryRunCoordinator(), manager)
    identity = _identity("spawn-conflict")
    await authority.spawn(identity, "first payload")

    with pytest.raises(AuthorityConflict, match="idempotency_conflict"):
        await authority.spawn(identity, "different payload")

    assert len(manager.spawn_calls) == 1


@pytest.mark.asyncio
async def test_keyed_queued_spawn_remains_claimed_until_manager_starts_it() -> None:
    manager = _Manager(register_spawn=False)
    coordinator = MemoryRunCoordinator()
    authority = SubagentCommandAuthority(coordinator, manager)
    identity = _identity("queued")

    first = await authority.spawn(identity, "wait for capacity")
    receipt = await coordinator.get_command_by_key(identity.idempotency_key)
    assert receipt is not None
    assert receipt.command.status is CommandStatus.CLAIMED

    replay_authority = SubagentCommandAuthority(coordinator, manager)
    with pytest.raises(AuthorityUnavailable, match="outcome is still pending"):
        await replay_authority.spawn(identity, "wait for capacity")

    await authority.execution_started(identity.run_id)
    replay = await replay_authority.spawn(identity, "wait for capacity")

    assert first.id == replay.id == identity.run_id
    assert len(manager.spawn_calls) == 1
    await authority.close()
    await replay_authority.close()


@pytest.mark.asyncio
async def test_rejected_execution_exact_retry_replays_stored_result() -> None:
    manager = _RejectingManager()
    coordinator = MemoryRunCoordinator()
    identity = _identity("rejected-replay")

    first = await SubagentCommandAuthority(coordinator, manager).spawn(
        identity, "reject this spawn"
    )
    replay = await SubagentCommandAuthority(coordinator, manager).spawn(
        identity, "reject this spawn"
    )

    assert replay.id == first.id
    assert replay.task == "reject this spawn"
    assert replay.done is True
    assert replay.error == "spawn refused by governance"
    assert len(manager.spawn_calls) == 1


@pytest.mark.asyncio
async def test_keyed_queued_spawn_renews_lease_before_manager_registration() -> None:
    clock = [100.0]
    heartbeat_ticks: asyncio.Queue[None] = asyncio.Queue()

    async def controlled_sleep(_delay: float) -> None:
        await heartbeat_ticks.get()

    manager = _Manager(register_spawn=False)
    coordinator = MemoryRunCoordinator(clock=lambda: clock[0])
    authority = SubagentCommandAuthority(
        coordinator,
        manager,
        clock=lambda: clock[0],
        sleep=controlled_sleep,
    )
    identity = _identity("queued-heartbeat")

    await authority.spawn(identity, "wait for capacity")
    clock[0] = 180.0
    heartbeat_ticks.put_nowait(None)
    for _ in range(10):
        await asyncio.sleep(0)
        run = await coordinator.get_run(identity.run_id)
        if run is not None and run.lease_expires_at > 180.0:
            break
    run = await coordinator.get_run(identity.run_id)
    assert run is not None
    assert run.lease_expires_at > 200.0
    await authority.close()


@pytest.mark.asyncio
async def test_finished_registered_spawn_does_not_remain_in_replay_cache() -> None:
    authority = SubagentCommandAuthority(MemoryRunCoordinator(), _Manager())
    identity = _identity("cache-eviction")

    await authority.spawn(identity, "complete admission")

    assert identity.run_id not in authority._execution_results


@pytest.mark.asyncio
async def test_post_effect_finish_failure_is_reported_as_transport_uncertainty() -> None:
    manager = _Manager()
    authority = SubagentCommandAuthority(_FinishUnavailableCoordinator(), manager)
    identity = _identity("finish-unavailable")

    with pytest.raises(AuthorityOutcomeUncertain, match="not durably finished"):
        await authority.spawn(identity, "child has started")

    assert [task for task, _kwargs in manager.spawn_calls] == ["child has started"]


@pytest.mark.asyncio
async def test_lookup_response_returns_none_for_unknown_key() -> None:
    authority = SubagentCommandAuthority(MemoryRunCoordinator(), _Manager())

    assert await authority.lookup_response("unknown") is None


@pytest.mark.asyncio
async def test_lookup_response_reconstructs_spawn_and_continuation() -> None:
    manager = _Manager()
    coordinator = MemoryRunCoordinator()
    authority = SubagentCommandAuthority(coordinator, manager)
    spawn_identity = _identity("lookup-spawn")
    continue_identity = _identity("lookup-continue")

    await authority.spawn(spawn_identity, "inspect", keep=True)
    await authority.continue_conversation(continue_identity, "conversation-one", "follow up")

    assert await authority.lookup_response(spawn_identity.idempotency_key) == {
        "found": True,
        "id": spawn_identity.run_id,
        "task": "inspect",
        "status": "spawned",
        "conversation": spawn_identity.run_id,
    }
    assert await authority.lookup_response(continue_identity.idempotency_key) == {
        "found": True,
        "id": continue_identity.run_id,
        "conversation": "conversation-one",
        "status": "spawned",
    }


@pytest.mark.asyncio
async def test_lookup_response_reports_pending_without_invoking_manager() -> None:
    manager = _Manager()
    coordinator = _UnclaimableCoordinator()
    authority = SubagentCommandAuthority(coordinator, manager)
    spawn_identity = _identity("pending-spawn")
    steer_identity = _identity("pending-steer")

    with pytest.raises(AuthorityUnavailable):
        await authority.spawn(spawn_identity, "wait durably")
    with pytest.raises(AuthorityUnavailable):
        await authority.steer(steer_identity, "legacy-target", "adjust")

    assert await authority.lookup_response(spawn_identity.idempotency_key) == {
        "found": True,
        "id": spawn_identity.run_id,
        "error": "command outcome is still pending",
        "status": "pending",
        "code": "command_pending",
    }
    assert await authority.lookup_response(steer_identity.idempotency_key) == {
        "found": True,
        "id": "legacy-target",
        "status": "pending",
        "code": "command_pending",
    }
    assert manager.spawn_calls == []
    assert manager.steer_calls == []


@pytest.mark.asyncio
async def test_exact_replay_of_unstarted_spawn_remains_pending() -> None:
    manager = _Manager()
    coordinator = _UnclaimableCoordinator()
    identity = _identity("pending-replay")

    with pytest.raises(AuthorityUnavailable, match="still pending"):
        await SubagentCommandAuthority(coordinator, manager).spawn(
            identity,
            "wait durably",
        )
    with pytest.raises(AuthorityUnavailable, match="still pending"):
        await SubagentCommandAuthority(coordinator, manager).spawn(
            identity,
            "wait durably",
        )
    assert manager.spawn_calls == []


@pytest.mark.asyncio
async def test_exact_replay_reclaims_never_claimed_spawn() -> None:
    manager = _Manager()
    coordinator = _FirstClaimUnavailableCoordinator()
    identity = _identity("pending-reclaim")

    with pytest.raises(AuthorityUnavailable, match="still pending"):
        await SubagentCommandAuthority(coordinator, manager).spawn(identity, "wait durably")

    replay = await SubagentCommandAuthority(coordinator, manager).spawn(
        identity,
        "wait durably",
    )

    assert replay.id == identity.run_id
    assert [task for task, _kwargs in manager.spawn_calls] == ["wait durably"]


@pytest.mark.asyncio
async def test_lookup_response_reconstructs_rejected_spawn() -> None:
    coordinator = MemoryRunCoordinator()
    authority = SubagentCommandAuthority(coordinator, _RejectingManager())
    identity = _identity("rejected-spawn")

    await authority.spawn(identity, "denied")

    assert await authority.lookup_response(identity.idempotency_key) == {
        "found": True,
        "id": identity.run_id,
        "error": "spawn refused by governance",
        "code": "spawn_rejected",
        "counted": True,
    }


@pytest.mark.asyncio
async def test_lookup_response_reconstructs_rejected_continuation() -> None:
    coordinator = MemoryRunCoordinator()
    authority = SubagentCommandAuthority(coordinator, _RejectingManager())
    identity = _identity("rejected-continue")

    await authority.continue_conversation(identity, "conversation-one", "denied")

    assert await authority.lookup_response(identity.idempotency_key) == {
        "found": True,
        "id": identity.run_id,
        "error": "conversation_busy: existing run",
        "code": "conversation_busy",
        "counted": True,
    }


@pytest.mark.asyncio
async def test_keyed_continuation_replay_preserves_conversation_and_run_identity() -> None:
    manager = _Manager()
    authority = SubagentCommandAuthority(MemoryRunCoordinator(), manager)
    identity = _identity("continue")

    first = await authority.continue_conversation(
        identity,
        "conversation-one",
        "follow up",
        parent_session_key="dashboard:one",
    )
    replay = await authority.continue_conversation(
        identity,
        "conversation-one",
        "follow up",
        parent_session_key="dashboard:one",
    )

    assert replay is first
    assert len(manager.continue_calls) == 1
    conversation, called_task, called_kwargs = manager.continue_calls[0]
    assert conversation == "conversation-one"
    assert called_task == "follow up"
    assert called_kwargs["parent_session_key"] == "dashboard:one"
    assert called_kwargs["_preassigned_id"] == "run-continue"
    assert called_kwargs["_coordinator_admitted"] is True
    assert called_kwargs["_coordinator_command"].command_id == "command-continue"
    assert called_kwargs["_coordinator_fence"].run_id == "run-continue"
    assert called_kwargs["_coordinator_version"] == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("method", "argument", "expected", "calls_attr"),
    [
        ("steer", "course correct", (True, "ok"), "steer_calls"),
        ("follow_up", "do this next", (True, "queued"), "followup_calls"),
        ("cancel", "", True, "cancel_calls"),
        ("release", "", (True, "released"), "release_calls"),
    ],
)
async def test_keyed_control_replay_invokes_manager_once(
    method: str, argument: str, expected: object, calls_attr: str
) -> None:
    manager = _Manager()
    target = "target-run"
    coordinator = await _coordinator_with_target(target)
    authority = SubagentCommandAuthority(coordinator, manager)
    identity = _identity(method)

    args = (identity, target, argument) if argument else (identity, target)
    first = await getattr(authority, method)(*args)
    replay = await getattr(authority, method)(*args)

    assert first == expected
    assert replay == expected
    assert len(getattr(manager, calls_attr)) == 1
    expected_lookup = {
        "steer": {"found": True, "id": target, "status": "steered"},
        "follow_up": {
            "found": True,
            "id": target,
            "status": "follow_up_queued",
        },
        "cancel": {"found": True, "ok": True, "cancelled": True},
        "release": {
            "found": True,
            "conversation": target,
            "status": "released",
        },
    }[method]
    assert await authority.lookup_response(identity.idempotency_key) == expected_lookup


@pytest.mark.asyncio
async def test_slow_control_result_is_durable_without_replaying_the_side_effect() -> None:
    clock = [100.0]
    manager = _SlowCancelManager(clock)
    coordinator = await _coordinator_with_target("target-run", clock=lambda: clock[0])
    authority = SubagentCommandAuthority(
        coordinator,
        manager,
        clock=lambda: clock[0],
    )
    identity = _identity("slow-cancel")

    first = await authority.cancel(identity, "target-run")
    replay = await authority.cancel(identity, "target-run")

    assert first is True
    assert replay is True
    assert manager.cancel_calls == ["target-run"]


@pytest.mark.asyncio
async def test_control_finish_failure_is_uncertain_and_never_replays_side_effect() -> None:
    clock = [100.0]
    manager = _Manager()
    coordinator = _FinishUnavailableCoordinator(clock=lambda: clock[0])
    authority = SubagentCommandAuthority(coordinator, manager, clock=lambda: clock[0])
    identity = _identity("uncertain-steer")

    with pytest.raises(AuthorityOutcomeUncertain, match="control result"):
        await authority.steer(identity, "target-run", "course correct")
    clock[0] += 31.0
    restarted = SubagentCommandAuthority(coordinator, manager, clock=lambda: clock[0])
    with pytest.raises(AuthorityOutcomeUncertain, match="control outcome"):
        await restarted.steer(identity, "target-run", "course correct")

    assert manager.steer_calls == [("target-run", "course correct")]
