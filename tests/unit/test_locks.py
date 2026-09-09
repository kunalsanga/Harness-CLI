import asyncio
import pytest

from harness_core.agents.locks import WorkspaceLockManager, WorkspaceResource, ResourceMode


@pytest.fixture
def lock_manager():
    return WorkspaceLockManager(workspace_path="/workspace")


@pytest.mark.asyncio
async def test_read_read_concurrent(lock_manager):
    res1 = WorkspaceResource("src/app.py", ResourceMode.READ)
    res2 = WorkspaceResource("src/app.py", ResourceMode.READ)
    
    assert await lock_manager.acquire("task1", [res1])
    assert await lock_manager.acquire("task2", [res2])
    
    assert await lock_manager.is_locked(WorkspaceResource("src/app.py", ResourceMode.WRITE))


@pytest.mark.asyncio
async def test_read_write_conflict(lock_manager):
    res1 = WorkspaceResource("src/app.py", ResourceMode.READ)
    res2 = WorkspaceResource("src/app.py", ResourceMode.WRITE)
    
    assert await lock_manager.acquire("task1", [res1])
    assert not await lock_manager.acquire("task2", [res2])
    
    # If task1 releases, task2 can acquire
    await lock_manager.release_all("task1")
    assert await lock_manager.acquire("task2", [res2])


@pytest.mark.asyncio
async def test_parent_child_conflict(lock_manager):
    res1 = WorkspaceResource("src/frontend", ResourceMode.WRITE)
    res2 = WorkspaceResource("src/frontend/App.tsx", ResourceMode.WRITE)
    
    assert await lock_manager.acquire("task1", [res1])
    assert not await lock_manager.acquire("task2", [res2])
    
    await lock_manager.release_all("task1")
    
    assert await lock_manager.acquire("task2", [res2])
    assert not await lock_manager.acquire("task1", [res1])


@pytest.mark.asyncio
async def test_glob_exact_conflict(lock_manager):
    res1 = WorkspaceResource("src/backend/**", ResourceMode.WRITE)
    res2 = WorkspaceResource("src/backend/app.py", ResourceMode.WRITE)
    
    assert await lock_manager.acquire("task1", [res1])
    assert not await lock_manager.acquire("task2", [res2])


@pytest.mark.asyncio
async def test_glob_glob_conflict(lock_manager):
    res1 = WorkspaceResource("src/backend/**", ResourceMode.WRITE)
    res2 = WorkspaceResource("src/frontend/**", ResourceMode.WRITE)
    res3 = WorkspaceResource("src/**/*.py", ResourceMode.WRITE)
    
    assert await lock_manager.acquire("task1", [res1])
    assert await lock_manager.acquire("task2", [res2])  # Disjoint directories
    assert not await lock_manager.acquire("task3", [res3])  # Overlaps with src/


@pytest.mark.asyncio
async def test_atomic_multi_resource(lock_manager):
    res1 = WorkspaceResource("A.txt", ResourceMode.WRITE)
    res2 = WorkspaceResource("B.txt", ResourceMode.WRITE)
    
    assert await lock_manager.acquire("task1", [res2])
    
    # Task2 requests [A, B], should fail and NOT acquire A
    assert not await lock_manager.acquire("task2", [res1, res2])
    
    # Task3 should be able to get A since Task2 rolled back
    assert await lock_manager.acquire("task3", [res1])


@pytest.mark.asyncio
async def test_self_reentry(lock_manager):
    res1 = WorkspaceResource("A.txt", ResourceMode.WRITE)
    
    assert await lock_manager.acquire("task1", [res1])
    # Re-acquiring the same resource by same task should succeed
    assert await lock_manager.acquire("task1", [res1])
    
    res2 = WorkspaceResource("B.txt", ResourceMode.WRITE)
    assert await lock_manager.acquire("task1", [res2])
    
    assert not await lock_manager.acquire("task2", [res1])


@pytest.mark.asyncio
async def test_windows_paths(lock_manager):
    res1 = WorkspaceResource("src\\frontend", ResourceMode.WRITE)
    res2 = WorkspaceResource("src/frontend/App.tsx", ResourceMode.WRITE)
    
    assert await lock_manager.acquire("task1", [res1])
    assert not await lock_manager.acquire("task2", [res2])
