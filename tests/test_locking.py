# tests/test_locking.py
import threading

from turbodb.locking import FileLock


def test_lock_creates_file(tmp_path):
    lock_path = tmp_path / "test.lock"
    with FileLock(lock_path):
        assert lock_path.exists()


def test_lock_is_reentrant_from_same_thread(tmp_path):
    lock_path = tmp_path / "test.lock"
    with FileLock(lock_path):
        # Should not deadlock
        with FileLock(lock_path):
            assert True


def test_lock_serializes_threads(tmp_path):
    lock_path = tmp_path / "test.lock"
    order = []

    def worker(name, delay):
        import time
        with FileLock(lock_path):
            order.append(f"{name}_start")
            time.sleep(delay)
            order.append(f"{name}_end")

    t1 = threading.Thread(target=worker, args=("a", 0.1))
    t2 = threading.Thread(target=worker, args=("b", 0.0))
    t1.start()
    import time
    time.sleep(0.02)  # Ensure t1 gets lock first
    t2.start()
    t1.join()
    t2.join()

    # t1 should complete before t2 starts
    assert order.index("a_end") < order.index("b_start")
