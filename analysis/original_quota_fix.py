"""Minimal negative-quota guard; valid original allocations are unchanged."""
import inspect
import numpy as np

REPAIRS = []

def repair_integer_counts(counts, capacity):
    values = np.asarray(counts)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
        raise ValueError('Expected a finite, nonempty quota vector')
    if not np.equal(values, np.floor(values)).all():
        raise ValueError('Quotas must be integer-valued')
    if not np.isfinite(capacity) or capacity < 0 or int(capacity) != capacity:
        raise ValueError('Capacity must be a nonnegative integer')
    result = values.astype(np.int64, copy=True)
    if result.sum() != capacity:
        raise ValueError('Original quota total differs from capacity; outside this fix scope')
    if (result >= 0).all():
        return result
    original = result.copy()
    result = np.maximum(result, 0)
    excess = int(result.sum() - capacity)
    # Move the invalid negative correction onto available positive counts.
    # Stable argmax breaks ties by original column order; no random draws.
    while excess:
        index = int(np.argmax(result))
        take = min(excess, int(result[index]))
        if take <= 0:
            raise AssertionError('Insufficient positive quota')
        result[index] -= take
        excess -= take
    assert result.sum() == capacity and (result >= 0).all()
    REPAIRS.append({'before': original.tolist(), 'after': result.tolist(), 'capacity':int(capacity)})
    return result

def install():
    from csmap_runner import _setup_import_path
    _setup_import_path()
    import csmap
    source = inspect.getsource(csmap.main_csmap)
    anchor = 'cell_type_spot_counts[spot_idx] = spot_type_counts'
    assert source.count(anchor) == 1, 'Original source changed; inspect before applying guard'
    source = source.replace(anchor, 'cell_type_spot_counts[spot_idx] = repair_integer_counts(spot_type_counts, spot_total_cells)')
    csmap.__dict__['repair_integer_counts'] = repair_integer_counts
    exec(compile(source, '<original_csmap_with_negative_quota_guard>', 'exec'), csmap.__dict__)
