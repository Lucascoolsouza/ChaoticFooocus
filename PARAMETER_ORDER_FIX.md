# Parameter Order Fix Summary

## Issue
The application was throwing a `ValueError: '896×1152 <span style="color: grey;"> ∣ 7:9</span>' is not a valid Performance` error. This occurred because the parameter order in `async_worker.py` didn't match the order in `webui.py`.

## Root Cause
The `performance_selection` parameter was being popped too early in the `AsyncTask` constructor, causing an aspect ratio string to be passed to the `Performance()` constructor instead of the actual performance value.

## Fix Applied

### Before (Incorrect Order)
```python
# In async_worker.py - WRONG POSITION
self.style_selections = args.pop()
self.performance_selection = Performance(args.pop())  # This was getting aspect ratio instead!
self.aspect_ratios_selection = args.pop()
```

### After (Correct Order)
```python
# In async_worker.py - CORRECT POSITION
self.style_selections = args.pop()
self.aspect_ratios_selection = args.pop()
# ... many other parameters ...
# At the very end, matching webui.py ctrls order:
self.performance_selection = Performance(args.pop())
self.steps = self.performance_selection.steps()
self.original_steps = self.steps
```

## Parameter Flow
The correct flow is now:
1. **webui.py**: `ctrls` list has `performance_selection` at the end
2. **async_worker.py**: `args.pop()` calls match the `ctrls` order
3. **Performance enum**: Gets the correct string value instead of aspect ratio

## Verification
- ✅ All Force Grid integration tests pass
- ✅ Parameter order tests pass
- ✅ Performance enum is used correctly
- ✅ No more ValueError about invalid Performance values

## Files Modified
- `modules/async_worker.py` - Moved `performance_selection` parsing to correct position

## Result
The Force Grid integration now works correctly without parameter order conflicts, and the Performance enum error is resolved.