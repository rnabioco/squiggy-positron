# Issue #92 Implementation Tasks

## Objective
Establish unified extension state so that File Explorer and Sample Comparison Manager coordinate their state while serving distinct roles.

**Issue**: #92 - Unified extension state: File Explorer and Sample Comparison Manager should coordinate while serving distinct roles

## Problem Statement

- **File Explorer** and **Sample Comparison Manager** both interact with the same Python kernel state (`_squiggy_session`)
- However, they operate on **separate extension-level state**, creating sync issues:
  - Loading files in File Explorer doesn't update Sample Comparison Manager
  - Loading files in Sample Comparison Manager doesn't update File Explorer
  - Changes to kernel state aren't reflected across UI panels
  - This creates fragmented workflows and potential data sync bugs

## Design Approach

### Current Architecture (Before)
```
File Explorer           Sample Comparison Manager
     ↓                           ↓
 Extension State A      Extension State B
     ↓                           ↓
    ↘                           ↙
       _squiggy_session (Python)
```

### Target Architecture (After)
```
File Explorer           Sample Comparison Manager
         ↘                   ↙
   Unified Extension State
         ↓
    _squiggy_session (Python)
```

## Implementation Tasks

### Phase 1: Analysis & Design
- [ ] **1.1** Map current state management in File Explorer
  - Identify where state is stored (variables/properties)
  - Document what state is tracked (loaded files, selected files, etc.)
  - Find state change triggers (event listeners, command handlers)

- [ ] **1.2** Map current state management in Sample Comparison Manager
  - Same analysis as File Explorer
  - Identify how it communicates with extension state

- [ ] **1.3** Identify state synchronization points
  - Where do files get loaded?
  - Where does state change need to propagate?
  - What's the minimal set of shared state needed?

- [ ] **1.4** Design unified state structure
  - Define a single source of truth for loaded samples
  - Document what data belongs in unified state
  - Plan how each panel will read/write to shared state

### Phase 2: Implementation
- [ ] **2.1** Create state management module (if needed)
  - Central location for tracking loaded files/samples
  - Event emitter for state changes
  - Type-safe interfaces for state

- [ ] **2.2** Integrate File Explorer with shared state
  - Refactor File Explorer to use unified state
  - Add listeners to state changes
  - Update UI when state changes from Sample Comparison Manager

- [ ] **2.3** Integrate Sample Comparison Manager with shared state
  - Refactor to read from unified state
  - Add listeners to state changes
  - Update UI when state changes from File Explorer

- [ ] **2.4** Implement state synchronization
  - Handle file loading (POD5/BAM/FASTA)
  - Handle file closing/clearing
  - Ensure both panels stay in sync

### Phase 3: Testing
- [ ] **3.1** Test File Explorer → Sample Comparison Manager sync
  - Load file in File Explorer
  - Verify it appears in Sample Comparison Manager
  - Verify Sample Comparison Manager shows correct file metadata

- [ ] **3.2** Test Sample Comparison Manager → File Explorer sync
  - Load file in Sample Comparison Manager
  - Verify it appears in File Explorer
  - Verify File Explorer reflects current state

- [ ] **3.3** Test edge cases
  - Load multiple files sequentially
  - Clear files in one panel, verify other updates
  - Handle errors gracefully

- [ ] **3.4** Test with demo data
  - Load demo data, verify both panels update
  - Switch between samples in comparison mode
  - Verify File Explorer tracks all loaded samples

### Phase 4: Documentation & Cleanup
- [ ] **4.1** Update code comments
  - Document shared state structure
  - Explain synchronization mechanism

- [ ] **4.2** Update CLAUDE.md if needed
  - Add section on state management pattern
  - Document how to extend for future panels

- [ ] **4.3** Test full workflow
  - Demo data loading
  - Multi-file comparison
  - Sample switching

## Key Considerations

### State Management Pattern
- Use events/emitters for loosely-coupled communication
- Avoid circular dependencies between panels
- Each panel should be independently testable

### Performance
- Avoid unnecessary re-renders
- Efficiently update only changed UI elements
- Consider virtualization for large sample lists

### User Experience
- Seamless workflow: load → immediate visibility
- Clear feedback when files are loaded/unloaded
- Consistent state representation across both panels

### Backward Compatibility
- Maintain existing command interfaces
- Don't break API for notebooks/CLI usage
- Gradual migration if refactoring Python state

## Success Criteria

✅ File loading in File Explorer updates Sample Comparison Manager immediately
✅ File loading in Sample Comparison Manager updates File Explorer immediately
✅ Both panels show consistent file/sample metadata
✅ Demo data loading works seamlessly with both panels
✅ State updates don't cause unnecessary re-renders
✅ All existing tests pass
✅ No console pollution or errors from sync mechanism
✅ Documented pattern for future panel integration

## Related Issues
- #86 - Replace File Explorer (deferred, revisit after #92 complete)
- #79 - TSV import & redesign (deferred, separate work)

## Branch & Worktree
- **Branch**: `unify-panels`
- **Worktree**: `/Users/laurawhite/squiggy-unify-panels`
- **Base Branch**: `main`
