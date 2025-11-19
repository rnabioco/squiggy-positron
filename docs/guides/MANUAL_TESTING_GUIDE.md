# Phase 2 Manual Testing Guide - Positron

Complete step-by-step instructions for verifying unified state functionality in Positron IDE.

## Setup

Before starting tests:
1. Build the extension: `npm run compile`
2. Open Positron
3. Open this project folder in Positron
4. Press `F5` to launch Extension Development Host
5. Open the test data folder: `tests/data/`

### Test Data Files
- `yeast_trna_reads.pod5` - 180 reads POD5 file
- `yeast_trna_mappings.bam` - Corresponding BAM file (alignments)

---

## Test 1: File Panel Load → Samples Panel Aware

**Objective**: Verify that loading a POD5 via File Panel notifies Samples Panel through unified state

### Steps

1. **Open the File Panel**
   - View → Sidebar → Look for "Squiggy File Explorer" panel
   - Should be empty initially

2. **Open POD5 File**
   - Command Palette (Cmd+Shift+P)
   - Type: `Squiggy: Open POD5 File`
   - Select: `tests/data/yeast_trna_reads.pod5`
   - Click "Open"

3. **Verify File Panel Updated**
   - ✅ File Panel should show:
     - Filename: `yeast_trna_reads.pod5`
     - Type: `POD5`
     - Number of reads: `180`
     - File size: Should display (e.g., "1.2 MB")

4. **Check Extension Output**
   - View → Output Panel
   - Select "Extension Host" from dropdown
   - Look for logs:
     - `[FilePanelProvider] Unified state changed, now showing 1 items`
     - `[SamplesPanelProvider]` messages if Samples Panel is subscribed

5. **Expected Result** ✅
   - File Panel displays the loaded POD5
   - No errors in console
   - Extension state contains the loaded item

---

## Test 2: Samples Panel Load → File Panel Aware

**Objective**: Verify that loading a sample via Samples Panel notifies File Panel through unified state

### Steps

1. **Open Samples Panel**
   - View → Sidebar → Look for "Sample Comparison Manager"
   - Should be empty initially

2. **Load Sample via Demo**
   - Click "Load Demo Session" button in Samples Panel (if available)
   - OR manually load: Command Palette → `Squiggy: Load Samples From Dropped`

3. **Verify Samples Panel Updated**
   - ✅ Samples Panel should show:
     - At least one sample entry
     - Sample name displayed
     - Read count
     - File size

4. **Check File Panel**
   - File Panel should now show:
     - The sample's POD5 file
     - Type: Should indicate it's from a sample
     - Same read count as Samples Panel shows

5. **Check Console Logs**
   - Look for:
     - `[FilePanelProvider] Unified state changed, now showing X items`
     - `[SamplesPanelProvider] onLoadedItemsChanged` messages

6. **Expected Result** ✅
   - Both panels show consistent data
   - File Panel automatically updated when Samples Panel loaded data
   - No console errors

---

## Test 3: BAM Association Updates Both Panels

**Objective**: Verify that associating a BAM file updates both File and Samples panels

### Prerequisites
- Complete Test 1 (POD5 loaded)
- Have BAM file available: `tests/data/yeast_trna_mappings.bam`

### Steps

1. **Verify Current State**
   - File Panel shows POD5 without BAM info
   - Check console: No BAM association yet

2. **Load BAM File**
   - Command Palette → `Squiggy: Open BAM File`
   - Select: `tests/data/yeast_trna_mappings.bam`
   - Click "Open"

3. **Verify File Panel Updated**
   - ✅ File Panel should show:
     - POD5 still displayed
     - New field showing: "Alignments: Yes" or similar
     - BAM path associated with POD5

4. **Verify Samples Panel Updated** (if samples loaded)
   - If samples are loaded:
     - Each sample should show: "Has BAM: Yes"
     - Alignment info displayed

5. **Check Console**
   - Look for:
     - `[FilePanelProvider] Unified state changed` (BAM update)
     - `[SamplesPanelProvider] Unified state changed` (if samples loaded)

6. **Expected Result** ✅
   - Both panels show BAM association
   - Events fired for state update
   - No console errors

---

## Test 4: Comparison Mode Selection

**Objective**: Verify comparison mode selection through unified state

### Prerequisites
- Multiple samples loaded in Samples Panel (load demo session or multiple samples)

### Steps

1. **Open Samples Panel**
   - Verify at least 2 samples are displayed

2. **Select Samples for Comparison**
   - Click checkbox next to first sample
   - Click checkbox next to second sample
   - ✅ Both should show as selected (checkmarks visible)

3. **Check Console Logs**
   - Look for:
     - `[SamplesPanelProvider] Comparison selection changed: [sample:name1, sample:name2]`
     - Or similar indicating comparison items set

4. **Verify Unified State**
   - Command Palette → `Squiggy: Debug Extension State` (if command exists)
   - Or check Output panel for state information

5. **Trigger Comparison**
   - Click "Compare Samples" button in Samples Panel
   - Should generate plot with 2+ samples overlaid

6. **Expected Result** ✅
   - Samples appear selected
   - Console shows comparison IDs with "sample:" prefix
   - Plot generates with correct samples
   - No errors

---

## Test 5: Item Removal Synchronization

**Objective**: Verify that removing items from one panel updates other panels

### Steps

1. **Verify Starting State**
   - File Panel shows 1+ items
   - Samples Panel shows 1+ samples

2. **Unload POD5 from File Panel**
   - Command Palette → `Squiggy: Close POD5 File`
   - Click "Confirm" in dialog

3. **Verify File Panel Updated**
   - ✅ File Panel should become empty or show remaining items

4. **Check Samples Panel** (if samples were from this POD5)
   - If samples depend on this POD5:
     - Samples Panel might also update
     - Or samples remain if they're independent

5. **Check Console**
   - Look for:
     - `[FilePanelProvider] Unified state changed, now showing 0 items`
     - `[SamplesPanelProvider]` updates if affected

6. **Expected Result** ✅
   - File Panel clears correctly
   - Other panels update if dependent
   - State remains consistent
   - No orphaned references

---

## Test 6: Session Save/Restore

**Objective**: Verify that sessions properly persist and restore unified state

### Prerequisites
- At least 1 POD5 loaded
- Ideally: 1-2 samples with BAM association

### Steps

1. **Verify Current State**
   - Note what's loaded in File Panel
   - Note what's loaded in Samples Panel
   - Note any comparison selections

2. **Save Session**
   - File → Save Workspace (or similar)
   - Or use extension command: `Squiggy: Save Session`

3. **Check Session File**
   - Look in workspace settings or `.vscode/settings.json`
   - Session should contain sample data and paths

4. **Close All Files**
   - Command Palette → `Squiggy: Close POD5 File`
   - Command Palette → `Squiggy: Close BAM File`
   - Verify panels are empty

5. **Reload Session**
   - Reload the Positron window (Cmd+R or View → Reload)
   - Wait for extension to activate

6. **Verify Restoration**
   - ✅ File Panel should show:
     - Same files as before reload
     - Same metadata (read counts, sizes)

   - ✅ Samples Panel should show:
     - Same samples as before
     - BAM associations if they existed

   - ✅ Check Console:
     - Look for session restore messages
     - No errors during restoration

7. **Expected Result** ✅
   - All items restored exactly as they were
   - Session state fully recovered
   - Comparison selections restored (if applicable)
   - No console errors

---

## Test 7: Edge Case - Duplicate Item

**Objective**: Verify that adding the same file twice replaces rather than duplicates

### Steps

1. **Load POD5**
   - Command Palette → `Squiggy: Open POD5 File`
   - Select same file twice (or load, then load again)

2. **Check File Panel**
   - ✅ Should show exactly 1 POD5 entry (not 2)
   - File info updated (if metadata changed)

3. **Check Console**
   - Should see state update, not duplicate addition

4. **Expected Result** ✅
   - No duplicate items in unified state
   - Previous item replaced with new one
   - Consistent behavior

---

## Test 8: Edge Case - Removal During Comparison

**Objective**: Verify that removing a sample in comparison mode cleans up automatically

### Prerequisites
- Multiple samples loaded
- At least 2 samples selected for comparison

### Steps

1. **Verify Comparison Mode**
   - Samples Panel shows selected checkmarks
   - Console shows comparison IDs

2. **Unload Sample from Comparison**
   - Right-click sample → "Unload Sample"
   - Click "Yes" to confirm

3. **Check Console**
   - Look for:
     - `[SamplesPanelProvider] Comparison selection changed`
     - Updated comparison IDs should not include removed sample

4. **Verify File Panel**
   - File Panel should no longer show that sample's POD5

5. **Expected Result** ✅
   - Sample removed from file panel
   - Sample removed from comparison selections
   - Events fired correctly
   - No orphaned state

---

## Test 9: Large File List Performance

**Objective**: Verify UI remains responsive with many items

### Steps

1. **Load Multiple Samples**
   - Use Command Palette to load 10-20 samples
   - Or load demo session that has many samples

2. **Monitor Performance**
   - File Panel should render all items
   - UI should remain responsive (no freezing)
   - Scrolling should be smooth

3. **Check Console**
   - Look for performance warnings
   - Event firing should be reasonable frequency

4. **Trigger Updates**
   - Load/unload additional files
   - Verify UI updates quickly

5. **Expected Result** ✅
   - UI responsive with 50+ items
   - No performance degradation
   - Events fire efficiently

---

## Test 10: Regression - Legacy Workflow

**Objective**: Verify backward compatibility with legacy (non-unified) code paths

### Steps

1. **Single File Mode**
   - Load POD5 via command
   - Load BAM via command
   - Verify plots generate correctly with old API

2. **Plot Generation**
   - Select read from reads panel
   - Plot should generate as before
   - No errors

3. **File Operations**
   - Open/close files via commands
   - Should work exactly as before unified state

4. **Expected Result** ✅
   - All legacy features work unchanged
   - No regressions in existing functionality

---

## Verification Checklist

After running all tests, verify:

- [ ] All console logs are clean (no errors)
- [ ] File Panel always matches Samples Panel on shared files
- [ ] Session save/restore works end-to-end
- [ ] Comparison mode selections properly maintained
- [ ] Removal of items properly synchronized
- [ ] BAM association updates both panels
- [ ] No duplicate items in unified state
- [ ] UI responsive with 50+ items
- [ ] Legacy workflows still function
- [ ] No memory leaks (check Chrome DevTools)

---

## Troubleshooting

### "File Panel shows file but Samples Panel doesn't know about it"
- Check Extension Host output for errors
- Look for `_handleLoadedItemsChanged` in logs
- Verify subscription is active in Samples Panel constructor

### "BAM association not showing in both panels"
- Verify BAM file path is correct
- Check console for BAM loading errors
- Verify `addLoadedItem` is called with updated item

### "Session doesn't restore"
- Check that items were in unified state before save
- Verify `toSessionState()` is populating samples correctly
- Check Extension Host logs for restoration errors

### "Removed item still appears in panels"
- Verify `removeLoadedItem` is being called
- Check that item ID is exactly correct (prefixes matter)
- Look for lingering event subscriptions

---

## Notes

- Console output is verbose - this is helpful for debugging
- Event handlers may fire multiple times during operations (expected)
- Session restore may take a few seconds
- BAM loading requires the file to exist on disk
- Demo session loads test data from `tests/data/`

Good luck with testing! 🚀
