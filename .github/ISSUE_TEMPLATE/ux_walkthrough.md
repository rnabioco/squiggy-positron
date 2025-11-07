---
name: UX Walkthrough
about: Systematic UX testing checklist for naive users
title: 'UX Walkthrough: [Your Name/Date]'
labels: ux, testing
assignees: ''
---

# UX Walkthrough Report

**Tester:** [Your name]
**Date:** [YYYY-MM-DD]
**Environment:**
- Positron version:
- Squiggy version:
- OS:
- Test data files:

---

## Instructions

1. Work through each section below in order
2. Check ✓ boxes for expected behaviors you observe
3. Leave boxes unchecked when something doesn't work as expected
4. Add notes in the "Issues found" sections describing what went wrong
5. Complete the summary at the end

---

## 1. Initial Extension Load

### 1.1 Extension Appears in UI

- [ ] **Action:** Open Positron IDE
- [ ] **Action:** Look at Activity Bar (left sidebar)
  - [ ] ✓ Squiggy icon visible
  - [ ] ✓ Icon has tooltip on hover
- [ ] **Action:** Click Squiggy icon
  - [ ] ✓ Sidebar opens
  - [ ] ✓ No errors appear

### 1.2 All Panels Present

- [ ] **Action:** Review sidebar sections
  - [ ] ✓ "Files" panel visible
  - [ ] ✓ "Reads" panel visible
  - [ ] ✓ "Plot Options" panel visible
  - [ ] ✓ "Modifications" panel visible
  - [ ] ✓ Panels expand/collapse

### 1.3 Commands Available

- [ ] **Action:** Open Command Palette (`Cmd/Ctrl+Shift+P`), type "Squiggy"
  - [ ] ✓ Commands appear
  - [ ] ✓ Clear descriptions
  - [ ] ✓ Includes "Open POD5 File", "Open BAM File"

**Issues found:**
<!-- Describe any problems, unexpected behaviors, or missing features here -->


---

## 2. Opening POD5 Files

### 2.1 First Load (No Kernel)

- [ ] **Action:** Run `Squiggy: Open POD5 File` without Python kernel
  - [ ] ✓ Prompt to start kernel OR clear error message
- [ ] **Action:** Start Python kernel and retry
  - [ ] ✓ File loads successfully

### 2.2 Load with Active Kernel

- [ ] **Action:** Load POD5 file with kernel running
  - [ ] ✓ File loads within 10 seconds (small file)
  - [ ] ✓ No error messages

### 2.3 Files Panel Updates

- [ ] **Action:** Check Files panel
  - [ ] ✓ Filename displayed
  - [ ] ✓ Read count shown
  - [ ] ✓ File size shown
  - [ ] ✓ Sample rate shown

### 2.4 Reads Panel Populates

- [ ] **Action:** Check Reads panel
  - [ ] ✓ Read IDs appear
  - [ ] ✓ Columns: Read ID, Length, Quality, Reference, Position
  - [ ] ✓ Read count matches Files panel
  - [ ] ✓ Data loads within 5 seconds

### 2.5 Multiple Files

- [ ] **Action:** Load second POD5 file
  - [ ] ✓ Confirmation prompt OR auto-replace
  - [ ] ✓ UI updates with new file
  - [ ] ✓ Old reads cleared

### 2.6 Large File Performance

- [ ] **Action:** Load file with >10,000 reads
  - [ ] ✓ UI stays responsive
  - [ ] ✓ Smooth scrolling in Reads panel

**Issues found:**
<!-- Describe any problems, unexpected behaviors, or missing features here -->


---

## 3. Reads Panel Interactions

### 3.1 Basic Display

- [ ] **Action:** Review Reads table
  - [ ] ✓ All columns visible
  - [ ] ✓ Data is readable
  - [ ] ✓ Rows align properly

### 3.2 Scrolling

- [ ] **Action:** Scroll through reads list
  - [ ] ✓ Smooth scrolling (no lag)
  - [ ] ✓ No flickering
  - [ ] ✓ Fast rendering

### 3.3 Column Resizing

- [ ] **Action:** Drag column dividers
  - [ ] ✓ Cursor changes on hover
  - [ ] ✓ Columns resize smoothly
  - [ ] ✓ Minimum width enforced
  - [ ] ✓ Text truncates properly

### 3.4 Sorting

- [ ] **Action:** Click column headers
  - [ ] ✓ Sorts ascending/descending
  - [ ] ✓ Sort indicator visible
  - [ ] ✓ Works for all sortable columns

### 3.5 Search

- [ ] **Action:** Use search box (if present)
  - [ ] ✓ Filters as you type
  - [ ] ✓ Clear button works
  - [ ] ✓ "No results" message when empty

### 3.6 Selection

- [ ] **Action:** Click a read row
  - [ ] ✓ Row highlights
  - [ ] ✓ Selection persists

**Issues found:**
<!-- Describe any problems, unexpected behaviors, or missing features here -->


---

## 4. BAM File Integration

### 4.1 Loading BAM

- [ ] **Action:** Load BAM file after POD5
  - [ ] ✓ BAM loads successfully
  - [ ] ✓ Files panel shows BAM info
  - [ ] ✓ Reads panel updates

### 4.2 Reference Grouping

- [ ] **Action:** Review grouped reads
  - [ ] ✓ Grouped by reference
  - [ ] ✓ Groups are collapsible
  - [ ] ✓ Read counts per group shown
  - [ ] ✓ Expand/collapse works

### 4.3 Alignment Columns

- [ ] **Action:** Check Reference and Position columns
  - [ ] ✓ Reference names populate
  - [ ] ✓ Positions populate
  - [ ] ✓ Data matches BAM file

### 4.4 BAM Without POD5

- [ ] **Action:** Try loading BAM without POD5
  - [ ] ✓ Clear error message
  - [ ] ✓ Suggests loading POD5 first

**Issues found:**
<!-- Describe any problems, unexpected behaviors, or missing features here -->


---

## 5. Plotting Reads

### 5.1 Single Read Plot (Default)

- [ ] **Action:** Right-click read → "Plot Read"
  - [ ] ✓ Plot panel opens
  - [ ] ✓ Bokeh visualization appears
  - [ ] ✓ Loads within 5 seconds
  - [ ] ✓ Title shows read ID
  - [ ] ✓ Axes labeled correctly

### 5.2 Plot Interactivity

- [ ] **Action:** Use Bokeh tools
  - [ ] ✓ Pan tool works
  - [ ] ✓ Zoom tool works
  - [ ] ✓ Reset tool works
  - [ ] ✓ Hover tooltip shows data (if enabled)
  - [ ] ✓ Tools are responsive

### 5.3 Multiple Plots

- [ ] **Action:** Plot multiple different reads
  - [ ] ✓ New plots appear (replace or new tab)
  - [ ] ✓ No crashes
  - [ ] ✓ Each plot independent

**Issues found:**
<!-- Describe any problems, unexpected behaviors, or missing features here -->


---

## 6. Plot Options Panel

### 6.1 Plot Mode Selection

- [ ] **Action:** Review available modes
  - [ ] ✓ SINGLE mode available
  - [ ] ✓ OVERLAY mode available
  - [ ] ✓ STACKED mode available
  - [ ] ✓ EVENTALIGN mode available
  - [ ] ✓ Mode selector is clear

### 6.2 Mode Changes

- [ ] **Action:** Change mode and replot
  - [ ] ✓ Plot updates correctly
  - [ ] ✓ Mode change is immediate or clearly communicated

### 6.3 Normalization Options

- [ ] **Action:** Try normalization methods
  - [ ] ✓ NONE option works
  - [ ] ✓ ZNORM option works
  - [ ] ✓ MEDIAN option works
  - [ ] ✓ MAD option works
  - [ ] ✓ Signal appearance changes appropriately

### 6.4 X-Axis Scaling

- [ ] **Action:** Toggle "Scale X by dwell time"
  - [ ] ✓ Checkbox works
  - [ ] ✓ X-axis updates when replotting
  - [ ] ✓ Labels change appropriately

**Issues found:**
<!-- Describe any problems, unexpected behaviors, or missing features here -->


---

## 7. Modifications Panel

### 7.1 Panel State (No Modifications)

- [ ] **Action:** Load POD5+BAM without mod tags
  - [ ] ✓ Panel shows "No modifications" message OR
  - [ ] ✓ Panel is disabled/grayed out

### 7.2 Panel State (With Modifications)

- [ ] **Action:** Load BAM with modifications
  - [ ] ✓ Panel becomes active
  - [ ] ✓ Modification types listed (5mC, 6mA, etc.)
  - [ ] ✓ Checkboxes/filters visible

### 7.3 Filtering Modifications

- [ ] **Action:** Enable/disable modification types
  - [ ] ✓ Checkboxes work
  - [ ] ✓ Threshold slider works
  - [ ] ✓ Plot updates show filtered mods

**Issues found:**
<!-- Describe any problems, unexpected behaviors, or missing features here -->


---

## 8. Export Functionality

### 8.1 HTML Export

- [ ] **Action:** Export plot as HTML
  - [ ] ✓ Save dialog appears
  - [ ] ✓ File saves successfully
  - [ ] ✓ HTML opens in browser
  - [ ] ✓ Interactivity preserved

### 8.2 PNG/SVG Export

- [ ] **Action:** Export as image
  - [ ] ✓ Export options available
  - [ ] ✓ Image quality good
  - [ ] ✓ Resolution appropriate

**Issues found:**
<!-- Describe any problems, unexpected behaviors, or missing features here -->


---

## 9. Error Handling

### 9.1 Invalid Files

- [ ] **Action:** Try opening wrong file types
  - [ ] ✓ Clear error messages
  - [ ] ✓ No crashes
  - [ ] ✓ Helpful suggestions

### 9.2 Kernel Issues

- [ ] **Action:** Restart kernel during operation
  - [ ] ✓ Graceful handling
  - [ ] ✓ Clear error or auto-reconnect

### 9.3 Missing Data

- [ ] **Action:** Search for non-existent read
  - [ ] ✓ "No results" message
  - [ ] ✓ No errors

**Issues found:**
<!-- Describe any problems, unexpected behaviors, or missing features here -->


---

## 10. Performance

### 10.1 Large Files

- [ ] **Action:** Test with >50,000 reads
  - [ ] ✓ Loading time acceptable
  - [ ] ✓ UI responsive during load
  - [ ] ✓ Scrolling remains smooth

### 10.2 Rapid Operations

- [ ] **Action:** Quickly plot 10+ reads
  - [ ] ✓ No crashes
  - [ ] ✓ Handles queue properly

**Issues found:**
<!-- Describe any problems, unexpected behaviors, or missing features here -->


---

## 11. Accessibility & Usability

### 11.1 Keyboard Navigation

- [ ] **Action:** Use Tab/Arrow keys
  - [ ] ✓ Focus moves logically
  - [ ] ✓ Arrow keys work in lists
  - [ ] ✓ Enter triggers actions

### 11.2 Visual Clarity

- [ ] **Action:** Review overall UI
  - [ ] ✓ Good color contrast
  - [ ] ✓ Clear labels
  - [ ] ✓ Tooltips helpful
  - [ ] ✓ Icons recognizable

**Issues found:**
<!-- Describe any problems, unexpected behaviors, or missing features here -->


---

## Summary

**Total Issues Found:** [Number]

**Critical Issues (blocking):**
-

**Major Issues (significant UX problems):**
-

**Minor Issues (polish):**
-

**Suggested Improvements:**
-

**What Worked Well:**
-

**Biggest Pain Points:**
-

**Overall UX Rating (1-10):**

**Additional Comments:**


