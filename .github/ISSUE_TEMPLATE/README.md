# Issue Templates

## UX Testing Workflow

When conducting UX testing for Squiggy:

### 1. Create a UX Walkthrough Issue

Use the **UX Walkthrough Report** template to track your testing session:

- Fill out the checklist as you test
- Check off expected behaviors you observe
- Note any that don't work as expected

### 2. Create Individual Bug Issues

When you find a specific problem:

- Create a separate issue using the **UX Bug (from walkthrough)** template
- Link it back to your walkthrough issue
- This keeps bugs trackable and actionable

### Example Workflow

```
Main walkthrough issue: #150 "UX Walkthrough: John Doe 2025-01-15"
├─ Bug #151: [UX Bug] Read selection doesn't highlight row
├─ Bug #152: [UX Bug] BAM loading shows no progress indicator
└─ Bug #153: [UX Bug] Column resize cursor doesn't appear on macOS
```

## Template Guide

- **ux_walkthrough.md**: Comprehensive testing checklist for complete UX testing sessions
- **ux_bug.md**: Individual bug reports found during testing (links back to walkthrough issue)

This approach keeps the main walkthrough issue as a high-level tracking document while maintaining detailed, actionable bug reports in separate issues.
