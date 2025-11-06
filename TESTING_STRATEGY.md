# Testing Strategy & Coverage Roadmap

**Status**: Phase 1 & 2 Complete ✅ | Phase 6 Configuration Complete ✅
**Current Coverage**: TypeScript: 2-4% | Python: 65.77%
**Target Coverage**: TypeScript: 60%+ | Python: 80%+

---

## 📊 Current Coverage Baseline

### TypeScript Coverage (125 tests passing)
```
Global:           ~2-4% (target: 60%)
Backend:          0-24% (target: 75%)
Utils:            28% (target: 80%)
State:            5-31% (target: 75%)
Services:         75% (target: 80%) ✅ CLOSE!
Commands:         0% (target: 60%)
Views:            0% (target: 60%)
Webviews:         0% (target: 60%)
```

### Python Coverage (587 tests passing)
```
Global:                           65.77% (target: 80%)
squiggy/__init__.py:              40.50% ⚠️
squiggy/cache.py:                 41.36% ⚠️
squiggy/utils.py:                 43.02% ⚠️
squiggy/modifications.py:         56.31%
squiggy/io.py:                    73.92%
squiggy/alignment.py:             95.00% ✅
squiggy/api.py:                   89.93% ✅
squiggy/normalization.py:         100.00% ✅
squiggy/plot_factory.py:          100.00% ✅
squiggy/constants.py:             100.00% ✅
```

---

## 🎯 Phase-by-Phase Testing Plan

### ✅ Phase 1-2: Foundation (COMPLETE)

**Completed Work:**
- ✅ Created 5 typed error classes (POD5Error, BAMError, FASTAError, PlottingError, ValidationError)
- ✅ Added retry logic with exponential backoff to PositronRuntimeClient
- ✅ Enhanced SquiggyRuntimeAPI with input validation and typed errors
- ✅ Added comprehensive error handling to BaseWebviewProvider
- ✅ All 125 existing tests passing

**Files Modified:**
- `src/utils/error-handler.ts` - New error classes + retry utilities
- `src/backend/positron-runtime-client.ts` - Retry logic, reduced logging
- `src/backend/squiggy-runtime-api.ts` - Validation + typed errors
- `src/views/base-webview-provider.ts` - 4 new error handling methods
- `src/types/messages.ts` - ErrorOutgoingMessage type
- `src/__mocks__/vscode.ts` - Enhanced mock API

---

### 🔄 Phase 3: Command & Provider Tests (NEXT PRIORITY)

**Target**: 80+ new tests, 60%+ coverage on commands/views

#### 3.1 Command Handler Tests (~40 tests)

**Priority Files** (0% coverage):
```typescript
src/commands/file-commands.ts     // 0% → 80%+ (20 tests)
  ✓ Test openPOD5, openBAM, openFASTA commands
  ✓ Test closePOD5, closeBAM, closeFASTA commands
  ✓ Test loadTestData command
  ✓ Test loadSample command
  ✓ Test error handling for missing files
  ✓ Test user cancellation flows

src/commands/plot-commands.ts     // 0% → 80%+ (12 tests)
  ✓ Test plotRead command
  ✓ Test plotMultipleReads command
  ✓ Test plotAggregate command
  ✓ Test plot option changes
  ✓ Test modification filter integration

src/commands/session-commands.ts // 0% → 80%+ (5 tests)
  ✓ Test saveSession command
  ✓ Test loadSession command
  ✓ Test session state serialization

src/commands/state-commands.ts   // 0% → 80%+ (3 tests)
  ✓ Test refreshReadList command
  ✓ Test clearState command
```

**Testing Approach:**
- Mock ExtensionState and SquiggyRuntimeAPI
- Test command registration
- Test user interaction flows (dialogs, confirmations)
- Test error handling and recovery
- Use existing `file-commands.test.ts` as template (needs TypeScript fixes)

#### 3.2 Panel Provider Tests (~40 tests)

**Priority Files** (0% coverage):
```typescript
src/views/squiggy-modifications-panel.ts    // 0% → 75%+ (8 tests)
  ✓ Test filter state management
  ✓ Test onDidChangeFilters event
  ✓ Test webview message handling
  ✓ Test UI updates when modifications change

src/views/squiggy-plot-options-view.ts     // 0% → 75%+ (8 tests)
  ✓ Test option state management
  ✓ Test onDidChangeOptions event
  ✓ Test aggregate plot request handling

src/views/squiggy-samples-panel.ts         // 0% → 75%+ (10 tests)
  ✓ Test sample list rendering
  ✓ Test onDidRequestUnload event
  ✓ Test comparison selection logic

src/views/squiggy-session-panel.ts         // 0% → 75%+ (6 tests)
  ✓ Test session display
  ✓ Test save/load integration

src/views/squiggy-reads-view-pane.ts       // 0% → 75%+ (8 tests)
  ✓ Test read list updates
  ✓ Test filtering and search
  ✓ Test reference grouping
```

**Testing Approach:**
- Extend BaseWebviewProvider tests
- Mock webview message passing
- Test event emission and handling
- Test visibility changes
- Test error handling via new BaseWebviewProvider methods

---

### 🔄 Phase 4: React & Webview Tests

**Target**: 40+ new tests, 70%+ coverage on React components

#### 4.1 React Component Tests (~30 tests)

**Setup Required:**
```bash
npm install --save-dev @testing-library/react @testing-library/jest-dom
```

**Priority Files** (0% coverage):
```typescript
src/views/components/squiggy-reads-core.tsx          // 0% → 70%+ (10 tests)
  ✓ Test read list rendering
  ✓ Test virtualization (react-window)
  ✓ Test sorting and filtering
  ✓ Test selection handling
  ✓ Test search functionality

src/views/components/squiggy-reads-instance.tsx      // 0% → 70%+ (5 tests)
  ✓ Test webview integration
  ✓ Test message passing
  ✓ Test state updates

src/views/components/squiggy-read-item.tsx           // 0% → 70%+ (5 tests)
  ✓ Test individual read rendering
  ✓ Test click handlers
  ✓ Test metadata display

src/views/components/squiggy-reference-group.tsx     // 0% → 70%+ (5 tests)
  ✓ Test grouping logic
  ✓ Test expand/collapse

src/views/components/column-resizer.tsx              // 0% → 70%+ (5 tests)
  ✓ Test resize drag handlers
  ✓ Test column width updates
```

**Testing Approach:**
- Use @testing-library/react for component rendering
- Test user interactions (clicks, drags, searches)
- Test prop changes and re-renders
- Mock postMessage for webview communication

#### 4.2 Webview Tests (~10 tests)

```typescript
src/webview/squiggy-plot-panel.ts           // 0% → 75%+ (10 tests)
  ✓ Test plot panel creation
  ✓ Test HTML content updates
  ✓ Test export functionality (HTML, PNG, SVG)
  ✓ Test zoom-level export
  ✓ Test error display
```

---

### 🔄 Phase 5: Integration & Edge Cases

**Target**: 30+ new tests, 80%+ Python coverage

#### 5.1 Integration Tests (~10 tests)

```typescript
tests/integration/file-loading-flow.test.ts         // New file
  ✓ Test complete POD5 → reads flow
  ✓ Test POD5 → BAM → annotations flow
  ✓ Test multi-sample loading
  ✓ Test file associations

tests/integration/plotting-flow.test.ts             // New file
  ✓ Test POD5 load → plot generation
  ✓ Test option changes → plot refresh
  ✓ Test modification filtering → plot update

tests/integration/sample-comparison-flow.test.ts    // New file
  ✓ Test loading multiple samples
  ✓ Test comparison selection
  ✓ Test aggregate plot generation
```

#### 5.2 Python Edge Case Tests (~20 tests)

**Priority Gaps:**
```python
tests/test_io_edge_cases.py                    // New file (10 tests)
  ✓ Test malformed POD5 files
  ✓ Test missing BAM index
  ✓ Test corrupted BAM headers
  ✓ Test empty POD5 files
  ✓ Test concurrent file access
  ✓ Test invalid FASTA format
  ✓ Test memory limits on large files

tests/test_cache_edge_cases.py                 // New file (5 tests)
  ✓ Test cache expiration
  ✓ Test cache invalidation
  ✓ Test concurrent cache access

tests/test_modifications_edge_cases.py         // New file (5 tests)
  ✓ Test missing MM/ML tags
  ✓ Test malformed modification strings
  ✓ Test zero-probability modifications
```

**Focus Areas** (from coverage report):
- `squiggy/__init__.py`: 40.50% → 80%+ (main API edge cases)
- `squiggy/cache.py`: 41.36% → 80%+ (cache invalidation)
- `squiggy/utils.py`: 43.02% → 80%+ (utility edge cases)
- `squiggy/modifications.py`: 56.31% → 80%+ (mod parsing errors)

---

### ✅ Phase 6: Coverage & Documentation (COMPLETE)

**Completed:**
- ✅ Configured Jest coverage thresholds (60% global, 75-80% critical modules)
- ✅ Configured pytest coverage (80% threshold, branch coverage enabled)
- ✅ Added coverage reporters (HTML, LCOV, JSON summary)
- ✅ Verified CI coverage reporting (Codecov integration already working)
- ✅ Documented testing strategy (this file)

**Coverage Configuration:**
```javascript
// jest.config.js - TypeScript
coverageThreshold: {
  global: { statements: 60%, branches: 55%, functions: 60%, lines: 60% },
  backend: 75%,
  utils: 80%,
  state: 75%,
  services: 80%,
}
```

```toml
# pyproject.toml - Python
[tool.coverage.report]
fail_under = 80
branch = true
```

---

## 📈 Coverage Milestones

| Milestone | TypeScript | Python | Total Tests | ETA |
|-----------|------------|--------|-------------|-----|
| **Baseline** | 2-4% | 65.77% | 712 | ✅ Done |
| **Phase 3** | 40%+ | 65.77% | 792+ | Week 3-4 |
| **Phase 4** | 55%+ | 65.77% | 832+ | Week 4-5 |
| **Phase 5** | 60%+ | 80%+ | 862+ | Week 5-6 |
| **Target** | 60%+ | 80%+ | 862+ | Week 6 |

---

## 🔧 Testing Utilities & Helpers

### Test Helper Functions Needed

```typescript
// tests/helpers/command-test-helpers.ts
export function createMockExtensionContext(): vscode.ExtensionContext
export function createMockExtensionState(): ExtensionState
export function createMockSquiggyAPI(): SquiggyRuntimeAPI

// tests/helpers/webview-test-helpers.ts
export function createMockWebviewView(): vscode.WebviewView
export function simulateWebviewMessage(view: vscode.WebviewView, message: any)
export function waitForWebviewUpdate(view: vscode.WebviewView, timeout?: number)

// tests/helpers/file-test-helpers.ts
export function createTempPOD5(): string
export function createTempBAM(): string
export async function cleanupTempFiles(paths: string[])
```

### Mock Enhancement Needed

```typescript
// src/__mocks__/vscode.ts - Already enhanced ✅
- window.showOpenDialog ✅
- window.showErrorMessage ✅
- commands.registerCommand ✅
- commands.executeCommand ✅
- ColorThemeKind enum ✅

// Still needed:
- Disposable interface
- CancellationToken interface
- Progress reporting mock
```

---

## 🚀 Getting Started with Testing

### Run Current Tests
```bash
# All tests
npm test                    # TypeScript (125 tests)
pytest                      # Python (587 tests)

# With coverage
npm run test:coverage       # TypeScript + coverage report
pytest                      # Python + coverage (auto-enabled)

# Watch mode
npm run test:watch          # TypeScript watch mode
pytest --watch              # Python watch mode (needs pytest-watch)
```

### View Coverage Reports
```bash
# Generate reports
npm run test:coverage       # Creates coverage/ directory
pytest                      # Creates coverage_html/ directory

# Open in browser
open coverage/index.html              # TypeScript
open coverage_html/index.html         # Python
```

### Write New Tests

1. **Create test file** in appropriate `__tests__/` directory
2. **Import test helpers** from `tests/helpers/`
3. **Follow existing patterns** (see `base-webview-provider.test.ts`, `file-loading-service.test.ts`)
4. **Run coverage** to verify improvement
5. **Update this document** with progress

---

## 📝 Notes & Best Practices

### TypeScript Testing
- ✅ Use Jest + ts-jest
- ✅ Mock vscode API via `src/__mocks__/vscode.ts`
- ✅ Use `beforeEach` to reset mocks
- ✅ Test both success and error paths
- ✅ Verify error messages sent to users
- ⚠️ Fix TypeScript strict mode issues in new tests

### Python Testing
- ✅ Use pytest + pytest-cov
- ✅ Leverage fixtures for test data
- ✅ Test with actual POD5/BAM files from `squiggy/data/`
- ✅ Use `@pytest.mark.parametrize` for edge cases
- ✅ Test concurrent access scenarios

### Common Pitfalls
- ❌ Don't test implementation details
- ❌ Don't over-mock (use real objects when possible)
- ❌ Don't skip error paths
- ❌ Don't forget to test async code properly
- ✅ DO test user-facing behavior
- ✅ DO test error messages
- ✅ DO test state management

---

## 🎯 Success Criteria

**Phase 3-5 Complete When:**
- ✅ TypeScript coverage ≥ 60% (all modules)
- ✅ Python coverage ≥ 80% (all modules)
- ✅ All CI tests passing
- ✅ No regressions in existing tests
- ✅ Coverage badges updated in README
- ✅ DEVELOPER.md updated with testing patterns

**Definition of Done:**
- Code merged to `main`
- Coverage thresholds passing in CI
- Documentation updated
- No breaking changes to public APIs

---

**Last Updated**: Phase 1-2 & 6 Complete
**Next Action**: Begin Phase 3 command handler tests
