# UX Walkthrough Automation

Squiggy includes an automated UX walkthrough system for testing user flows, generating documentation, and validating functionality.

## Overview

The walkthrough system allows you to:

- **Automate user flows**: Systematically trigger commands and UI interactions
- **Capture logs**: Record what happens during each step
- **Validate state**: Check that operations produce expected results
- **Generate reports**: Create comprehensive execution logs
- **Regression testing**: Ensure core workflows remain functional

## Quick Start

### Running a Walkthrough

1. Open Command Palette (`Cmd+Shift+P` or `Ctrl+Shift+P`)
2. Type "Squiggy: Run UX Walkthrough"
3. Select a scenario from the list
4. Click "Run" to execute

The walkthrough will:
- Execute commands automatically
- Log each step and result
- Display progress in the output channel
- Show a summary when complete

### Available Commands

- **Run UX Walkthrough**: Execute a single walkthrough scenario
- **Run All UX Walkthroughs**: Execute all scenarios sequentially
- **List Available Walkthroughs**: Display all available scenarios

## Predefined Scenarios

### Basic Workflow
Tests fundamental operations:
1. Load test data (POD5 + BAM)
2. Refresh read list
3. Verify data loaded
4. Show extension logs

### Session Management
Tests session save/restore:
1. Load demo session
2. Save session state
3. Clear all data
4. Restore session
5. Verify restoration

### Multi-Sample Comparison
Tests sample loading:
1. Load test multi-read dataset
2. Verify multiple samples loaded
3. Refresh UI

### Plot Generation
Tests plotting workflows:
1. Load test data
2. Refresh reads
3. Prepare for plotting

### File Operations
Tests file open/close:
1. Load test data
2. Close BAM file
3. Close POD5 file
4. Verify cleanup

### State Management
Tests state cleanup:
1. Load demo session
2. Clear all state
3. Clear saved session
4. Verify cleanup

## Architecture

### Components

#### WalkthroughLogger (`src/utils/walkthrough-logger.ts`)
- Captures events with timestamps
- Logs commands, state, and validations
- Generates formatted reports
- Exports JSON for analysis

#### WalkthroughRunner (`src/utils/walkthrough-runner.ts`)
- Executes walkthrough scenarios
- Manages command execution
- Handles delays and timing
- Validates state between steps

#### Walkthrough Scenarios (`src/utils/walkthrough-scenarios.ts`)
- Predefined user flow scenarios
- Step-by-step command sequences
- Validation logic
- Reusable test cases

#### Walkthrough Commands (`src/commands/walkthrough-commands.ts`)
- VSCode command registration
- User interface for running walkthroughs
- Scenario selection
- Batch execution

### Data Flow

```
User triggers command
    ↓
WalkthroughRunner loads scenario
    ↓
For each step:
    - WalkthroughLogger logs step start
    - Execute command via VSCode API
    - Capture result
    - Validate state (if specified)
    - WalkthroughLogger logs outcome
    ↓
Generate and display report
```

## Creating Custom Walkthroughs

### Define a Scenario

Create a new scenario in `src/utils/walkthrough-scenarios.ts`:

```typescript
export const myCustomWorkflow: WalkthroughScenario = {
    name: 'My Custom Workflow',
    description: 'Description of what this tests',
    steps: [
        {
            name: 'Step 1: Load data',
            command: 'squiggy.loadTestData',
            description: 'Load test POD5 and BAM',
            delay: 0,
        },
        {
            name: 'Step 2: Wait for load',
            delay: 2000, // Wait 2 seconds
            validate: async (state: ExtensionState) => {
                // Check that data loaded successfully
                return state.getLoadedItems().length > 0;
            },
        },
        {
            name: 'Step 3: Refresh UI',
            command: 'squiggy.refreshReads',
            delay: 500,
        },
    ],
};
```

### Add to Scenario List

Add your scenario to the `allScenarios` array:

```typescript
export const allScenarios: WalkthroughScenario[] = [
    basicWorkflow,
    sessionWorkflow,
    // ... existing scenarios
    myCustomWorkflow, // Add your scenario
];
```

### Step Options

Each step supports:

- **name** (required): Human-readable step name
- **command** (optional): VSCode command to execute
- **args** (optional): Arguments to pass to command
- **delay** (optional): Milliseconds to wait before step
- **validate** (optional): Function to validate state
- **description** (optional): Additional context

### Validation Functions

Validation functions receive the `ExtensionState` and return a boolean:

```typescript
validate: async (state: ExtensionState) => {
    // Check loaded items
    const items = state.getLoadedItems();

    // Check samples
    const samples = state.getAllSampleNames();

    // Check comparison state
    const comparison = state.getComparisonItems();

    // Return true if valid, false otherwise
    return items.length > 0 && samples.length >= 2;
}
```

## Report Format

### Console Output

```
================================================================================
UX Walkthrough Report: Basic Workflow
================================================================================

[0.000s] 🚀 Started: Basic Workflow

[0.015s] Step 1: squiggy.loadTestData
       ✅ Success (1523ms)

[1.538s] Step 2: Validation
       ✓ Check: Data loaded

[1.540s] Step 3: squiggy.refreshReads
       ✅ Success (245ms)

================================================================================
✨ Completed: 3 steps in 1.85s
================================================================================
```

### JSON Export

```json
{
  "name": "Basic Workflow",
  "startTime": 1699564800000,
  "events": [
    {
      "timestamp": 0,
      "step": 0,
      "action": "START",
      "params": { "name": "Basic Workflow" }
    },
    {
      "timestamp": 15,
      "step": 1,
      "action": "COMMAND_START",
      "params": { "command": "squiggy.loadTestData" }
    },
    {
      "timestamp": 1538,
      "step": 1,
      "action": "COMMAND_SUCCESS",
      "params": { "command": "squiggy.loadTestData", "duration": 1523 }
    }
  ]
}
```

## Best Practices

### Timing

- **Add delays**: Commands need time to complete
- **Typical delays**:
  - Data loading: 2000ms
  - UI refresh: 500-1000ms
  - State updates: 500ms
- **Validate after delays**: Check state after operations complete

### Validation

- **Check state changes**: Verify operations succeeded
- **Use specific checks**: Test exact conditions
- **Handle failures gracefully**: Return false, don't throw

### Scenario Design

- **Test one flow per scenario**: Keep focused
- **Add descriptions**: Explain what each step does
- **Include cleanup**: Leave extension in known state
- **Chain operations**: Build complex flows from simple steps

### Debugging

1. **Run single scenario**: Test one flow at a time
2. **Check output channel**: View detailed logs
3. **Add state captures**: Use `runner.captureState()` in custom scenarios
4. **Increase delays**: If commands fail, they may need more time

## Advanced Usage

### Manual Logging

Create a custom walkthrough with manual logging:

```typescript
const runner = new WalkthroughRunner(state, outputChannel);
const logger = runner.getLogger();

logger.start('Custom Walkthrough');

// Manual step
logger.log('CUSTOM_STEP', { data: 'value' });

// Capture state
runner.captureState('After operation');

// Run command with logging
await logger.logCommand('my.command', {}, async () => {
    return await vscode.commands.executeCommand('my.command');
});

logger.end();
console.log(logger.generateReport());
```

### Integration Testing

Use walkthroughs in integration tests:

```typescript
import { WalkthroughRunner } from '../utils/walkthrough-runner';
import { basicWorkflow } from '../utils/walkthrough-scenarios';

test('Basic workflow completes successfully', async () => {
    const runner = new WalkthroughRunner(state, outputChannel);
    await runner.run(basicWorkflow);

    const summary = runner.getLogger().getSummary();
    expect(summary.errorCount).toBe(0);
    expect(summary.successCount).toBeGreaterThan(0);
});
```

### Continuous Integration

Run walkthroughs in CI/CD:

```bash
# Run extension tests including walkthroughs
npm run test

# Run specific walkthrough tests
npm run test -- walkthrough
```

## Troubleshooting

### Commands Not Found
- Ensure extension is activated
- Check command registration in `package.json`
- Verify command implementation exists

### Validation Failures
- Check timing - add more delay
- Verify state changes are complete
- Use `captureState()` to debug

### Timeout Errors
- Increase delays between steps
- Check if commands are actually completing
- Review output channel for errors

### State Inconsistencies
- Clear state before running (`squiggy.clearState`)
- Run scenarios in isolation
- Check for race conditions

## Future Enhancements

Potential improvements:

- **Interactive mode**: Pause between steps for manual verification
- **Screenshot capture**: Visual regression testing
- **Performance metrics**: Track timing trends
- **Parallel execution**: Run independent scenarios concurrently
- **Custom assertions**: Rich validation DSL
- **Webview interactions**: Simulate button clicks, form inputs
- **Network mocking**: Test with simulated data

## Contributing

To add new walkthrough scenarios:

1. Define scenario in `walkthrough-scenarios.ts`
2. Add to `allScenarios` array
3. Test manually first
4. Add validation checks
5. Document in this file
6. Submit PR with test results

## Resources

- **Walkthrough Logger**: `src/utils/walkthrough-logger.ts`
- **Walkthrough Runner**: `src/utils/walkthrough-runner.ts`
- **Scenarios**: `src/utils/walkthrough-scenarios.ts`
- **Commands**: `src/commands/walkthrough-commands.ts`
- **Tests**: `src/utils/__tests__/walkthrough-logger.test.ts`
