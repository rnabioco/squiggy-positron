# UI Development Guidelines for Squiggy

This document contains important patterns and best practices for UI development with PySide6/Qt in the Squiggy project.

## QCheckBox Signal Handling

### ✅ CORRECT: Use `toggled` Signal

Always use the `toggled` signal which passes a clean boolean value:

```python
# Create checkbox
self.my_checkbox = QCheckBox("My Option")
self.my_checkbox.setChecked(True)

# Connect to toggled signal (passes bool: True/False)
self.my_checkbox.toggled.connect(self.on_my_checkbox_toggled)

# Handler receives a simple bool
def on_my_checkbox_toggled(self, checked):
    if checked:
        # Do something when checked
        pass
    else:
        # Do something when unchecked
        pass
```

### ❌ WRONG: Don't Use `stateChanged` Signal

The `stateChanged` signal passes a Qt.CheckState enum (0, 1, 2) which is error-prone:

```python
# DON'T DO THIS
self.my_checkbox.stateChanged.connect(self.on_state_changed)

# Handler receives an int that needs complex comparison
def on_state_changed(self, state):
    # state is 0 (unchecked), 1 (partially checked), or 2 (checked)
    is_checked = state == 2  # Fragile and confusing!
```

### Key Points

- **`toggled`** → Clean `bool` (True/False) - USE THIS
- **`stateChanged`** → Messy `int` enum value (0/1/2) - AVOID THIS
- Only use `stateChanged` if you specifically need tri-state checkbox support
- The `toggled` signal works perfectly for standard checkboxes and is much more readable

## Signal Loop Prevention

When updating UI controls programmatically that might trigger their own signals (creating infinite loops), use `blockSignals()`:

```python
def update_checkbox_from_code(self, new_state):
    # Block signals to prevent recursive loops
    self.my_checkbox.blockSignals(True)
    self.my_checkbox.setChecked(new_state)
    self.my_checkbox.blockSignals(False)
```

## Async Signal Handlers with qasync

For async slot handlers, use the `@qasync.asyncSlot()` decorator and connect with `asyncio.ensure_future()`:

```python
# Connect signal to async handler
self.my_checkbox.toggled.connect(
    lambda checked: asyncio.ensure_future(self.on_checkbox_toggled_async(checked))
)

# Async handler
@qasync.asyncSlot()
async def on_checkbox_toggled_async(self, checked):
    # Can use await here
    await some_async_operation()
```

## Additional Qt Patterns

### Radio Buttons

Use QButtonGroup to manage exclusive selection:

```python
self.button_group = QButtonGroup()
self.button_group.addButton(self.radio1)
self.button_group.addButton(self.radio2)
```

### ComboBox

Use `currentIndexChanged` or `currentTextChanged` signals:

```python
self.combo.currentIndexChanged.connect(self.on_combo_changed)

def on_combo_changed(self, index):
    # index is an int
    pass
```

### Sliders and SpinBoxes

Use `valueChanged` signal which passes the new value directly:

```python
self.slider.valueChanged.connect(self.on_slider_changed)

def on_slider_changed(self, value):
    # value is an int (for sliders/spinboxes)
    pass
```
