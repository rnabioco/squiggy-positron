/**
 * Base Modifications Panel Webview View
 *
 * Displays base modification information from BAM files with MM/ML tags
 */

import * as vscode from 'vscode';

export class ModificationsPanelProvider implements vscode.WebviewViewProvider {
    public static readonly viewType = 'squiggyModificationsPanel';

    private _view?: vscode.WebviewView;
    private _hasModifications: boolean = false;
    private _modificationTypes: string[] = [];
    private _hasProbabilities: boolean = false;
    private _minProbability: number = 0.5;
    private _enabledModTypes: Set<string> = new Set();

    // Event emitter for when filter options change
    private _onDidChangeFilters = new vscode.EventEmitter<void>();
    public readonly onDidChangeFilters = this._onDidChangeFilters.event;

    constructor(private readonly _extensionUri: vscode.Uri) {}

    public resolveWebviewView(
        webviewView: vscode.WebviewView,
        _context: vscode.WebviewViewResolveContext,
        _token: vscode.CancellationToken
    ) {
        this._view = webviewView;

        webviewView.webview.options = {
            enableScripts: true,
            localResourceRoots: [this._extensionUri],
        };

        webviewView.webview.html = this._getHtmlForWebview(webviewView.webview);

        // Handle messages from the webview
        webviewView.webview.onDidReceiveMessage((data) => {
            switch (data.type) {
                case 'probabilityChanged':
                    this._minProbability = data.value;
                    this._onDidChangeFilters.fire();
                    break;
                case 'modTypeToggled':
                    if (data.enabled) {
                        this._enabledModTypes.add(data.modType);
                    } else {
                        this._enabledModTypes.delete(data.modType);
                    }
                    this._onDidChangeFilters.fire();
                    break;
            }
        });

        // If we already have modification data, send it to the newly created webview
        if (this._hasModifications) {
            this._updateView();
        }
    }

    /**
     * Update modification info display
     */
    public setModificationInfo(
        hasModifications: boolean,
        modificationTypes: string[],
        hasProbabilities: boolean
    ) {
        this._hasModifications = hasModifications;
        this._modificationTypes = modificationTypes;
        this._hasProbabilities = hasProbabilities;

        // Initialize all modification types as enabled by default
        this._enabledModTypes = new Set(modificationTypes);

        this._updateView();
    }

    /**
     * Get current filter settings
     */
    public getFilters() {
        return {
            minProbability: this._minProbability,
            enabledModTypes: Array.from(this._enabledModTypes),
        };
    }

    /**
     * Clear modification info (when no BAM loaded or BAM has no modifications)
     */
    public clear() {
        this._hasModifications = false;
        this._modificationTypes = [];
        this._hasProbabilities = false;
        this._updateView();
    }

    private _updateView() {
        if (this._view) {
            this._view.webview.postMessage({
                type: 'updateModificationInfo',
                hasModifications: this._hasModifications,
                modificationTypes: this._modificationTypes,
                hasProbabilities: this._hasProbabilities,
            });
        }
    }

    private _getHtmlForWebview(_webview: vscode.Webview) {
        // Map common modification codes to names
        const modCodeToName: Record<string, string> = {
            // Single-letter codes
            m: '5-methylcytosine (5mC)',
            h: '5-hydroxymethylcytosine (5hmC)',
            a: '6-methyladenine (6mA)',
            o: '8-oxoguanine (8-oxoG)',
            // ChEBI codes (common RNA modifications)
            '17596': 'Inosine (I)',
            '28177': '1-methyladenosine (m1A)',
            '21863': '1-methylguanosine (m1G)',
            '28527': '7-methylguanosine (m7G)',
            '17802': 'Pseudouridine (Ψ)',
            '27301': '5-methyluridine (m5U)',
            '18421': 'Dihydrouridine (D)',
        };

        // Modification colors (shades matching base colors from eventalign view)
        // Each modification uses shades within the same color family as its canonical base
        // BASE_COLORS: C=#F0E442 (yellow), A=#009E73 (green), G=#0072B2 (blue), T/U=#D55E00 (orange)
        const modificationColors: Record<string, string> = {
            // Cytosine modifications (yellow family - C=#F0E442)
            m: '#F0E442', // 5mC - base yellow
            h: '#E6D835', // 5hmC - dark yellow
            f: '#DCC728', // 5fC - darker yellow
            c: '#FFF78A', // 5caC - light yellow
            '21839': '#FFFC9E', // 4mC - very light yellow
            '19228': '#D4BC1F', // Cm - deep yellow
            C: '#F0E442', // any C* - base yellow
            // Adenine modifications (green family - A=#009E73)
            a: '#009E73', // 6mA - base green
            '17596': '#00C490', // I - light green
            '69426': '#007A57', // Am - dark green
            A: '#009E73', // any A* - base green
            // Guanine modifications (blue family - G=#0072B2)
            o: '#0072B2', // 8oxoG - base blue
            '19229': '#4DA6E0', // Gm - light blue
            G: '#0072B2', // any G* - base blue
            // Thymine/Uracil modifications (orange family - T/U=#D55E00)
            g: '#D55E00', // 5hmU - base orange
            e: '#FF7518', // 5fU - light orange
            b: '#B34C00', // 5caU - dark orange
            '17802': '#FF9447', // Ψ - lighter orange
            '16450': '#8F3D00', // dU - deep orange
            '19227': '#FFB880', // Um - very light orange
            T: '#D55E00', // any T* - base orange
            // Default
            default: '#000000',
        };

        return `<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Base Modifications</title>
    <style>
        body {
            padding: 10px;
            font-family: var(--vscode-font-family);
            font-size: var(--vscode-font-size);
            color: var(--vscode-foreground);
        }
        .no-data {
            color: var(--vscode-descriptionForeground);
            font-style: italic;
            padding: 10px;
            text-align: center;
        }
        .section-label {
            font-weight: bold;
            font-size: 0.85em;
            color: var(--vscode-descriptionForeground);
            margin-bottom: 4px;
        }
        .mod-code {
            font-family: var(--vscode-editor-font-family);
            color: var(--vscode-descriptionForeground);
            font-size: 0.85em;
        }
        .mod-name {
            color: var(--vscode-foreground);
        }
        .filter-section {
            margin-bottom: 16px;
            padding-bottom: 12px;
            border-bottom: 1px solid var(--vscode-panel-border);
        }
        .slider-container {
            margin: 8px 0;
        }
        .slider-label {
            display: flex;
            justify-content: space-between;
            font-size: 0.85em;
            margin-bottom: 4px;
            align-items: center;
        }
        .slider-value {
            font-weight: bold;
        }
        input[type="range"] {
            width: 100%;
            margin: 4px 0;
        }
        .checkbox-item {
            display: flex;
            align-items: center;
            padding: 4px 0;
        }
        .checkbox-item input[type="checkbox"] {
            margin-right: 8px;
        }
        .checkbox-item label {
            flex: 1;
            cursor: pointer;
            font-size: 0.9em;
            display: flex;
            align-items: center;
        }
        .mod-color-square {
            display: inline-block;
            width: 12px;
            height: 12px;
            margin-right: 6px;
            border-radius: 2px;
            flex-shrink: 0;
        }
    </style>
</head>
<body>
    <div id="content">
        <div class="no-data">
            No BAM file loaded or no modifications detected
        </div>
    </div>

    <script>
        const vscode = acquireVsCodeApi();

        const modCodeToName = ${JSON.stringify(modCodeToName)};
        const modificationColors = ${JSON.stringify(modificationColors)};

        // Receive updates from extension
        window.addEventListener('message', event => {
            const message = event.data;
            switch (message.type) {
                case 'updateModificationInfo':
                    updateDisplay(message);
                    break;
            }
        });

        function updateDisplay(data) {
            const content = document.getElementById('content');

            if (!data.hasModifications || data.modificationTypes.length === 0) {
                content.innerHTML = '<div class="no-data">No BAM file loaded or no modifications detected</div>';
                return;
            }

            let html = '';

            // Probability filter section (only show if probabilities are present)
            if (data.hasProbabilities) {
                html += '<div class="filter-section">';
                html += '<div class="section-label">Minimum Probability Threshold</div>';
                html += '<div class="slider-container">';
                html += '<div class="slider-label">';
                html += '<span>0.00</span>';
                html += '<span id="probValue" class="slider-value">0.50</span>';
                html += '<span>1.00</span>';
                html += '</div>';
                html += '<input type="range" id="probSlider" min="0" max="100" value="50" step="1">';
                html += '</div>';
                html += '</div>';
            }

            // Modification type filters
            html += '<div class="filter-section">';
            html += '<div class="section-label">Available Modifications</div>';

            for (const code of data.modificationTypes) {
                let name = modCodeToName[code];
                let displayCode = code;

                // If code is numeric and we don't have a name, show it as ChEBI code
                if (!name && /^[0-9]+$/.test(code)) {
                    displayCode = 'ChEBI:' + code;
                    name = 'Unknown modification';
                } else if (!name) {
                    name = 'Unknown modification';
                }

                // Get color for this modification
                const color = modificationColors[code] || modificationColors.default;

                html += '<div class="checkbox-item">';
                html += '<input type="checkbox" id="mod_' + code + '" data-modtype="' + code + '" checked>';
                html += '<label for="mod_' + code + '">';
                html += '<span class="mod-color-square" style="background-color: ' + color + ';"></span>';
                html += '<span class="mod-name">' + name + '</span> ';
                html += '<span class="mod-code">(' + displayCode + ')</span>';
                html += '</label>';
                html += '</div>';
            }

            html += '</div>';

            content.innerHTML = html;

            // Attach event listeners after HTML is set
            if (data.hasProbabilities) {
                const probSlider = document.getElementById('probSlider');
                const probValue = document.getElementById('probValue');

                // Update display live while dragging
                probSlider.addEventListener('input', (e) => {
                    const value = parseInt(e.target.value) / 100;
                    probValue.textContent = value.toFixed(2);
                });

                // Only trigger plot update when slider is released
                probSlider.addEventListener('change', (e) => {
                    const value = parseInt(e.target.value) / 100;
                    vscode.postMessage({
                        type: 'probabilityChanged',
                        value: value
                    });
                });
            }

            // Attach checkbox listeners
            for (const code of data.modificationTypes) {
                const checkbox = document.getElementById('mod_' + code);
                checkbox.addEventListener('change', (e) => {
                    vscode.postMessage({
                        type: 'modTypeToggled',
                        modType: e.target.dataset.modtype,
                        enabled: e.target.checked
                    });
                });
            }
        }
    </script>
</body>
</html>`;
    }
}
