/**
 * SessionCore Component
 *
 * Top-level React component for the session manager panel.
 * Manages session state and handles communication with the extension host.
 */

import * as React from 'react';
import { vscode } from './vscode-api';
import './squiggy-session-core.css';

interface SessionState {
    hasSamples: boolean;
    hasSavedSession: boolean;
    sampleCount: number;
    sampleNames: string[];
}

export const SessionCore: React.FC = () => {
    const [state, setState] = React.useState<SessionState>({
        hasSamples: false,
        hasSavedSession: false,
        sampleCount: 0,
        sampleNames: [],
    });

    const [isDemoExpanded, setIsDemoExpanded] = React.useState<boolean>(false);

    // Handle messages from extension
    React.useEffect(() => {
        const messageHandler = (event: MessageEvent) => {
            const message = event.data;

            if (message.type === 'updateSession') {
                setState({
                    hasSamples: message.hasSamples,
                    hasSavedSession: message.hasSavedSession,
                    sampleCount: message.sampleCount,
                    sampleNames: message.sampleNames,
                });
            }
        };

        window.addEventListener('message', messageHandler);

        // Notify extension that webview is ready
        vscode.postMessage({ type: 'ready' });

        return () => window.removeEventListener('message', messageHandler);
    }, []);

    const handleLoadDemo = () => {
        vscode.postMessage({ type: 'loadDemo' });
    };

    const handleSave = () => {
        vscode.postMessage({ type: 'save' });
    };

    const handleRestore = () => {
        vscode.postMessage({ type: 'restore' });
    };

    const handleExport = () => {
        vscode.postMessage({ type: 'export' });
    };

    const handleImport = () => {
        vscode.postMessage({ type: 'import' });
    };

    const handleClear = () => {
        vscode.postMessage({ type: 'clear' });
    };

    return (
        <div className="session-container">
            {/* Demo Session Card */}
            <div className={`demo-card ${!isDemoExpanded ? 'collapsed' : ''}`}>
                <div
                    className="demo-title"
                    onClick={() => setIsDemoExpanded(!isDemoExpanded)}
                    style={{ cursor: 'pointer' }}
                >
                    <span className="icon">{isDemoExpanded ? '▼' : '▶'}</span>
                    <span className="icon">🚀</span>
                    <span>Try Demo Data</span>
                </div>
                {isDemoExpanded && (
                    <>
                        <p className="demo-description">
                            Explore 180 yeast tRNA reads with base annotations
                        </p>
                        <button className="btn btn-primary btn-full" onClick={handleLoadDemo}>
                            <span className="icon">▶</span> Load Demo Session
                        </button>
                        <div className="demo-info">
                            <span className="icon">ℹ️</span> Uses packaged test data - no files
                            needed!
                        </div>
                    </>
                )}
            </div>

            {/* Current Session Status */}
            <div className="section">
                <h4 className="section-title">Current Session</h4>
                <div className="status-box">
                    {state.hasSamples ? (
                        <>
                            <span className="icon">✓</span> {state.sampleCount} sample(s) loaded:{' '}
                            {state.sampleNames.join(', ')}
                        </>
                    ) : (
                        <>
                            <span className="icon">○</span> No data loaded
                        </>
                    )}
                </div>
            </div>

            {/* Session Actions */}
            <div className="section">
                <h4 className="section-title">Actions</h4>
                <div className="btn-grid">
                    <button className="btn" onClick={handleSave} disabled={!state.hasSamples}>
                        <span className="icon">💾</span> Save Positron
                    </button>
                    <button
                        className="btn btn-secondary"
                        onClick={handleRestore}
                        disabled={!state.hasSavedSession}
                    >
                        <span className="icon">📜</span> Restore
                    </button>
                    <button
                        className="btn btn-secondary"
                        onClick={handleExport}
                        disabled={!state.hasSamples}
                    >
                        <span className="icon">📄</span> Export JSON
                    </button>
                    <button className="btn btn-secondary" onClick={handleImport}>
                        <span className="icon">📥</span> Import
                    </button>
                    <button
                        className="btn btn-secondary"
                        onClick={handleClear}
                        disabled={!state.hasSavedSession}
                    >
                        <span className="icon">🗑️</span> Clear
                    </button>
                </div>
            </div>
        </div>
    );
};
